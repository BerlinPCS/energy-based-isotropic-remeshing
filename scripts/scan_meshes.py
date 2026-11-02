#!/usr/bin/env python3
"""Measures on the meshes behind the numbers of Ch. 4 that remesher-cli does not report.

scans.csv has one row per run of the main comparison, and one per baseline run of Fig. iteration-comparison:
    long_fraction, short_fraction   edges longer than 4/3 or shorter than 4/5 of the target
    split_*, collapse_*             for dynamic queue, hybrid and baseline: what rejects the split or collapse of those edges
    long_edges, cap_sequences       for the two queues: long edges whose flip and following collapse would pass (Fig. horse-cap)

inputs.csv has one row per input mesh: its size, edge statistics and the area share thinner than each target.
horse-cap.json has the numbers of Fig. horse-cap.

The checks are written again here on the stored meshes, after validity.cpp and evaluate.cpp, since they only look at
the final mesh. A queue that ends empty leaves no candidate that passes both, so split_passes_own and collapse_passes_own
must be 0 on those runs, which guards this copy against drifting from the code.

usage: scan_meshes.py MESHES_DIR OUT_DIR
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ELL_MAX_RATIO, ELL_MIN_RATIO = 4 / 3, 0.8
COSINE_LIMIT, FLOAT_EPSILON = 0.5, 1.1920929e-7
# The code computes the normal test in float, so a new triangle in the plane of the most deviating old one (a tie at the
# bound) can go either way there. Such a tie counts as failing here.
COSINE_TIE = 1e-6
OBJECTIVES = ['sum', 'maximum', 'local-average', 'global-average']
TARGETS = [0.75, 1, 1.5]
STUCK_APPROACHES = ['dynamic-queue', 'hybrid', 'baseline']
CAP_APPROACHES = ['dynamic-queue', 'batched']
# The long edge of Fig. horse-cap in the output of batched, local average, T = 1.
HORSE_CAP_EDGE = (7898, 21556)
HORSE_CAP_RUN = 'horse__batched__local-average__T1'
# The baseline runs of Fig. iteration-comparison, whose edge shares the text quotes.
ITERATION_FIGURE_MESH, ITERATION_FIGURE_CYCLES = 'stanford-bunny', [1, 5, 25]


def loss(ratio):
    return np.log2(ratio) ** 2


def load_obj(path):
    vertices, faces = [], []
    with open(path) as file:
        for line in file:
            if line.startswith('v '):
                vertices.append(line.split()[1:4])
            elif line.startswith('f '):
                faces.append([int(part.split('/')[0]) - 1 for part in line.split()[1:4]])
    return np.asarray(vertices, float), np.asarray(faces, int)


def unique_edges(faces):
    return np.unique(np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1), axis=0)


def count_elements(path):
    vertices = faces = 0
    with open(path) as file:
        for line in file:
            vertices += line.startswith('v ')
            faces += line.startswith('f ')
    return vertices, faces, faces * 3 // 2


def triangle_area_vector(points):
    return np.cross(points[1] - points[0], points[2] - points[0]) / 2


def passes(before, after, mean_loss, edge_count, epsilon):
    """The acceptance test of every objective for the losses of the affected edges, and the gain of the sum."""
    gain = sum(before) - sum(after)
    change = len(after) - len(before)
    total = mean_loss * edge_count
    return {
        'sum': gain > epsilon,
        'maximum': max(before) - max(after) > epsilon,
        'local-average': sum(before) / len(before) - sum(after) / len(after) > epsilon / len(after),
        'global-average': mean_loss - (total - gain) / (edge_count + change) > epsilon / (edge_count + change),
    }, gain


def normal_test_passes(old, new):
    """The ring-bounded normal test of collapses, against the summed area normal of the replaced triangles."""
    reference = old.sum(axis=0)
    reference_norm = np.linalg.norm(reference)

    def cosine(normal):
        norm = np.linalg.norm(normal)
        return 0.0 if norm <= FLOAT_EPSILON * reference_norm else normal @ reference / (norm * reference_norm)

    outside = sum(cosine(normal) < COSINE_LIMIT for normal in new)
    if outside == 0:
        return True
    if outside > sum(cosine(normal) < COSINE_LIMIT for normal in old):
        return False
    bound = min([COSINE_LIMIT] + [cosine(normal) for normal in old])
    return all(np.linalg.norm(normal) > FLOAT_EPSILON * reference_norm and cosine(normal) >= bound + COSINE_TIE
               for normal in new)


def flip_normal_passes(old, new):
    reference_norm, norm = np.linalg.norm(old), np.linalg.norm(new)
    return reference_norm > FLOAT_EPSILON and norm > FLOAT_EPSILON * reference_norm and \
        new @ old / (reference_norm * norm) >= COSINE_LIMIT


class StoredMesh:
    def __init__(self, path, summary):
        self.v, self.f = load_obj(path)
        self.target = summary['target_length']
        self.epsilon = summary['epsilon']
        self.vertex_faces = [[] for _ in self.v]
        self.neighbors = [set() for _ in self.v]
        self.opposite = {}
        for index, face in enumerate(self.f):
            for k in range(3):
                a, b, c = int(face[k]), int(face[(k + 1) % 3]), int(face[(k + 2) % 3])
                self.vertex_faces[a].append(index)
                self.neighbors[a].add(b)
                self.opposite[(a, b)] = (c, index)
        self.edges = np.array(sorted((a, b) for a, b in self.opposite if a < b))
        self.ratio = np.linalg.norm(self.v[self.edges[:, 0]] - self.v[self.edges[:, 1]], axis=1) / self.target
        self.losses = loss(self.ratio)
        self.mean = float(self.losses.mean())
        self.edge_count = len(self.edges)

    def length(self, a, b):
        return float(np.linalg.norm(self.v[a] - self.v[b]) / self.target)

    def edge_loss(self, a, b):
        return float(loss(self.length(a, b)))

    def split_candidate(self, a, b):
        middle = (self.v[a] + self.v[b]) / 2
        spokes = [float(np.linalg.norm(middle - self.v[self.opposite[pair][0]]) / self.target)
                  for pair in ((a, b), (b, a))]
        half = self.length(a, b) / 2
        return passes([self.edge_loss(a, b)], [float(loss(half))] * 2 + [float(loss(s)) for s in spokes], self.mean,
                      self.edge_count, self.epsilon), spokes

    def collapse_candidate(self, a, b):
        """(acceptance per objective, what makes it invalid or 'yes') of the midpoint collapse of edge (a, b)."""
        v, f = self.v, self.f
        middle = (v[a] + v[b]) / 2
        ring = sorted((self.neighbors[a] | self.neighbors[b]) - {a, b})
        new_lengths = np.linalg.norm(v[ring] - middle, axis=1) / self.target
        before = [self.edge_loss(a, b)] + [self.edge_loss(a, u) for u in self.neighbors[a] - {b}] + \
            [self.edge_loss(b, u) for u in self.neighbors[b] - {a}]
        accepted, _ = passes(before, [float(x) for x in loss(new_lengths)], self.mean, self.edge_count, self.epsilon)
        apexes = {self.opposite[(a, b)][0], self.opposite[(b, a)][0]}
        if new_lengths.max() > ELL_MAX_RATIO:
            return accepted, 'length'
        if self.neighbors[a] & self.neighbors[b] != apexes:
            return accepted, 'link'
        shared = set(self.vertex_faces[a]) & set(self.vertex_faces[b])
        old = np.array([triangle_area_vector(v[f[t]]) for t in set(self.vertex_faces[a]) | set(self.vertex_faces[b])])
        new = np.array([triangle_area_vector(np.where((f[t] == x)[:, None], middle, v[f[t]]))
                        for x in (a, b) for t in self.vertex_faces[x] if t not in shared])
        return accepted, 'yes' if normal_test_passes(old, new) else 'normal'

    def flip_quad(self, a, c):
        (b, face_1), (d, face_2) = self.opposite[(a, c)], self.opposite[(c, a)]
        return b, d, face_1, face_2, [[b, a, d], [d, c, b]]

    def cap_sequence(self, a, c):
        """Flip (a, c), then collapse the new diagonal. None if the flip is invalid or the new diagonal is not short."""
        v, f = self.v, self.f
        b, d, face_1, face_2, new_faces = self.flip_quad(a, c)
        if b == d or d in self.neighbors[b]:
            return None
        old = triangle_area_vector(v[f[face_1]]) + triangle_area_vector(v[f[face_2]])
        if not all(flip_normal_passes(old, triangle_area_vector(v[t])) for t in new_faces):
            return None
        diagonal = self.length(b, d)
        if diagonal >= ELL_MIN_RATIO:
            return None
        flip_gain = self.edge_loss(a, c) - self.edge_loss(b, d)
        flip_accepted = passes([self.edge_loss(a, c)], [self.edge_loss(b, d)], self.mean, self.edge_count, self.epsilon)[0]
        # The mesh after the flip: b and d gain each other, a and c lose each other.
        neighbors_b, neighbors_d = self.neighbors[b] | {d}, self.neighbors[d] | {b}
        mean = (self.mean * self.edge_count - flip_gain) / self.edge_count
        middle = (v[b] + v[d]) / 2
        ring = sorted((neighbors_b | neighbors_d) - {b, d})
        new_lengths = np.linalg.norm(v[ring] - middle, axis=1) / self.target
        before = [self.edge_loss(b, d)] + [self.edge_loss(b, u) for u in neighbors_b - {d}] + \
            [self.edge_loss(d, u) for u in neighbors_d - {b}]
        collapse_accepted, collapse_gain = passes(before, [float(x) for x in loss(new_lengths)], mean, self.edge_count,
                                                  self.epsilon)
        remaining = [f[i].tolist() for x in (b, d) for i in self.vertex_faces[x] if i not in (face_1, face_2)]
        old_normals = [triangle_area_vector(v[t]) for t in remaining + new_faces]
        moved = [triangle_area_vector(np.array([middle if x in (b, d) else v[x] for x in t])) for t in remaining]
        return {
            'b': b, 'd': d, 'ac': self.length(a, c), 'bd': diagonal, 'flip_gain': flip_gain,
            'flip_accepted': flip_accepted, 'collapse_gain': collapse_gain, 'collapse_accepted': collapse_accepted,
            'net_gain': flip_gain + collapse_gain,
            'valid': bool(new_lengths.max() <= ELL_MAX_RATIO and (neighbors_b & neighbors_d) == {a, c} and
                          normal_test_passes(np.array(old_normals), np.array(moved))),
        }


def scan_run(job):
    """One row of scans.csv."""
    folder, tier, cycles, mesh, approach, objective, target, stop_reason = job
    summary = json.loads((folder / 'summary.json').read_text())
    stored = StoredMesh(folder / 'remeshed.obj', summary)
    row = {'tier': tier, 'cycles': cycles, 'input': mesh, 'approach': approach, 'objective': objective, 'target': target,
           'long_fraction': float(np.mean(stored.ratio > ELL_MAX_RATIO)),
           'short_fraction': float(np.mean(stored.ratio < ELL_MIN_RATIO))}
    long_edges = [(int(a), int(b)) for (a, b), r in zip(stored.edges, stored.ratio) if r > ELL_MAX_RATIO]
    short_edges = [(int(a), int(b)) for (a, b), r in zip(stored.edges, stored.ratio) if r < ELL_MIN_RATIO]

    if tier != 'core':
        return row

    if approach in STUCK_APPROACHES:
        split_results = [stored.split_candidate(a, b) for a, b in long_edges]
        collapse_results = [stored.collapse_candidate(a, b) for a, b in short_edges]
        row['split_candidates'] = len(long_edges)
        row['collapse_candidates'] = len(short_edges)
        row['collapse_invalid_length'] = sum(valid == 'length' for _, valid in collapse_results)
        row['collapse_invalid_link'] = sum(valid == 'link' for _, valid in collapse_results)
        row['collapse_invalid_normal'] = sum(valid == 'normal' for _, valid in collapse_results)
        if approach != 'baseline':
            row['split_passes_own'] = sum(bool(accepted[objective]) for (accepted, _), _ in split_results)
            row['collapse_passes_own'] = sum(bool(accepted[objective]) and valid == 'yes'
                                             for accepted, valid in collapse_results)
            if stop_reason == 'no-candidate' and (row['split_passes_own'] or row['collapse_passes_own']):
                raise RuntimeError(f'{folder.name}: a candidate passes the checks of this scan, but the queue ended empty')

    if approach in CAP_APPROACHES:
        sequences = [stored.cap_sequence(a, c) for a, c in long_edges]
        row['long_edges'] = len(long_edges)
        row['cap_sequences'] = sum(bool(s and s['valid'] and s['collapse_accepted'][objective] and s['net_gain'] > 0)
                                   for s in sequences)
    return row


def horse_cap_numbers(out_dir):
    folder = out_dir / 'core' / HORSE_CAP_RUN
    summary = json.loads((folder / 'summary.json').read_text())
    stored = StoredMesh(folder / 'remeshed.obj', summary)
    a, c = HORSE_CAP_EDGE
    sequence = stored.cap_sequence(a, c)
    if not sequence or not sequence['valid']:
        raise RuntimeError('the edge of Fig. horse-cap is no longer the base of a valid sequence')
    b, d = sequence['b'], sequence['d']
    (split_accepted, split_gain), spokes = stored.split_candidate(a, c)
    v = stored.v

    def angle(p, apex, q):
        u, w = v[p] - v[apex], v[q] - v[apex]
        return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(u, w)), u @ w)))

    return {
        'target_length': stored.target, 'vertices': {'A': a, 'B': b, 'C': c, 'D': d}, 'edge_ac': sequence['ac'],
        'edge_bd': sequence['bd'], 'split_spokes': spokes, 'split_gain_sum': split_gain,
        'split_passes': {k: bool(x) for k, x in split_accepted.items()}, 'flip_gain_sum': sequence['flip_gain'],
        'flip_passes': {k: bool(x) for k, x in sequence['flip_accepted'].items()},
        'collapse_gain_sum': sequence['collapse_gain'],
        'collapse_passes': {k: bool(x) for k, x in sequence['collapse_accepted'].items()},
        'net_gain_sum': sequence['net_gain'], 'apex_angles_deg': [angle(a, b, c), angle(a, d, c)],
    }


def thickness(vertices, faces, reach):
    """Distance from each face's centroid along its inner normal to the nearest other face; inf beyond `reach`."""
    corners = vertices[faces]
    centroids = corners.mean(axis=1)
    cross = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    doubled_area = np.linalg.norm(cross, axis=1)
    normals = cross / np.maximum(doubled_area[:, None], 1e-300)
    longest = np.linalg.norm(corners - np.roll(corners, 1, axis=1), axis=2).max()
    distances = np.full(len(faces), np.inf)
    for i, near in enumerate(cKDTree(centroids).query_ball_point(centroids, reach + longest)):
        others = np.asarray([j for j in near if j != i])
        if not len(others):
            continue
        # Moeller-Trumbore from the centroid along -normal.
        a = corners[others, 0]
        e1, e2 = corners[others, 1] - a, corners[others, 2] - a
        direction = -normals[i]
        h = np.cross(direction, e2)
        determinant = np.einsum('ij,ij->i', e1, h)
        valid = np.abs(determinant) > 1e-15
        determinant = np.where(valid, determinant, 1)
        s = centroids[i] - a
        u = np.einsum('ij,ij->i', s, h) / determinant
        q = np.cross(s, e1)
        w = q @ direction / determinant
        t = np.einsum('ij,ij->i', e2, q) / determinant
        hit = valid & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 1e-9)
        if hit.any():
            distances[i] = t[hit].min()
    return doubled_area / 2, distances


def scan_input(path):
    vertices, faces = load_obj(path)
    edges = unique_edges(faces)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    average = float(lengths.mean())
    area, distances = thickness(vertices, faces, max(TARGETS) * average)
    # Closed and connected (Sec. 3.1.1), so Euler's formula gives the genus.
    genus = (2 - (len(vertices) - len(edges) + len(faces))) // 2
    row = {'mesh': path.stem, 'vertices': len(vertices), 'edges': len(edges), 'faces': len(faces), 'genus': genus,
           'average_edge_length': average, 'edge_length_cv': float(lengths.std() / average),
           'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    for target in TARGETS:
        row[f'thin_area_share_T{target:g}'] = float(area[distances < target * average].sum() / area.sum())
    return row


def scan(meshes_dir, out_dir):
    results = pd.read_csv(out_dir / 'results.csv', keep_default_na=False)
    core = results[results.tier == 'core']
    jobs = []
    for run in core.itertuples():
        name = '__'.join([run.input, run.approach, *([run.objective] if run.objective else []), f'T{float(run.target):g}'])
        jobs.append((out_dir / 'core' / name, 'core', '', run.input, run.approach, run.objective, float(run.target),
                     run.stop_reason))
    figure_runs = results[(results.tier == 'cycles') & (results.input == ITERATION_FIGURE_MESH) &
                          results.cycles.astype(int).isin(ITERATION_FIGURE_CYCLES)]
    for run in figure_runs.itertuples():
        name = f'{run.input}__{run.approach}__T{float(run.target):g}__k{int(run.cycles):02d}'
        jobs.append((out_dir / 'cycles' / name, 'cycles', int(run.cycles), run.input, run.approach, '',
                     float(run.target), run.stop_reason))
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(scan_run, jobs, chunksize=4))
        input_rows = list(pool.map(scan_input, sorted(meshes_dir / f'{mesh}.obj' for mesh in core.input.unique())))
    pd.DataFrame(rows).to_csv(out_dir / 'scans.csv', index=False)
    pd.DataFrame(input_rows).to_csv(out_dir / 'inputs.csv', index=False)
    if (out_dir / 'core' / HORSE_CAP_RUN).exists():
        (out_dir / 'horse-cap.json').write_text(json.dumps(horse_cap_numbers(out_dir), indent=2) + '\n')
    print(len(rows), 'runs in scans.csv,', len(input_rows), 'meshes in inputs.csv', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('meshes_dir', type=Path)
    parser.add_argument('out_dir', type=Path)
    args = parser.parse_args()
    scan(args.meshes_dir, args.out_dir)


if __name__ == '__main__':
    main()
