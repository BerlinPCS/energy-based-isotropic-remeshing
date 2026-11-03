#!/usr/bin/env python3
"""Edge measures on the meshes behind the numbers of Ch. 1 and 4 that remesher-cli does not report.

scans.csv has one row per run of the main comparison, and one per baseline run of Fig. iteration-comparison:
    long_fraction, short_fraction   edges longer than 4/3 or shorter than 4/5 of the target
    collapse_candidates             (main comparison only) the short edges, which a collapse would remove ...
    collapse_invalid_length         ... and those whose collapse the length check of validity.cpp blocks: an edge from
                                    the midpoint to the ring would be longer than 4/3 of the target

inputs.csv has one row per input mesh: its size and edge statistics.

usage: scan_meshes.py MESHES_DIR OUT_DIR
"""
import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import pandas as pd

ELL_MAX_RATIO, ELL_MIN_RATIO = 4 / 3, 0.8
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


def scan_run(job):
    """One row of scans.csv."""
    folder, tier, cycles, mesh, approach, objective, target = job
    target_length = json.loads((folder / 'summary.json').read_text())['target_length']
    vertices, faces = load_obj(folder / 'remeshed.obj')
    edges = unique_edges(faces)
    ratio = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1) / target_length
    row = {'tier': tier, 'cycles': cycles, 'input': mesh, 'approach': approach, 'objective': objective,
           'target': target, 'long_fraction': float(np.mean(ratio > ELL_MAX_RATIO)),
           'short_fraction': float(np.mean(ratio < ELL_MIN_RATIO))}
    if tier != 'core':
        return row

    neighbors = [set() for _ in vertices]
    for a, b in edges:
        neighbors[a].add(b)
        neighbors[b].add(a)

    def collapse_too_long(a, b):
        middle = (vertices[a] + vertices[b]) / 2
        ring = list((neighbors[a] | neighbors[b]) - {a, b})
        return np.linalg.norm(vertices[ring] - middle, axis=1).max() > ELL_MAX_RATIO * target_length

    short_edges = edges[ratio < ELL_MIN_RATIO]
    row['collapse_candidates'] = len(short_edges)
    row['collapse_invalid_length'] = sum(collapse_too_long(a, b) for a, b in short_edges)
    return row


def scan_input(path):
    vertices, faces = load_obj(path)
    edges = unique_edges(faces)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return {'mesh': path.stem, 'vertices': len(vertices), 'edges': len(edges), 'faces': len(faces),
            'average_edge_length': float(lengths.mean()), 'edge_length_cv': float(lengths.std() / lengths.mean())}


def scan(meshes_dir, out_dir):
    results = pd.read_csv(out_dir / 'results.csv', keep_default_na=False)
    core = results[results.tier == 'core']
    jobs = []
    for run in core.itertuples():
        objective = [run.objective] if run.objective else []
        name = '__'.join([run.input, run.approach, *objective, f'T{float(run.target):g}'])
        jobs.append((out_dir / 'core' / name, 'core', '', run.input, run.approach, run.objective, float(run.target)))
    figure_runs = results[(results.tier == 'cycles') & (results.input == ITERATION_FIGURE_MESH) &
                          results.cycles.astype(int).isin(ITERATION_FIGURE_CYCLES)]
    for run in figure_runs.itertuples():
        name = f'{run.input}__{run.approach}__T{float(run.target):g}__k{int(run.cycles):02d}'
        jobs.append((out_dir / 'cycles' / name, 'cycles', int(run.cycles), run.input, run.approach, '',
                     float(run.target)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(scan_run, jobs, chunksize=4))
    input_rows = [scan_input(path) for path in sorted(meshes_dir / f'{mesh}.obj' for mesh in core.input.unique())]
    pd.DataFrame(rows).to_csv(out_dir / 'scans.csv', index=False)
    pd.DataFrame(input_rows).to_csv(out_dir / 'inputs.csv', index=False)
    print(len(rows), 'runs in scans.csv,', len(input_rows), 'meshes in inputs.csv', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('meshes_dir', type=Path)
    parser.add_argument('out_dir', type=Path)
    args = parser.parse_args()
    scan(args.meshes_dir, args.out_dir)


if __name__ == '__main__':
    main()
