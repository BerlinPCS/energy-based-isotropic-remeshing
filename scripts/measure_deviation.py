#!/usr/bin/env python3
"""The surface deviation d_RMS of Sec. 4.1, measured with the Hausdorff distance filter of MeshLab through PyMeshLab.

For every run that kept its output mesh, the filter samples N points by area on the faces of the output and finds the
closest point on the input's triangles for each, and then the other way round. Both directions use the same N, so

    d_RMS = sqrt((rms(output -> input)^2 + rms(input -> output)^2) / 2) / D0,

with D0 the bounding-box diagonal of the input. Two defaults of the filter are overridden: it samples the vertices
unless told otherwise, and it drops samples whose closest point is farther than half the diagonal. Every sample must
count in both directions.

The filter draws its samples from one random generator per process, which each call continues, so every mesh is
measured in a fresh process: a mesh then gets the same value on every run.

Each run folder gets deviation.json. results.csv gets these values as d_rms, matched by tier, input and mesh_sha256. A
tier that deletes its meshes (input-order) gets no d_rms.

usage: measure_deviation.py MESHES_DIR OUT_DIR
"""
import argparse
import concurrent.futures
import csv
import json
import math
from pathlib import Path

import pymeshlab

SAMPLES = 100_000


def hausdorff(meshes):
    """The deviation of an output mesh from its input; runs in a process of its own (see above)."""
    input_mesh, output_mesh = meshes
    mesh_set = pymeshlab.MeshSet()
    mesh_set.load_new_mesh(str(input_mesh))
    mesh_set.load_new_mesh(str(output_mesh))
    options = dict(samplevert=False, sampleedge=False, samplefauxedge=False, sampleface=True, samplenum=SAMPLES,
                   maxdist=pymeshlab.PercentageValue(100))
    to_input = mesh_set.get_hausdorff_distance(sampledmesh=1, targetmesh=0, **options)
    to_output = mesh_set.get_hausdorff_distance(sampledmesh=0, targetmesh=1, **options)
    if to_input['n_samples'] != SAMPLES or to_output['n_samples'] != SAMPLES:
        raise RuntimeError(f'{output_mesh}: samples dropped beyond the maximum distance')
    diagonal = to_output['diag_mesh_0']
    return {'d_rms': math.sqrt((to_input['RMS'] ** 2 + to_output['RMS'] ** 2) / 2) / diagonal,
            'rms_output_to_input': to_input['RMS'], 'rms_input_to_output': to_output['RMS'],
            'input_diagonal': diagonal, 'samples': SAMPLES}


def measure(meshes_dir, out_dir):
    """Writes deviation.json into every run folder of results.csv that is missing one, then d_rms into results.csv."""
    with open(out_dir / 'results.csv', newline='') as file:
        rows = list(csv.DictReader(file))
    # An output mesh and its input determine the deviation, so identical meshes (such as five baseline cycles in core
    # and in cycles) are measured once.
    folders = {}
    for tier in sorted({row['tier'] for row in rows}):
        for timing_file in sorted((out_dir / tier).glob('*/timing.json')):
            folder = timing_file.parent
            if not (folder / 'remeshed.obj').exists():
                continue
            mesh = Path(json.loads((folder / 'summary.json').read_text())['input']).stem.partition('__')[0]
            digest = json.loads(timing_file.read_text())['mesh_sha256']
            folders.setdefault((mesh, digest), []).append(folder)
    pending = {key: group for key, group in folders.items()
               if not all((folder / 'deviation.json').exists() for folder in group)}
    with concurrent.futures.ProcessPoolExecutor(max_tasks_per_child=1) as pool:
        futures = {pool.submit(hausdorff, (meshes_dir / f'{mesh}.obj', group[0] / 'remeshed.obj')): group
                   for (mesh, _), group in pending.items()}
        for future in concurrent.futures.as_completed(futures):
            text = json.dumps(future.result())
            for folder in futures[future]:
                (folder / 'deviation.json').write_text(text)

    deviation = {(folder.parent.name, mesh, digest): json.loads((folder / 'deviation.json').read_text())['d_rms']
                 for (mesh, digest), group in folders.items() for folder in group}
    measured_tiers = {tier for tier, _, _ in deviation}
    fields = list(rows[0])
    if 'd_rms' not in fields:
        # Among remesher-cli's summary keys, which are sorted, where older builds wrote their own d_rms.
        fields.insert(fields.index('edges'), 'd_rms')
    for row in rows:
        row['d_rms'] = deviation.get((row['tier'], row['input'], row['mesh_sha256']), '')
        if row['tier'] in measured_tiers and row['d_rms'] == '':
            raise RuntimeError(f'no deviation for a {row["tier"]} run of {row["input"]}')
    with open(out_dir / 'results.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return len(pending), sum(row['d_rms'] != '' for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('meshes_dir', type=Path)
    parser.add_argument('out_dir', type=Path)
    args = parser.parse_args()
    measured, filled = measure(args.meshes_dir, args.out_dir)
    print(measured, 'meshes measured,', filled, 'rows of results.csv with d_rms')


if __name__ == '__main__':
    main()
