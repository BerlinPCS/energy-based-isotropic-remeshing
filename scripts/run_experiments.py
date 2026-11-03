#!/usr/bin/env python3
"""Runs the experiments of Ch. 4 with remesher-cli, one run at a time, and collects the dataset in OUT_DIR.

Tiers:
    core            the main comparison: 19 configurations on 30 mesh-target cases
    cycles          the baseline at every k = 1 to 50
    ordered-cycles  the ordered phases at every k = 1 to 50
    no-knee         alternating with --no-knee
    traces          alternating with --trace, and the dynamic queue run of the knee figure; each mesh must equal core's
    input-order     five baseline cycles on reordered copies of each input, one repetition

Every other tier repeats a configuration three times in a row. Runtime is the median algorithm time, and the output
meshes must be identical across the repetitions. Only the mesh and summary of the first repetition are kept. A run that
exceeds the safety cap of 900 s counts as failed and is listed in failures.csv. A finished configuration is skipped when
the script is started again.

Writes results.csv, traces.csv and failures.csv. measure_deviation.py then adds the surface deviation d_rms to
results.csv, measured with MeshLab, and scan_meshes.py adds scans.csv and inputs.csv.

usage: run_experiments.py MESHES_DIR OUT_DIR [--cli PATH] [--tiers TIER ...] [--meshes MESH ...]
"""
import argparse
import csv
import datetime
import hashlib
import json
import random
import shutil
import statistics
import subprocess
from pathlib import Path

import measure_deviation
import scan_meshes

MESHES = ['armadillo-r128', 'elk', 'fandisk', 'homer', 'horse', 'max-planck', 'rocker-arm', 'stanford-bunny-low',
          'stanford-bunny', 'stanford-dragon-r128']
TARGETS = ['0.75', '1', '1.5']
OBJECTIVES = ['sum', 'maximum', 'local-average', 'global-average']
QUEUE_APPROACHES = ['dynamic-queue', 'batched', 'hybrid', 'alternating']
CYCLE_COUNTS = range(1, 51)
INPUT_ORDERS = ['reverse', 'sorted-x', *(f'random{seed}' for seed in range(5))]
KNEE_FIGURE_RUN = ('stanford-bunny', '1', {'approach': 'dynamic-queue', 'objective': 'local-average'})
REPETITIONS = 3
SAFETY_CAP_SECONDS = 900


def main_comparison():
    yield {'approach': 'baseline'}
    yield {'approach': 'repeated'}
    yield {'approach': 'ordered'}
    for approach in QUEUE_APPROACHES:
        for objective in OBJECTIVES:
            yield {'approach': approach, 'objective': objective}


TIERS = {
    'core': main_comparison,
    'cycles': lambda: ({'approach': 'baseline', 'cycles': k} for k in CYCLE_COUNTS),
    'ordered-cycles': lambda: ({'approach': 'ordered', 'cycles': k} for k in CYCLE_COUNTS),
    'no-knee': lambda: ({'approach': 'alternating', 'objective': objective, 'no_knee': True}
                        for objective in OBJECTIVES),
}


def folder_name(mesh, target, configuration):
    parts = [mesh, configuration['approach']]
    if 'objective' in configuration:
        parts.append(configuration['objective'])
    parts.append(f'T{target}')
    if 'cycles' in configuration:
        parts.append(f'k{configuration["cycles"]:02d}')
    return '__'.join(parts)


def run_once(cli, input_mesh, folder, target, configuration, extra=()):
    command = [str(cli), str(input_mesh), str(folder), '--approach', configuration['approach'], '--target', target]
    if 'objective' in configuration:
        command += ['--objective', configuration['objective']]
    if 'cycles' in configuration:
        command += ['--cycles', str(configuration['cycles'])]
    if configuration.get('no_knee'):
        command += ['--no-knee']
    subprocess.run([*command, *extra], check=True, stdout=subprocess.DEVNULL, timeout=SAFETY_CAP_SECONDS)
    digest = hashlib.sha256(next(folder.glob('remeshed.*')).read_bytes()).hexdigest()
    return json.loads((folder / 'summary.json').read_text())['algorithm_time_ms'], digest


def run_configuration(cli, input_mesh, folder, target, configuration, repetitions=REPETITIONS):
    timing_file = folder / 'timing.json'
    failure_file = folder / 'failure.txt'
    if timing_file.exists() or failure_file.exists():
        return
    times, digests = [], []
    try:
        for repetition in range(repetitions):
            # Only the first repetition stays in the configuration's folder.
            output = folder if repetition == 0 else folder / 'repetition'
            time, digest = run_once(cli, input_mesh, output, target, configuration)
            times.append(time)
            digests.append(digest)
            if repetition > 0:
                shutil.rmtree(output)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        folder.mkdir(parents=True, exist_ok=True)
        failure_file.write_text(str(error))
        return
    if len(set(digests)) != 1:
        raise RuntimeError(f'{folder.name}: the repetitions differ')
    timing_file.write_text(json.dumps({'algorithm_time_ms': times, 'mesh_sha256': digests[0]}))


def reorder(input_mesh, output, order):
    """A copy of the mesh with its vertices and faces in another order, which is the same surface for the remesher."""
    vertices, faces = [], []
    for line in input_mesh.read_text().splitlines():
        if line.startswith('v '):
            vertices.append(line)
        elif line.startswith('f '):
            faces.append([int(index) - 1 for index in line.split()[1:]])
    vertex_order, face_order = list(range(len(vertices))), list(range(len(faces)))
    if order == 'reverse':
        vertex_order.reverse()
        face_order.reverse()
    elif order == 'sorted-x':
        x = [float(line.split()[1]) for line in vertices]
        vertex_order.sort(key=lambda v: x[v])
        face_order.sort(key=lambda f: sum(x[v] for v in faces[f]))
    else:
        generator = random.Random(int(order.removeprefix('random')))
        generator.shuffle(vertex_order)
        generator.shuffle(face_order)
    new_index = {old: new for new, old in enumerate(vertex_order)}
    lines = [vertices[v] for v in vertex_order]
    lines += ['f ' + ' '.join(str(new_index[v] + 1) for v in faces[f]) for f in face_order]
    output.write_text('\n'.join(lines) + '\n')


def run_input_order(cli, meshes_dir, out_dir, meshes):
    for mesh in meshes:
        for order in INPUT_ORDERS:
            folders = [out_dir / 'input-order' / f'{mesh}__{order}__T{target}' for target in TARGETS]
            if all((folder / 'timing.json').exists() for folder in folders):
                continue
            reordered = out_dir / 'input-order' / f'{mesh}__{order}.obj'
            reordered.parent.mkdir(parents=True, exist_ok=True)
            reorder(meshes_dir / f'{mesh}.obj', reordered, order)
            for folder, target in zip(folders, TARGETS):
                run_configuration(cli, reordered, folder, target, {'approach': 'baseline'}, repetitions=1)
                (folder / 'remeshed.obj').unlink(missing_ok=True)
            reordered.unlink()


def count_elements(path):
    vertices = faces = 0
    with open(path) as file:
        for line in file:
            vertices += line.startswith('v ')
            faces += line.startswith('f ')
    return vertices, faces, faces * 3 // 2


def knee_trace_interval(input_mesh):
    """A divisor of the knee window K of Sec. 3.2.3, so that the trace has a row at every window boundary."""
    vertices, _, edges = count_elements(input_mesh)
    window = max(50, (vertices + edges) // 50)
    divisors = [d for d in range(1, window // 8 + 1) if window % d == 0]
    return max(divisors) if divisors else 1


def run_traces(cli, meshes_dir, out_dir, meshes):
    runs = [(mesh, target, {'approach': 'alternating', 'objective': objective})
            for mesh in meshes for target in TARGETS for objective in OBJECTIVES]
    if KNEE_FIGURE_RUN[0] in meshes:
        runs.append(KNEE_FIGURE_RUN)
    work = out_dir / 'traces-work'
    work.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'traces.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['mesh', 'approach', 'objective', 'target', 'event', 'edits', 'average_edge_loss', 'edges',
                         'vertices'])
        for mesh, target, configuration in runs:
            input_mesh = meshes_dir / f'{mesh}.obj'
            extra = ['--trace', str(work / 'trace.csv')]
            if (mesh, target, configuration) == KNEE_FIGURE_RUN:
                extra += ['--trace-interval', str(knee_trace_interval(input_mesh))]
            _, digest = run_once(cli, input_mesh, work / 'run', target, configuration, extra)
            name = folder_name(mesh, target, configuration)
            core = json.loads((out_dir / 'core' / name / 'timing.json').read_text())['mesh_sha256']
            if digest != core:
                raise RuntimeError(f'{name}: the traced mesh differs from core')
            with open(work / 'trace.csv') as trace:
                for row in list(csv.reader(trace))[1:]:
                    writer.writerow([mesh, configuration['approach'], configuration['objective'], target, *row])
    shutil.rmtree(work)


def collect(out_dir, tiers):
    rows, failures = [], []
    for tier in tiers:
        tier_dir = out_dir / tier
        for failure_file in sorted(tier_dir.glob('*/failure.txt')):
            failures.append({'tier': tier, 'run': failure_file.parent.name, 'reason': failure_file.read_text()})
        for timing_file in sorted(tier_dir.glob('*/timing.json')):
            timing = json.loads(timing_file.read_text())
            summary = json.loads((timing_file.parent / 'summary.json').read_text())
            mesh, _, order = Path(summary['input']).stem.partition('__')
            summary['input'] = mesh
            summary.pop('algorithm_time_ms')
            rows.append({'tier': tier, 'order': order, **summary,
                         'algorithm_time_ms_median': statistics.median(timing['algorithm_time_ms']),
                         'algorithm_time_ms_min': min(timing['algorithm_time_ms']),
                         'algorithm_time_ms_max': max(timing['algorithm_time_ms']),
                         'mesh_sha256': timing['mesh_sha256']})
    with open(out_dir / 'results.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / 'failures.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['tier', 'run', 'reason'])
        writer.writeheader()
        writer.writerows(failures)
    return len(rows), len(failures)


def record_build(cli, out_dir):
    cache = cli.parents[1] / 'CMakeCache.txt'
    flags = [line.strip() for line in cache.read_text().splitlines()
             if line.startswith(('CMAKE_BUILD_TYPE:', 'CMAKE_CXX_FLAGS_RELEASE:', 'CMAKE_CXX_COMPILER:'))] \
        if cache.exists() else []
    processor = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout
    compiler = subprocess.run(['c++', '--version'], capture_output=True, text=True).stdout.splitlines()[:1]
    (out_dir / 'BUILD_INFO').write_text('\n'.join([
        datetime.datetime.now().isoformat(timespec='seconds'), processor.strip(), *compiler, *flags,
        'remesher-cli sha256 ' + hashlib.sha256(cli.read_bytes()).hexdigest()]) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('meshes_dir', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--cli', type=Path, default=Path(__file__).resolve().parents[1] / 'build/bin/remesher-cli')
    parser.add_argument('--tiers', nargs='+', choices=[*TIERS, 'traces', 'input-order'],
                        default=[*TIERS, 'traces', 'input-order'])
    parser.add_argument('--meshes', nargs='+', choices=MESHES, default=MESHES)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # A bare relative name would make subprocess search the PATH for it.
    args.cli = args.cli.resolve()

    record_build(args.cli, args.out_dir)
    for tier in args.tiers:
        if tier == 'traces':
            run_traces(args.cli, args.meshes_dir, args.out_dir, args.meshes)
        elif tier == 'input-order':
            run_input_order(args.cli, args.meshes_dir, args.out_dir, args.meshes)
        else:
            for mesh in args.meshes:
                for target in TARGETS:
                    for configuration in TIERS[tier]():
                        folder = args.out_dir / tier / folder_name(mesh, target, configuration)
                        run_configuration(args.cli, args.meshes_dir / f'{mesh}.obj', folder, target, configuration)
        print(tier, 'finished', datetime.datetime.now().isoformat(timespec='seconds'), flush=True)
    finished, failed = collect(args.out_dir, [tier for tier in args.tiers if tier != 'traces'])
    print(finished, 'configurations in results.csv,', failed, 'failed')
    measure_deviation.measure(args.meshes_dir, args.out_dir)
    scan_meshes.scan(args.meshes_dir, args.out_dir)


if __name__ == '__main__':
    main()
