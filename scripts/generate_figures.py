#!/usr/bin/env python3
"""The generated figures and tables of the thesis, one module in figures/ per figure or table. Helpers are in shared/.

A module's generate() takes what it needs:
    generate(out)              the schematic figures of Ch. 2 and 3, which need no data
    generate(data, out)        the plots and tables, from the CSV files of run_experiments.py (--data)
    generate(data, runs, out)  the mesh images of Ch. 1 and 4, which also need that run's output meshes (--runs)
The mesh images and the gallery of App. A are rendered with build/bin/mesh-screenshot, which needs a display. Without
--runs, these are left out unless named. The numbers of the text come from claims.py, not from here.

usage: generate_figures.py [NAME ...] [--data DATA] [--runs RUNS] [--out OUT]
"""
import argparse
import importlib
import logging
import os
from pathlib import Path

# Figures are written to files only. Latin Modern's font files carry a zero creation date, which fontTools warns about
# when it embeds them.
os.environ.setdefault('MPLBACKEND', 'Agg')
logging.getLogger('fontTools').setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
# In order of appearance. figures/length-thresholds.tikz and figures/loop-local-average.tikz of the thesis are written
# by hand.
SCHEMATIC = ['mesh_defects', 'halfedge', 'small_operations', 'edge_loss_mesh', 'validity_checks', 'schedule_overview',
             'queue_footprint']
FROM_DATA = ['knee_trace', 'queues_heatmap', 'cycles_combined', 'comparison_table', 'mesh_gallery', 'mesh_table',
             'evaluation_data']
FROM_RUNS = ['views_bunny_iterations', 'views_bunny_queue_objectives', 'views_rocker_arm_objectives', 'views_horse_cap',
             'views_max_planck_hybrid', 'views_elk_antlers']
RENDERED = ['mesh_gallery', *FROM_RUNS]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('names', nargs='*', choices=[[], *SCHEMATIC, *FROM_DATA, *FROM_RUNS], metavar='NAME',
                        default=[])
    parser.add_argument('--data', type=Path, default=ROOT / 'data', help='the CSV files of run_experiments.py')
    parser.add_argument('--runs', type=Path, help='the output folder of run_experiments.py, for the mesh images')
    parser.add_argument('--out', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    names = args.names or [name for name in [*SCHEMATIC, *FROM_DATA, *FROM_RUNS] if args.runs or name not in RENDERED]
    if not args.runs and any(name in FROM_RUNS for name in names):
        parser.error('the mesh images need the output meshes: --runs RUNS')
    args.out.mkdir(parents=True, exist_ok=True)

    data = None
    if any(name not in SCHEMATIC for name in names):
        from shared.dataset import Dataset  # needs pandas, which the schematic figures do not
        data = Dataset(args.data)
    for name in names:
        module = importlib.import_module(f'figures.{name}')
        if name in SCHEMATIC:
            module.generate(args.out)
        elif name in FROM_DATA:
            module.generate(data, args.out)
        else:
            module.generate(data, args.runs, args.out)
        print('wrote', name, flush=True)


if __name__ == '__main__':
    main()
