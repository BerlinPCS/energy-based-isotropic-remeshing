#!/usr/bin/env python3
"""The schematic figures of Ch. 2 and 3 as TikZ files, one module in figures/ per figure. They need no data.

usage: generate_figures.py [NAME ...] [--out OUT]
"""
import argparse
import importlib
from pathlib import Path

# figures/length-thresholds.tikz of the thesis is written by hand.
MODULES = ['edge_loss_mesh', 'halfedge', 'mesh_defects', 'neighborhoods', 'overlapping_edits_collapse', 'queue_footprint', 'collapse_legality',
           'schedule_overview', 'small_operations', 'validity_checks']


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('names', nargs='*', choices=[[], *MODULES], metavar='NAME', default=[])
    parser.add_argument('--out', type=Path, default=Path(__file__).resolve().parents[1] / 'results')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.names or MODULES:
        importlib.import_module(f'figures.{name}').generate(args.out)
        print('wrote', name, flush=True)


if __name__ == '__main__':
    main()
