"""The CSV files of one run of run_experiments.py, and the ratios that compare its configurations."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TARGETS = [0.75, 1.0, 1.5]
OBJECTIVES = ['sum', 'maximum', 'local-average', 'global-average']
# Columns of results.csv: the metrics of Sec. 4.1.
LOSS, TIME = 'average_edge_loss', 'algorithm_time_ms_median'
MIN_ANGLE, ANGLE_TAIL, VALENCE, DEVIATION = 'minimum_angle', 'angle_p05', 'valence_error', 'd_rms'


class Dataset:
    """The CSV files of one run. Rows of a configuration are indexed by (input, target), so that ratios align.

    results.csv has one row per run, grouped into tiers: core (the 19 configurations of the main comparison on the 30
    mesh-target cases), cycles and ordered-cycles (the baseline and the ordered phases stopped after k = 1 to 50
    cycles), no-knee (alternating without the knee) and input-order (the baseline on seven reorderings of each
    input)."""

    def __init__(self, folder):
        results = pd.read_csv(folder / 'results.csv').fillna({'objective': '', 'order': ''})
        self.tiers = {tier: rows for tier, rows in results.groupby('tier')}
        self.core = self.tiers['core']
        self.traces = pd.read_csv(folder / 'traces.csv').fillna({'objective': ''})
        self.scans = pd.read_csv(folder / 'scans.csv').fillna({'objective': ''})
        self.inputs = pd.read_csv(folder / 'inputs.csv').set_index('mesh')
        self.failures = pd.read_csv(folder / 'failures.csv')
        self.reference = self.runs('baseline')

    def runs(self, approach, objective=''):
        """The 30 runs of one configuration of the main comparison."""
        rows = self.core[(self.core.approach == approach) & (self.core.objective == objective)]
        return rows.set_index(['input', 'target'])

    def fixed(self, cycles, ordered=False):
        """The baseline, or the ordered phases, stopped after a fixed number of cycles."""
        rows = self.tiers['ordered-cycles' if ordered else 'cycles']
        return rows[rows.cycles == cycles].set_index(['input', 'target'])

    def scan(self, approach, objective=''):
        """The edge scans of the outputs of one configuration (scans.csv): edges outside the length band, and the short
        edges whose collapse the length check blocks."""
        rows = self.scans[(self.scans.tier == 'core') & (self.scans.approach == approach) &
                          (self.scans.objective == objective)]
        return rows.set_index(['input', 'target'])

    def no_knee(self, objective):
        rows = self.tiers['no-knee']
        return rows[rows.objective == objective].set_index(['input', 'target'])

    def ratio(self, runs, metric, reference=None):
        """Matched ratios: each case divided by five baseline cycles on the same mesh and target, or by another
        reference."""
        return runs[metric] / (self.reference if reference is None else reference)[metric]

    def median_ratio(self, runs, metric, reference=None):
        return float(self.ratio(runs, metric, reference).median())
