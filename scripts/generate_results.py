#!/usr/bin/env python3
"""The numbers, tables, plots and images of the thesis, all from one dataset of run_experiments.py.

Outputs, each selectable with --only:
    claims              claims.csv: every number the thesis states, in order of appearance.
    comparison-table    the table of the Discussion
    appendix-b          the two tables of Appendix B, as a whole chapter
    queues-heatmap      Fig. queue-heatmap
    knee-trace          Fig. knee trace
    cycles-combined     Fig. eval-cycles
    views-*             the mesh images of result_images.py

Everything but the images needs only the CSV files in DATA. The images also need the output meshes of a full run, given
with --runs, and a display. Nothing here writes to the dataset.

usage: generate_results.py [--data DATA] [--out OUT] [--runs FOLDER] [--only NAME ...]
"""
import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
# Latin Modern's font files carry a zero creation date, which fontTools warns about when it embeds them.
logging.getLogger('fontTools').setLevel(logging.ERROR)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

import result_images
from common.style import BLUE, MESH_NAMES, ORANGE, PLOT_STYLE, TEXT_WIDTH_INCHES, decimal_column

TARGETS = [0.75, 1.0, 1.5]
OBJECTIVES = ['sum', 'maximum', 'local-average', 'global-average']
LOSS, TIME = 'average_edge_loss', 'algorithm_time_ms_median'
MIN_ANGLE, ANGLE_TAIL, VALENCE, DEVIATION = 'minimum_angle', 'angle_p05', 'valence_error', 'd_rms'
KNEE_FIGURE = dict(mesh='stanford-bunny', approach='dynamic-queue', objective='local-average', target=1.0)
KNEE_RATE_FRACTION = 0.25


class Dataset:
    """The CSV files of one run. Rows of a configuration are indexed by (input, target), so that ratios align."""

    def __init__(self, folder):
        results = pd.read_csv(folder / 'results.csv').fillna({'objective': '', 'order': ''})
        self.tiers = {tier: rows for tier, rows in results.groupby('tier')}
        self.core = self.tiers['core']
        self.traces = pd.read_csv(folder / 'traces.csv').fillna({'objective': ''})
        self.scans = pd.read_csv(folder / 'scans.csv').fillna({'objective': ''})
        self.inputs = pd.read_csv(folder / 'inputs.csv').set_index('mesh')
        self.failures = pd.read_csv(folder / 'failures.csv')
        self.horse_cap = json.loads((folder / 'horse-cap.json').read_text())
        self.reference = self.runs('baseline')

    def runs(self, approach, objective=''):
        rows = self.core[(self.core.approach == approach) & (self.core.objective == objective)]
        return rows.set_index(['input', 'target'])

    def fixed(self, cycles, ordered=False):
        rows = self.tiers['ordered-cycles' if ordered else 'cycles']
        return rows[rows.cycles == cycles].set_index(['input', 'target'])

    def scan(self, approach, objective=''):
        rows = self.scans[(self.scans.tier == 'core') & (self.scans.approach == approach) &
                          (self.scans.objective == objective)]
        return rows.set_index(['input', 'target'])

    def no_knee(self, objective):
        rows = self.tiers['no-knee']
        return rows[rows.objective == objective].set_index(['input', 'target'])

    def ratio(self, runs, metric, reference=None):
        """Matched ratios to five baseline cycles, or to another reference."""
        return runs[metric] / (self.reference if reference is None else reference)[metric]

    def median_ratio(self, runs, metric, reference=None):
        return float(self.ratio(runs, metric, reference).median())


def span(values, digits=2, scale=1, unit=''):
    values = [scale * value for value in values]
    return f'{min(values):.{digits}f}{unit} to {max(values):.{digits}f}{unit}'


def per_objective(function):
    return ', '.join(f'{objective}: {function(objective)}' for objective in OBJECTIVES)


def share(value):
    return f'{100 * value:.1f}%'


def signed_share(value):
    return f'{100 * value:+.1f}%'


def knee_windows(trace):
    """The windows of the knee rule of Sec. 3.2.3 on a traced queue stage, and the first weak one."""
    first = trace.iloc[0]
    window = max(50, int(first.edges + first.vertices) // 50)
    by_edits = trace.drop_duplicates('edits').set_index('edits').average_edge_loss
    rows, best, start = [], 0.0, 0
    while start + window <= by_edits.index.max():
        end = start + window
        gain = (by_edits[start] - by_edits[end]) / abs(by_edits[start])
        best = max(best, gain)
        weak = gain < 0 or (best > 0 and gain < KNEE_RATE_FRACTION * best)
        rows.append(dict(start=start, end=end, gain=gain, threshold=KNEE_RATE_FRACTION * best, weak=weak))
        start = end
    windows = pd.DataFrame(rows)
    return window, windows, windows[windows.weak].iloc[0]


def knee_figure_trace(data):
    traces = data.traces
    selected = (traces.mesh == KNEE_FIGURE['mesh']) & (traces.approach == KNEE_FIGURE['approach']) & \
        (traces.objective == KNEE_FIGURE['objective']) & (traces.target == KNEE_FIGURE['target'])
    return traces[selected]


def alternating_stages(data):
    """The relative change of the average loss in every cycle and in every queue stage directly after a cycle."""
    rows = []
    alternating = data.traces[data.traces.approach == 'alternating']
    for (mesh, objective, target), trace in alternating.groupby(['mesh', 'objective', 'target'], sort=False):
        previous_event, previous_loss, round_number = None, None, 0
        for event, loss in zip(trace.event, trace.average_edge_loss):
            if event not in ('initial', 'cycle', 'stage'):
                continue
            if event == 'cycle':
                round_number += 1
            if previous_loss is not None and (event == 'cycle' or previous_event == 'cycle'):
                rows.append(dict(objective=objective, event=event, round=round_number,
                                 change=(loss - previous_loss) / previous_loss))
            previous_event, previous_loss = event, loss
    return pd.DataFrame(rows)


def claim_list(data):
    """(where, statement, value) for every number in the text, in order of appearance. The prose is free to change."""
    core, inputs = data.core, data.inputs

    def outside_band(k, target):
        rows = data.scans[(data.scans.tier == 'cycles') & (data.scans.cycles == k) & (data.scans.target == target)]
        return share(float(rows.long_fraction.iloc[0] + rows.short_fraction.iloc[0]))

    for k in [5, 25]:
        yield ('Introduction', f'Bunny: edges outside the length thresholds after {k} baseline cycles at T = 1, T = 1.5',
               f'{outside_band(k, 1.0)}, {outside_band(k, 1.5)}')

    window, windows, knee = knee_windows(knee_figure_trace(data))
    trace = knee_figure_trace(data).drop_duplicates('edits').set_index('edits')
    yield 'Fig. knee trace', 'Edits until no acceptable candidate remains (Bunny, T = 1, dynamic queue, local average)', \
        f'{int(trace.index.max())}'
    yield 'Fig. knee trace', 'Edits at which a knee-stopped stage would end', f'{int(knee.end)} (K = {window})'
    yield 'Fig. knee trace', 'Further decrease of the average loss after the knee', \
        share(1 - trace.average_edge_loss.iloc[-1] / trace.average_edge_loss[int(knee.end)])

    configurations = core.groupby(['approach', 'objective']).ngroups
    cases = core.groupby(['input', 'target']).ngroups
    yield 'Setup', 'Configurations of the main comparison, mesh-target cases', f'{configurations}, {cases}'
    smallest = inputs.loc[['elk', 'stanford-bunny-low']]
    others = inputs.drop(smallest.index)
    yield 'Setup', 'Vertices of the two smallest inputs (Elk, Bunny (low))', \
        ', '.join(str(v) for v in smallest.vertices)
    ratios = [a / b for a in smallest.average_edge_length for b in others.average_edge_length]
    yield 'Setup', 'Average edge length of the two smallest inputs relative to the other meshes', f'{span(ratios, 1)} x'
    yield 'Setup', "Area of the Elk thinner than the target at T = 0.75, 1, 1.5", \
        ', '.join(share(inputs.loc['elk', f'thin_area_share_T{target:g}']) for target in TARGETS)
    yield 'Setup', 'Runs of the main comparison, failed runs in the whole dataset', f'{len(core)}, {len(data.failures)}'
    yield 'Setup', 'Longest run of the main comparison (seconds; the safety cap is 900)', \
        f'{core.algorithm_time_ms_max.max() / 1000:.1f}'
    yield 'Setup', 'Stop reasons of the main comparison (no edit-cap or cycle-cap stop means no run reached those limits)', \
        ', '.join(f'{reason}: {count}' for reason, count in core.stop_reason.value_counts().items())
    yield 'Setup', 'Most cycles or rounds of any run of the main comparison', f'{int(core.cycles.max())}'

    for approach in ['dynamic-queue', 'batched']:
        for objective in OBJECTIVES:
            runs = data.ratio(data.runs(approach, objective), LOSS)
            yield 'Queues', f'Median loss ratio at T = 0.75, 1, 1.5: {approach}, {objective}', \
                ', '.join(f'{runs.xs(target, level="target").median():.2f}' for target in TARGETS)
    for approach in ['dynamic-queue', 'batched', 'hybrid']:
        for (objective, reason), count in core[core.approach == approach].groupby(['objective', 'stop_reason']).size().items():
            yield 'Queues', f'Stop reason of {approach}, {objective}: {reason}', f'{count}'

    for objective in ['local-average', 'maximum']:
        ratio = data.ratio(data.runs('dynamic-queue', objective), LOSS).loc[('stanford-bunny', 1.5)]
        yield 'Queues', f'Bunny, T = 1.5: loss relative to the baseline, dynamic queue, {objective}', f'{ratio:.1f} x'

    bunny = ('stanford-bunny', 0.75)
    yield 'Queues', 'Bunny, T = 0.75: edges too long, dynamic queue with sum, baseline', \
        f"{share(data.scan('dynamic-queue', 'sum').long_fraction.loc[bunny])}, " \
        f"{share(data.scan('baseline').long_fraction.loc[bunny])}"
    yield 'Queues', 'T = 1.5: edges too short (median over meshes), dynamic queue', per_objective(
        lambda o: share(data.scan('dynamic-queue', o).short_fraction.xs(1.5, level='target').median()))

    def blocked(objective):
        scan = data.scan('dynamic-queue', objective).xs(1.5, level='target')
        return share(scan.collapse_invalid_length.sum() / scan.collapse_candidates.sum())

    yield 'Queues', 'T = 1.5: short edges whose collapse the length check blocks (all meshes pooled), dynamic queue', \
        per_objective(blocked)

    def cap_shares(objective):
        values = []
        for approach in ['dynamic-queue', 'batched']:
            scan = data.scan(approach, objective)
            for target in [0.75, 1.0]:
                rows = scan.xs(target, level='target')
                values.append(float((rows.cap_sequences / rows.long_edges).median()))
        return span(values, 0, 100, '%')

    yield 'Queues', 'Long edges with a valid flip-collapse sequence at T <= 1 (range of the medians over meshes; both queues)', \
        per_objective(cap_shares)
    yield 'Queues', 'Fig. horse-cap: sum gain of the split, the flip, the collapse, flip and collapse together', \
        ', '.join(f'{data.horse_cap[key]:.2f}' for key in
                  ['split_gain_sum', 'flip_gain_sum', 'collapse_gain_sum', 'net_gain_sum'])
    yield 'Queues', 'Dynamic queue outputs with an angle below 1 degree (of 30)', per_objective(
        lambda o: int((data.runs('dynamic-queue', o).minimum_angle < 1).sum()))
    yield 'Queues', 'Baseline outputs with an angle below 1 degree', f'{int((data.reference.minimum_angle < 1).sum())}'
    yield 'Queues', 'Dynamic queue outputs with an angle below 0.001 degrees', per_objective(
        lambda o: int((data.runs('dynamic-queue', o).minimum_angle < 0.001).sum()))

    def edit_ratio(approach, objective, column):
        return data.ratio(data.runs(approach, objective), column).xs(1.5, level='target').median()

    yield 'Queues', 'T = 1.5, local average: collapses relative to the baseline, dynamic queue then batched', \
        f"{edit_ratio('dynamic-queue', 'local-average', 'collapses'):.2f}, " \
        f"{edit_ratio('batched', 'local-average', 'collapses'):.2f}"
    yield 'Queues', 'T = 1.5, sum: flips relative to the baseline, dynamic queue then batched', \
        f"{edit_ratio('dynamic-queue', 'sum', 'flips'):.2f}, {edit_ratio('batched', 'sum', 'flips'):.2f}"

    ten = data.fixed(10)
    alternating = {o: data.runs('alternating', o) for o in OBJECTIVES}
    yield 'Combined', 'Alternating: median loss ratio', per_objective(lambda o: f'{data.median_ratio(alternating[o], LOSS):.2f}')
    yield 'Combined', 'Alternating: cases below five baseline cycles (of 30)', per_objective(
        lambda o: int((data.ratio(alternating[o], LOSS) < 1).sum()))
    yield 'Combined', 'Alternating: cases below ten baseline cycles (of 30)', per_objective(
        lambda o: int((data.ratio(alternating[o], LOSS, ten) < 1).sum()))
    stages = alternating_stages(data)
    first = stages[(stages.event == 'cycle') & (stages['round'] == 1)].change.median()
    later = stages[(stages.event == 'cycle') & (stages['round'] > 1)].change.median()
    raising = stages[(stages.event == 'cycle') & (stages['round'] > 1) & (stages.change > 0)].change.median()
    yield 'Combined', 'Alternating: median change of the loss in the first cycle, in later cycles, in later cycles that raise it', \
        f'{signed_share(first)}, {signed_share(later)}, {signed_share(raising)}'
    yield 'Combined', 'Alternating: median change of the loss in a queue stage after a later cycle', per_objective(
        lambda o: signed_share(stages[(stages.event == 'stage') & (stages['round'] > 1) &
                                      (stages.objective == o)].change.median()))
    lowest = pd.DataFrame({o: alternating[o][LOSS] for o in OBJECTIVES}).idxmin(axis=1).value_counts()
    yield 'Combined', 'Alternating: cases where each objective has the lowest loss', per_objective(
        lambda o: int(lowest.get(o, 0)))

    fifty = data.fixed(50)
    ten_to_five = data.ratio(ten, LOSS)
    fifty_to_ten = data.ratio(fifty, LOSS, ten)
    yield 'Convergent', 'Ten cycles: median loss ratio to five, cases lower (of 30)', \
        f'{ten_to_five.median():.3f}, {int((ten_to_five < 1).sum())}'
    yield 'Convergent', 'Fifty cycles: median loss ratio to ten', f'{fifty_to_ten.median():.3f}'
    yield 'Convergent', 'Fifty cycles relative to ten at T = 1: Horse, Elk', \
        f"{fifty_to_ten.loc[('horse', 1.0)]:.3f}, {fifty_to_ten.loc[('elk', 1.0)]:.3f}"

    convergent, ordered = data.runs('convergent'), data.runs('ordered')
    stop = int(convergent.cycles.median())
    fourteen = data.fixed(stop)
    against = data.ratio(convergent, LOSS, fourteen)
    yield 'Convergent', 'Convergent: median stop, shortest, longest', \
        f'{stop}, {int(convergent.cycles.min())}, {int(convergent.cycles.max())}'
    yield 'Convergent', 'Convergent: median loss ratio to five cycles', f'{data.median_ratio(convergent, LOSS):.3f}'
    yield 'Convergent', f'{stop} fixed cycles: median loss ratio, time ratio to five cycles', \
        f'{data.median_ratio(fourteen, LOSS):.3f}, {data.median_ratio(fourteen, TIME):.2f}'
    yield 'Convergent', 'Convergent: median time ratio to five cycles', f'{data.median_ratio(convergent, TIME):.2f}'
    yield 'Convergent', f'Convergent against {stop} fixed cycles: lower, higher, equal', \
        f'{int((against < 1).sum())}, {int((against > 1).sum())}, {int((against == 1).sum())}'
    yield 'Convergent', 'Ordered, five cycles: cases below the baseline (of 30)', \
        f'{int((data.ratio(data.fixed(5, ordered=True), LOSS) < 1).sum())}'

    elk = ('elk', 1.0)
    yield 'Convergent', 'Elk, T = 1: loss after 25 and after 5 cycles', \
        f'{data.fixed(25).loc[elk, LOSS]:.3f}, {data.reference.loc[elk, LOSS]:.3f}'
    yield 'Convergent', 'Elk, T = 1: deviation after 14 and after 25 cycles, relative to 5', \
        f'{data.ratio(data.fixed(14), DEVIATION).loc[elk]:.2f}, {data.ratio(data.fixed(25), DEVIATION).loc[elk]:.2f}'
    yield 'Convergent', 'Convergent: median deviation ratio', f'{data.median_ratio(convergent, DEVIATION):.3f}'
    yield 'Convergent', 'Baseline after 1, 2 cycles: median deviation ratio', \
        f'{data.median_ratio(data.fixed(1), DEVIATION):.3f}, {data.median_ratio(data.fixed(2), DEVIATION):.3f}'
    for approach in ['dynamic-queue', 'hybrid', 'alternating']:
        yield 'Convergent', f'{approach}: median deviation ratio', per_objective(
            lambda o: f'{data.median_ratio(data.runs(approach, o), DEVIATION):.3f}')

    for approach in ['dynamic-queue', 'batched', 'hybrid', 'alternating']:
        yield 'Runtime', f'{approach}: median time ratio to five cycles', per_objective(
            lambda o: f'{data.median_ratio(data.runs(approach, o), TIME):.2f}')
    yield 'Runtime', 'Alternating without the knee: median time relative to alternating', per_objective(
        lambda o: f'{data.median_ratio(data.no_knee(o), TIME, alternating[o]):.2f}')
    yield 'Runtime', 'Convergent, ordered: median time ratio to five cycles', \
        f'{data.median_ratio(convergent, TIME):.2f}, {data.median_ratio(ordered, TIME):.2f}'
    yield 'Runtime', 'Convergent, ordered: median stop', f'{stop}, {int(ordered.cycles.median())}'
    yield 'Runtime', 'Ten fixed cycles: median time ratio to five cycles', f'{data.median_ratio(ten, TIME):.2f}'

    orders = data.tiers['input-order'].set_index(['input', 'target'])
    difference = (orders[LOSS] / data.reference[LOSS] - 1).abs()
    largest = difference.groupby(level=['input', 'target']).max()
    yield 'Discussion', 'Baseline on reordered inputs: largest change of the loss per case (median case, worst case)', \
        f'{share(largest.median())}, {share(largest.max())}'


def claims(data, out):
    with open(out / 'claims.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['where', 'statement', 'value'])
        writer.writerows(claim_list(data))


# The symbols of Sec. 4.1 as row and column headers; the captions say what they measure.
LOSS_SYMBOL, TIME_SYMBOL = r'$\bar\lambda$', '$t$'
MIN_ANGLE_SYMBOL, ANGLE_TAIL_SYMBOL = r'$\theta_{\min}$', r'$\theta_{5\%}$'
VALENCE_SYMBOL, DEVIATION_SYMBOL = r'$\sigma_{\mathrm{val}}$', r'$d_{\mathrm{RMS}}$'
COMPARISON_ROWS = [(LOSS_SYMBOL, LOSS, 3), (ANGLE_TAIL_SYMBOL, ANGLE_TAIL, 3), (VALENCE_SYMBOL, VALENCE, 3),
                   (DEVIATION_SYMBOL, DEVIATION, 3), (TIME_SYMBOL, TIME, 2)]


def comparison_table(data, out):
    columns = [data.fixed(10), data.runs('convergent'), data.runs('ordered'),
               *(data.runs('alternating', objective) for objective in OBJECTIVES)]
    rows = [[name, *(f'{data.median_ratio(runs, metric):.{digits}f}' for runs in columns)]
            for name, metric, digits in COMPARISON_ROWS]
    spec = ''.join(decimal_column([row[index] for row in rows]) for index in range(1, len(rows[0])))
    lines = ['% Generated by scripts/generate_results.py.', rf'\begin{{tabular}}{{@{{}}l{spec}@{{}}}}', r'\toprule',
             r'& & & & \multicolumn{4}{c}{Alternating} \\', r'\cmidrule(l){5-8}',
             r'& {10 cycles} & {Convergent} & {Ordered} & {Sum} & {Max.} & {Local\ avg.} & {Global\ avg.} \\',
             r'\midrule', *(' & '.join(row) + r' \\' for row in rows), r'\bottomrule', r'\end{tabular}']
    (out / 'comparison-table.tex').write_text('\n'.join(lines) + '\n')


def median_cell(metric, digits=3):
    """The median ratio, or an upper bound when it would round to zero."""
    def cell(data, runs):
        value = data.median_ratio(runs, metric)
        return f'{value:.{digits}f}' if value >= 0.5 * 10 ** -digits else f'<{10 ** -digits:.{digits}f}'
    return cell


def target_cell(target):
    return lambda data, runs: f'{data.ratio(runs, LOSS).xs(target, level="target").median():.3f}'


# One table column per entry: (group over several columns, header, cell of a configuration).
ALL_CONFIGURATION_COLUMNS = [
    (LOSS_SYMBOL, 'All', median_cell(LOSS)),
    (LOSS_SYMBOL, '$T=0.75$', target_cell(0.75)),
    (LOSS_SYMBOL, '$T=1$', target_cell(1.0)),
    (LOSS_SYMBOL, '$T=1.5$', target_cell(1.5)),
    (LOSS_SYMBOL, 'Worst', lambda data, runs: f'{data.ratio(runs, LOSS).max():.2f}'),
    ('Lower than', '$k=5$', lambda data, runs: str(int((data.ratio(runs, LOSS) < 1).sum()))),
    ('Lower than', '$k=10$', lambda data, runs: str(int((data.ratio(runs, LOSS, data.fixed(10)) < 1).sum()))),
    ('', MIN_ANGLE_SYMBOL, median_cell(MIN_ANGLE)),
    ('', ANGLE_TAIL_SYMBOL, median_cell(ANGLE_TAIL)),
    ('', VALENCE_SYMBOL, median_cell(VALENCE)),
    ('', DEVIATION_SYMBOL, median_cell(DEVIATION)),
    ('', TIME_SYMBOL, median_cell(TIME, 2)),
    ('Outputs with', r'$\theta_{\min}{<}1^\circ$', lambda data, runs: str(int((runs.minimum_angle < 1).sum()))),
    ('Outputs with', r'$\theta_{\min}{<}0.001^\circ$',
     lambda data, runs: str(int((runs.minimum_angle < 0.001).sum()))),
]
OBJECTIVE_LABELS = [('sum', 'Sum'), ('maximum', 'Max.'), ('local-average', r'L.\ avg.'), ('global-average', r'G.\ avg.')]


def all_configurations(data):
    yield 'Fixed cycles', [('1 cycle', data.fixed(1)), ('2 cycles', data.fixed(2)), ('5 cycles', data.reference),
                           *((f'{k} cycles', data.fixed(k)) for k in [10, 14, 25, 50]),
                           ('Ordered, 5 cycles', data.fixed(5, ordered=True))]
    yield 'Convergent phases', [('Repeated cycles', data.runs('convergent')), ('Ordered', data.runs('ordered'))]
    for approach, name in [('dynamic-queue', 'Dynamic queue'), ('batched', 'Batched queue'), ('hybrid', 'Hybrid'),
                           ('alternating', 'Alternating')]:
        yield name, [(label, data.runs(approach, objective)) for objective, label in OBJECTIVE_LABELS]


def group_header(columns):
    """The header rows of the table: groups over the columns they span, then the column headers."""
    cells, rules, position = [], [], 2
    index = 0
    while index < len(columns):
        group = columns[index][0]
        length = 1
        while group and index + length < len(columns) and columns[index + length][0] == group:
            length += 1
        if group:
            cells.append(rf'\multicolumn{{{length}}}{{c}}{{{group}}}')
            rules.append(rf'\cmidrule(lr){{{position}-{position + length - 1}}}')
        else:
            cells.append('')
        position += length
        index += length
    headers = [f'{{{column[1]}}}' for column in columns]  # braced, so that siunitx does not parse them
    return [' & '.join(['', *cells]) + r' \\', ''.join(rules), ' & '.join(['Configuration', *headers]) + r' \\']


def table_all_configurations(data):
    columns = ALL_CONFIGURATION_COLUMNS
    groups = [(group, [[rf'\quad {label}', *(column[2](data, runs) for column in columns)]
                       for label, runs in configurations]) for group, configurations in all_configurations(data)]
    rows = [row for _, group_rows in groups for row in group_rows]
    spec = ''.join(decimal_column([row[index] for row in rows]) for index in range(1, len(columns) + 1))
    lines = [r'\begin{landscape}', r'\begin{table}[p]', r'\centering\footnotesize', r'\setlength{\tabcolsep}{4pt}',
             rf'\begin{{tabular}}{{@{{}}l{spec}@{{}}}}', r'\toprule', *group_header(columns), r'\midrule']
    for group, group_rows in groups:
        lines.append(rf'\multicolumn{{{len(columns) + 1}}}{{@{{}}l}}{{\emph{{{group}}}}} \\')
        lines += [' & '.join(row) + r' \\' for row in group_rows]
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{All configurations relative to five baseline cycles: the median over all applicable cases '
              r'for the average edge loss $\bar\lambda$, the smallest angle $\theta_{\min}$, the fifth-percentile '
              r'angle $\theta_{5\%}$, the valence error $\sigma_{\mathrm{val}}$, the surface deviation '
              r'$d_{\mathrm{RMS}}$ and the runtime $t$. \emph{Worst} is the largest ratio of $\bar\lambda$ in any '
              r'case. \emph{Lower than} counts the cases with strictly lower $\bar\lambda$ than $k=5$ or $k=10$ '
              r'baseline cycles. \emph{Outputs with} counts the outputs whose smallest angle $\theta_{\min}$ is below '
              r'$1^\circ$ or $0.001^\circ$.}',
              r'\label{tab:eval-all}', r'\end{table}', r'\end{landscape}']
    return lines


def table_baseline(data):
    def cells(mesh, name, target):
        row = data.reference.loc[(mesh, target)]
        return [name if target == 1.0 else '', f'{target:g}', f'{int(row.vertices)}', f'{row[LOSS]:.4f}',
                f'{row.minimum_angle:.1f}', f'{row[ANGLE_TAIL]:.1f}', f'{row[VALENCE]:.3f}',
                f'{1e3 * row[DEVIATION]:.3f}', f'{row[TIME]:.1f}']

    groups = [[cells(mesh, name, target) for target in TARGETS] for mesh, name in MESH_NAMES.items()]
    rows = [row for group in groups for row in group]
    spec = ''.join(decimal_column([row[index] for row in rows]) for index in range(1, len(rows[0])))
    lines = [r'\begin{table}[htbp]', r'\centering\footnotesize', r'\setlength{\tabcolsep}{6pt}',
             r'\sisetup{group-digits=integer, group-minimum-digits=4}', rf'\begin{{tabular}}{{@{{}}l{spec}@{{}}}}',
             r'\toprule',
             r'Mesh & {$T$} & {$|V|$} & {$\bar\lambda$} & {$\theta_{\min}$} & {$\theta_{5\%}$} & '
             r'{$\sigma_{\mathrm{val}}$} & {$d_{\mathrm{RMS}}$} & {$t$ (ms)} \\', r'\midrule']
    for index, group in enumerate(groups):
        if index:
            lines.append(r'\addlinespace[2pt]')
        lines += [' & '.join(row) + r' \\' for row in group]
    lines += [r'\bottomrule', r'\end{tabular}',
              r'\caption{Absolute values of the five-cycle baseline, the reference of all ratios: number of vertices '
              r'$|V|$, average edge loss $\bar\lambda$, smallest angle $\theta_{\min}$ and fifth-percentile angle '
              r'$\theta_{5\%}$ in degrees, valence error $\sigma_{\mathrm{val}}$, surface deviation '
              r'$d_{\mathrm{RMS}}$ in units of $10^{-3}$, and runtime $t$.}', r'\label{tab:eval-baseline}',
              r'\end{table}']
    return lines


def appendix_b(data, out):
    lines = ['% Generated by scripts/generate_results.py: edit the script, not this file.',
             r'\chapter{Evaluation Data}', r'\label{app:evaluation-data}', '',
             *table_baseline(data), '', *table_all_configurations(data)]
    (out / 'B-evaluation-data.tex').write_text('\n'.join(lines) + '\n')


def queues_heatmap(data, out):
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(TEXT_WIDTH_INCHES, 1.5))
        fig.subplots_adjust(left=.085, right=.945, bottom=.03, top=.74)
        medians = np.array([[data.ratio(data.runs(approach, objective), LOSS).xs(target, level='target').median()
                             for approach in ['dynamic-queue', 'batched'] for objective in OBJECTIVES]
                            for target in TARGETS])
        image = ax.imshow(medians, cmap='YlOrRd', norm=LogNorm(.8, 8), aspect='auto')
        for row, column in np.ndindex(medians.shape):
            ax.text(column, row, f'{medians[row, column]:.2f}', ha='center', va='center', fontsize=9,
                    color='white' if medians[row, column] > 3 else 'black')
        ax.set_yticks(range(3), ['$T=0.75$', '$T=1$', '$T=1.5$'])
        ax.xaxis.tick_top()
        ax.set_xticks(range(8), ['Sum', 'Max.', 'Local avg.', 'Global avg.'] * 2, fontsize=7)
        ax.tick_params(length=0)
        for index, label in enumerate(['Dynamic', 'Batch']):
            ax.text(1.5 + 4 * index, 1.17, label, transform=ax.get_xaxis_transform(), ha='center', va='bottom')
        ax.axvline(3.5, color='white', lw=2)
        bar = fig.colorbar(image, ax=ax, fraction=.04, pad=.03, label=r'$\bar\lambda$ / baseline')
        bar.set_ticks([1, 2, 4, 8], labels=['1', '2', '4', '8'])
        bar.ax.minorticks_off()
        fig.savefig(out / 'queues-heatmap.pdf')
        plt.close(fig)


KNEE_STYLE = {
    'font.family': 'serif', 'font.serif': ['Latin Modern Roman', 'CMU Serif', 'Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm', 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6,
    'ytick.labelsize': 6, 'legend.fontsize': 6, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.grid': True, 'grid.alpha': 0.25,
    'grid.linewidth': 0.5, 'legend.frameon': False, 'pdf.fonttype': 42,
}


def knee_trace(data, out):
    ink, weak, bar = '#303C44', '#D77D43', '#9AA3AB'
    trace = knee_figure_trace(data)
    window, windows, knee = knee_windows(trace)
    by_edits = trace.drop_duplicates('edits').set_index('edits').average_edge_loss
    with plt.rc_context(KNEE_STYLE):
        fig, (ax_loss, ax_gain) = plt.subplots(1, 2, figsize=(4.79, 1.85))
        edits = by_edits.index / 1000
        start_loss, final_loss = by_edits.iloc[0], by_edits.iloc[-1]
        knee_edits, knee_loss = knee.end / 1000, by_edits[int(knee.end)]
        ax_loss.plot(edits, by_edits, color=ink, lw=1.0)
        ax_loss.plot([0], [start_loss], 'o', ms=2.6, color=ink, zorder=3, clip_on=False)
        ax_loss.annotate('start\n' rf'$\bar\lambda={start_loss:.3f}$', (0, start_loss),
                         xytext=(14, start_loss - 0.022), fontsize=6, color=ink,
                         arrowprops=dict(arrowstyle='-', color=ink, lw=0.5))
        ax_loss.vlines(knee_edits, 0, knee_loss, color=weak, lw=0.7, ls='--')
        ax_loss.plot([knee_edits], [knee_loss], 'o', ms=2.8, color=weak, zorder=3)
        ax_loss.annotate('knee\n' rf'$\bar\lambda={knee_loss:.3f}$', (knee_edits, knee_loss),
                         xytext=(knee_edits + 12, knee_loss + 0.03), fontsize=6, color=weak,
                         arrowprops=dict(arrowstyle='-', color=weak, lw=0.5))
        ax_loss.plot([edits[-1]], [final_loss], 's', ms=2.6, color=ink, zorder=3)
        ax_loss.annotate('no improving candidate\n' rf'$\bar\lambda={final_loss:.3f}$', (edits[-1], final_loss),
                         xytext=(edits[-1], final_loss + 0.04), ha='right', fontsize=6, color=ink,
                         arrowprops=dict(arrowstyle='-', color=ink, lw=0.5))
        ax_loss.set_xlabel('executed operations (thousands)')
        ax_loss.set_ylabel(r'$\bar\lambda$')
        ax_loss.set_title('(a) dynamic queue', loc='left')
        ax_loss.set_ylim(bottom=0)
        ax_loss.set_xlim(0, edits[-1] * 1.02)

        # Windows before the knee, the first weak window (which stops a knee stage), and windows a stage never runs.
        colors = [bar if end < knee.end else weak if end == knee.end else '#D9DCDF' for end in windows.end]
        ax_gain.bar((windows.start + windows.end) / 2000, windows.gain * 100,
                    width=(windows.end - windows.start) / 1000 * 0.85, color=colors, lw=0)
        ax_gain.step(windows.end / 1000, windows.threshold * 100, where='pre', color=ink, lw=0.7,
                     label=rf'${KNEE_RATE_FRACTION:g}\times$ best window so far')
        ax_gain.set_xlim(0, knee.end / 1000 * 2.2)
        ax_gain.set_xlabel('executed operations (thousands)')
        ax_gain.set_ylabel(r'relative decrease of $\bar\lambda$ (\%)')
        ax_gain.set_title(rf'(b) windows of $K={window}$ operations', loc='left')
        ax_gain.legend(loc='upper right')
        fig.tight_layout(w_pad=1.2)
        fig.savefig(out / 'knee-trace.pdf', bbox_inches='tight')
        plt.close(fig)


def cycle_panel(ax, data, tier, stops, target, reference_tier=None):
    """The loss over k >= 5 for every mesh and its median, relative to five baseline cycles."""
    counts = list(range(5, 51))
    meshes = sorted(data.core.input.unique())
    reference = data.reference[LOSS].xs(target, level='target')

    def paths(rows):
        rows = rows[(rows.target == target) & (rows.cycles >= 5)]
        table = rows.pivot(index='input', columns='cycles', values=LOSS).loc[meshes, counts]
        return table.div(reference.loc[meshes], axis=0).to_numpy()

    lines = paths(tier)
    for mesh, values in zip(meshes, lines):
        named = target == 1 and mesh in ['horse', 'elk']
        ax.plot(counts, values, color='.4' if named else '#b5bec6', lw=.85 if named else .55, alpha=.9)
        if named:
            ax.annotate(mesh.capitalize(), (counts[-1], values[-1]), xytext=(-2, 5 if mesh == 'elk' else 6),
                        textcoords='offset points', ha='right', fontsize=7.5, color='.25')
    if reference_tier is not None:
        ax.plot(counts, np.median(paths(reference_tier), axis=0), color='.15', ls='--', lw=1)
    ax.plot(counts, np.median(lines, axis=0), color=BLUE, lw=1.6)
    stopped = stops.xs(target, level='target')
    ax.scatter(stopped.cycles, stopped[LOSS] / reference.loc[stopped.index], s=10, facecolor='white', edgecolor='.15',
               lw=.6, zorder=5)
    ax.axvline(10, color=ORANGE, ls=':', lw=1)
    ax.set_xlim(5, 50)
    ax.set_ylim(.65, 1.15)
    ax.set_xticks([5, 10, 25, 50])
    ax.grid(axis='y', color='.92', lw=.5)


def cycles_combined(data, out):
    handles = [Line2D([], [], color='#b5bec6', lw=.8, label='Mesh'), Line2D([], [], color=BLUE, lw=1.6, label='Median'),
               Line2D([], [], color='.15', ls='--', lw=1, label='Baseline median'),
               Line2D([], [], ls='', marker='o', ms=3.2, mfc='white', mec='.15', mew=.6, label='Convergent stop')]
    fixed, ordered = data.tiers['cycles'], data.tiers['ordered-cycles']
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH_INCHES, 3.75), sharex=True, sharey=True)
        fig.subplots_adjust(left=.09, right=.985, bottom=.155, top=.945, wspace=.1, hspace=.17)
        for column, target in enumerate(TARGETS):
            cycle_panel(axes[0, column], data, fixed, data.runs('convergent'), target)
            axes[0, column].set_title(f'Baseline, $T={target:g}$', pad=3)
            cycle_panel(axes[1, column], data, ordered, data.runs('ordered'), target, fixed)
            axes[1, column].set_title(f'Ordered, $T={target:g}$', pad=3)
            axes[1, column].set_xlabel('Cycles $k$')
        fig.text(.012, (.155 + .945) / 2, r'$\bar\lambda$ / baseline', rotation=90, ha='left', va='center')
        fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=7, handlelength=1.6,
                   columnspacing=1.4, borderaxespad=.2)
        fig.savefig(out / 'cycles-combined.pdf')
        plt.close(fig)


OUTPUTS = {'claims': claims, 'comparison-table': comparison_table, 'appendix-b': appendix_b,
           'queues-heatmap': queues_heatmap, 'knee-trace': knee_trace, 'cycles-combined': cycles_combined}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data', type=Path, default=root / 'data')
    parser.add_argument('--out', type=Path, default=root / 'results')
    parser.add_argument('--runs', type=Path, help='the output folder of run_experiments.py, for the images')
    parser.add_argument('--only', nargs='+', choices=[*OUTPUTS, *result_images.FIGURES], metavar='NAME')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    selected = args.only or [*OUTPUTS, *(result_images.FIGURES if args.runs else [])]
    data = Dataset(args.data)
    for name in selected:
        if name in OUTPUTS:
            OUTPUTS[name](data, args.out)
        elif args.runs:
            result_images.FIGURES[name](data, args.runs, args.out)
        else:
            parser.error(f'{name} needs the output meshes: --runs FOLDER')
        print('wrote', name, flush=True)


if __name__ == '__main__':
    main()
