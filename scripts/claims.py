#!/usr/bin/env python3
"""Generation of the numbers in the thesis based on the dataset.

usage: claims.py [--data DATA]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from shared.dataset import DEVIATION, LOSS, OBJECTIVES, ROOT, TARGETS, TIME, Dataset
from shared.knee import knee_figure_trace, knee_windows
from shared.style import MESH_NAMES

DELTA_C = 1e-3  # the stagnation tolerance of Sec. 3.1.4
DELTA_C_TESTS = [0, 1e-3, 3e-3, 1e-2]
QUEUE_APPROACHES = ['dynamic-queue', 'batched', 'hybrid', 'alternating']


def print_indent(message, spaces=2):
    """Print a message with an indent, e.g. for the thesis."""
    print(f'{" " * spaces}{message}')


def claim(name, value, level=1):
    """Print a claim and its value. A dict prints as a group: the name, then one indented line per key."""
    if isinstance(value, dict):
        print_indent(f'{name}:', 2 * level)
        for key, item in value.items():
            claim(key, item, level + 1)
    else:
        print_indent(f'{name}: {value}', 2 * level)


def count(condition):
    """The number of cases where a condition holds, e.g. count(ratios < 1)."""
    return int(condition.sum())


def at_target(rows, target):
    """The rows of one target, one per mesh."""
    return rows.xs(target, level='target')


def percent(fraction, digits=1):
    return f'{100 * fraction:.{digits}f}%'


def signed_percent(fraction, digits=1):
    return f'{100 * fraction:+.{digits}f}%'


def value_range(values, digits=2, scale=1, unit=''):
    """'lowest to highest', e.g. over the four objectives."""
    values = [scale * value for value in values]
    return f'{min(values):.{digits}f}{unit} to {max(values):.{digits}f}{unit}'


def introduction(data):
    """Fig. iteration-comparison and the paragraph after it. scans.csv has the edges outside the length band of these
    baseline runs."""
    scans = data.scans[(data.scans.tier == 'cycles') & (data.scans.input == 'stanford-bunny')]
    for k in [5, 25]:
        outside = {}
        for target in [1.0, 1.5]:
            scan = scans[(scans.cycles == k) & (scans.target == target)].iloc[0]
            outside[f'T = {target:g}'] = percent(scan.long_fraction + scan.short_fraction)
        claim(f'Bunny: edges outside the length thresholds after {k} baseline cycles', outside)


def knee_figure(data):
    """The caption of the knee figure, replayed on the traced run (shared/knee.py)."""
    trace = knee_figure_trace(data)
    window, _, knee = knee_windows(trace)
    loss = trace.drop_duplicates('edits').set_index('edits').average_edge_loss  # the average loss after n edits
    claim('Edits without knee', int(loss.index.max()))
    claim('Edits with knee', int(knee.end))
    claim('Window size', window)
    claim('Further decrease of the average loss after the knee', percent(1 - loss.iloc[-1] / loss[int(knee.end)], 0))


def setup(data):
    core, inputs = data.core, data.inputs
    claim('Experiment stats', {
            '# of strategy-objective pairs': core.groupby(['approach', 'objective']).ngroups,
            '# of mesh-target pairs': core.groupby(['input', 'target']).ngroups,
            '# of runs': len(core),
            '# of failed runs': len(data.failures),
            'longest run (cap 900s)': f'{core.algorithm_time_ms_max.max() / 1000:.1f}s',
            'most cycles / rounds': int(core.cycles.max()),
            'stop reasons': {reason: int(n) for reason, n in core.stop_reason.value_counts().items()}})          
    print()

    smallest = inputs.loc[['elk', 'stanford-bunny-low']]
    others = inputs.drop(smallest.index)
    ratios = [a / b for a in smallest.average_edge_length for b in others.average_edge_length]
    claim('Smallest inputs', {
        'Elk edges': int(smallest.loc['elk'].edges),
        'Bunny (low) edges': int(smallest.loc['stanford-bunny-low'].edges),
        'Lowest edges of any other input': int(others.edges.min()),
        'Elk / Bunny (low) avg. edge length vs other inputs': f'{value_range(ratios, 0)} x'})
    print()

    # The baseline on seven reorderings of each input, against the original order: the largest change per case.
    orders = data.tiers['input-order'].set_index(['input', 'target'])
    change = (orders[LOSS] / data.reference[LOSS] - 1).abs()
    largest = change.groupby(level=['input', 'target']).max()
    mesh, target = largest.idxmax()
    claim('Baseline on reordered inputs', {
        '# of further orders': orders.order.nunique(),
        'largest loss change per case, median': percent(largest.median()),
        'largest loss change per case, worst': f'{percent(largest.max())} ({MESH_NAMES[mesh]}, T = {target:g})'})


def queues(data):
    def heatmap_cell(approach, objective, target):
        return f'{at_target(data.ratio(data.runs(approach, objective), LOSS), target).median():.2f}'

    claim('Median loss ratio (Fig. queue-heatmap)', {
        approach: {o: {f'T = {t:g}': heatmap_cell(approach, o, t) for t in TARGETS} for o in OBJECTIVES}
        for approach in ['dynamic-queue', 'batched']})
    print()

    claim('Runs ended by the safeguard', {
        approach: {o: count(data.runs(approach, o).stop_reason == 'safeguard') for o in OBJECTIVES}
        for approach in ['dynamic-queue', 'hybrid']})
    queue_runs = data.core[data.core.approach.isin(['dynamic-queue', 'batched'])]
    claim('Dynamic and batched queues: runs ending with an empty queue (of 240)',
          count(queue_runs.stop_reason == 'no-candidate'))
    print()

    bunny = ('stanford-bunny', 1.5)
    claim('Bunny, T = 1.5, dynamic queue: loss vs baseline', {
        o: f"{data.ratio(data.runs('dynamic-queue', o), LOSS).loc[bunny]:.1f} x" for o in ['local-average', 'maximum']})

    # scans.csv: per output, the edges outside the length band and, of the short ones, those whose collapse the length
    # check blocks.
    bunny = ('stanford-bunny', 0.75)
    claim('Bunny, T = 0.75: edges too long', {
        'dynamic queue, sum': percent(data.scan('dynamic-queue', 'sum').long_fraction.loc[bunny]),
        'baseline': percent(data.scan('baseline').long_fraction.loc[bunny])})
    failing = ['local-average', 'maximum']

    def blocked(objective):
        scan = at_target(data.scan('dynamic-queue', objective), 1.5)
        return percent(scan.collapse_invalid_length.sum() / scan.collapse_candidates.sum(), 0)

    claim('T = 1.5, dynamic queue', {
        'edges too short (median over meshes)':
            {o: percent(at_target(data.scan('dynamic-queue', o).short_fraction, 1.5).median(), 0) for o in failing},
        'short edges whose collapse the length check blocks (pooled)': {o: blocked(o) for o in failing}})
    print()

    claim('Outputs with an angle below 1 degree (of 30)', {
        'dynamic queue': {o: count(data.runs('dynamic-queue', o).minimum_angle < 1) for o in OBJECTIVES},
        'baseline': count(data.reference.minimum_angle < 1)})
    claim('Dynamic queue, local average: outputs with an angle below 0.001 degrees (of 30)',
          count(data.runs('dynamic-queue', 'local-average').minimum_angle < 0.001))
    print()

    def edit_ratio(approach, objective, operation):
        """The median over meshes at T = 1.5 of an operation count relative to the baseline's."""
        return f'{at_target(data.ratio(data.runs(approach, objective), operation), 1.5).median():.2f}'

    pure_queues = ['dynamic-queue', 'batched']
    claim('T = 1.5: operations vs baseline (median over meshes)', {
        'collapses, local average': {a: edit_ratio(a, 'local-average', 'collapses') for a in pure_queues},
        'flips, sum': {a: edit_ratio(a, 'sum', 'flips') for a in pure_queues}})


def alternating_stages(data):
    """From the traces of the alternating runs: the relative change of the average loss in every cycle, and in every
    queue stage that follows a cycle. Each cycle starts a new round."""
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


def combined_schedules(data):
    alternating = {o: data.runs('alternating', o) for o in OBJECTIVES}
    rollbacks = data.traces[(data.traces.approach == 'alternating') & (data.traces.event == 'rollback')]
    stages = alternating_stages(data)
    cycles = stages[stages.event == 'cycle']
    later_stages = stages[(stages.event == 'stage') & (stages['round'] > 1)]
    losses = pd.DataFrame({o: alternating[o][LOSS] for o in OBJECTIVES})
    lowest = losses.idxmin(axis=1)  # the objective with the lowest loss, per case
    claim('Alternating', {
        'median loss ratio (over objectives)':
            value_range([data.median_ratio(alternating[o], LOSS) for o in OBJECTIVES]),
        'cases below five cycles (of 30)': {o: count(data.ratio(alternating[o], LOSS) < 1) for o in OBJECTIVES},
        'cases below ten cycles (of 30)':
            {o: count(data.ratio(alternating[o], LOSS, data.fixed(10)) < 1) for o in OBJECTIVES},
        'runs undoing their last round (of 30, over objectives)':
            value_range([count(rollbacks.objective == o) for o in OBJECTIVES], 0),
        'median loss change': {
            'first cycle': signed_percent(cycles[cycles['round'] == 1].change.median(), 0),
            'later cycles': signed_percent(cycles[cycles['round'] > 1].change.median(), 0)},
        'median loss decrease in a stage after a later cycle (over objectives)':
            value_range([-later_stages[later_stages.objective == o].change.median() for o in OBJECTIVES], 1, 100, '%'),
        'cases with the lowest loss': {o: count(lowest == o) for o in OBJECTIVES},
        'faces below 0.001 degrees (all 30 outputs)':
            {o: int(alternating[o].faces_below_0_001_degree.sum()) for o in OBJECTIVES}})


def stagnation_rule(data, delta):
    """The stagnation rule of Sec. 3.1.4 with its rollback, replayed on the baseline stopped after k = 1 to 50 cycles.
    Per case: the cycles it keeps, their loss, whether it undid a cycle that raised the loss, and the lowest loss of any
    later cycle. The first cycle lowers the loss far more than any tested delta, so the replay starts at the second."""
    rows = {}
    for case, fixed in data.tiers['cycles'].groupby(['input', 'target']):
        loss = fixed.set_index('cycles')[LOSS]  # the loss after k cycles
        stop = 2
        while stop < 50 and (loss[stop - 1] - loss[stop]) / loss[stop - 1] >= delta:
            stop += 1
        kept = stop - 1 if loss[stop] > loss[stop - 1] else stop
        rows[case] = dict(kept=kept, kept_loss=loss[kept], undone=kept < stop, undone_rise=loss[stop] / loss[kept] - 1,
                          best_later=loss.loc[kept + 1:].min() if kept < 50 else np.nan)
    rule = pd.DataFrame.from_dict(rows, orient='index')
    rule.index = pd.MultiIndex.from_tuples(rule.index, names=['input', 'target'])
    rule['kept_ratio'] = rule.kept_loss / data.reference[LOSS]
    return rule


def repeated_cycles(data):
    ten = data.fixed(10)
    ten_to_five = data.ratio(ten, LOSS)
    fifty_to_ten = data.ratio(data.fixed(50), LOSS, ten)
    claim('Fixed cycles', {
        '10 vs 5: median loss decrease': percent(1 - ten_to_five.median(), 0),
        '10 vs 5: cases lower (of 30)': count(ten_to_five < 1),
        '50 vs 10: median loss decrease': percent(1 - fifty_to_ten.median(), 0),
        '50 vs 10 at T = 1': {MESH_NAMES[mesh]: signed_percent(fifty_to_ten.loc[(mesh, 1.0)] - 1, 0)
                              for mesh in ['horse', 'elk']}})
    print()

    repeated, ordered = data.runs('repeated'), data.runs('ordered')
    rule = stagnation_rule(data, DELTA_C)
    assert (rule.kept == repeated.cycles.loc[rule.index]).all(), 'the replay disagrees with the repeated runs'
    claim('Repeated', {
        'kept cycles': {'median': f'{repeated.cycles.median():g}', 'fewest': int(repeated.cycles.min()),
                        'most': int(repeated.cycles.max())},
        'median loss decrease from five cycles': percent(1 - data.median_ratio(repeated, LOSS), 0),
        'runs where a later fixed cycle (up to 50) is > 1% lower (of 30)':
            count(rule.best_later < .99 * rule.kept_loss),
        'runs undoing their last cycle (of 30)': count(rule.undone),
        'median loss rise undone': percent(rule[rule.undone].undone_rise.median())})
    print()

    # Each rule against a fixed count at its median cycle count, in loss, time and case by case.
    stop, ordered_stop = int(repeated.cycles.median()), int(ordered.cycles.median())
    same_count = {f'Repeated vs {stop} fixed cycles': (repeated, data.fixed(stop)),
                  f'Ordered vs {ordered_stop} fixed ordered cycles': (ordered, data.fixed(ordered_stop, ordered=True))}
    for name, (runs, fixed) in same_count.items():
        ratios = data.ratio(runs, LOSS, fixed)
        claim(name, {
            'median loss ratio, rule': f'{data.median_ratio(runs, LOSS):.3f}',
            'median loss ratio, fixed': f'{data.median_ratio(fixed, LOSS):.3f}',
            'median time ratio, rule': f'{data.median_ratio(runs, TIME):.2f}',
            'median time ratio, fixed': f'{data.median_ratio(fixed, TIME):.2f}',
            'cases lower': count(ratios < 1),
            'cases higher': count(ratios > 1),
            'cases equal': count(ratios == 1),
            'median absolute loss difference': percent((ratios - 1).abs().median(), 2)})
    print()

    rules = {delta: stagnation_rule(data, delta) for delta in DELTA_C_TESTS}
    claim('Repeated with other delta_c', {
        f'delta_c = {delta:g}': {'median loss ratio': f'{rules[delta].kept_ratio.median():.3f}',
                                 'median kept cycles': f'{rules[delta].kept.median():g}'}
        for delta in DELTA_C_TESTS})
    print()

    # The median loss of the baseline over the original and the seven reordered inputs, per case.
    orders = pd.concat([data.tiers['input-order'].set_index(['input', 'target'])[LOSS], data.reference[LOSS]])
    order_median = orders.groupby(level=['input', 'target']).median()
    ordered_five = data.fixed(5, ordered=True)[LOSS] / order_median
    ordered_to_repeated = data.ratio(ordered, LOSS, repeated)
    claim('Ordered', {
        'five cycles vs median of the eight baseline orders': {
            'cases lower (of 30)': count(ordered_five < 1),
            'median loss ratio': f'{ordered_five.median():.3f}'},
        'vs repeated': {
            'cases lower (of 30)': count(ordered_to_repeated < 1),
            'median loss ratio': f'{ordered_to_repeated.median():.3f}'}})
    print()

    elk = ('elk', 1.0)
    claim('Elk, T = 1', {
        'loss after 5 cycles': f'{data.reference.loc[elk, LOSS]:.3f}',
        'loss after 25 cycles': f'{data.fixed(25).loc[elk, LOSS]:.3f}',
        'deviation ratio after 25 cycles': f'{data.ratio(data.fixed(25), DEVIATION).loc[elk]:.1f}'})
    deviation = data.ratio(repeated, DEVIATION)
    other_meshes = deviation.drop('elk', level='input')
    mesh, target = other_meshes.idxmax()
    claim('Repeated: deviation ratio', {
        'largest, Elk': f"{deviation.xs('elk', level='input').max():.2f}",
        'largest, any other mesh': f'{other_meshes.max():.2f} ({MESH_NAMES[mesh]}, T = {target:g})',
        'median change': signed_percent(deviation.median() - 1, 0)})
    claim('Median deviation ratio', {
        approach: {**{o: f'{data.median_ratio(data.runs(approach, o), DEVIATION):.3f}' for o in OBJECTIVES},
                   f'baseline, k = {k}': f'{data.median_ratio(data.fixed(k), DEVIATION):.3f}'}
        for approach, k in [('dynamic-queue', 1), ('hybrid', 2), ('alternating', 5)]})


def runtime(data):
    repeated, ordered = data.runs('repeated'), data.runs('ordered')
    claim('Median time ratio', {
        **{f'{approach} (over objectives)':
               value_range([data.median_ratio(data.runs(approach, o), TIME) for o in OBJECTIVES], 1)
           for approach in QUEUE_APPROACHES},
        'alternating without knee vs alternating (over objectives)':
            value_range([data.median_ratio(data.no_knee(o), TIME, data.runs('alternating', o)) for o in OBJECTIVES]),
        'repeated': f'{data.median_ratio(repeated, TIME):.2f}',
        'ordered': f'{data.median_ratio(ordered, TIME):.2f}',
        'ten fixed cycles': f'{data.median_ratio(data.fixed(10), TIME):.1f}'})
    claim('Median kept cycles',
          {'repeated': f'{repeated.cycles.median():g}', 'ordered': f'{ordered.cycles.median():g}'})


def discussion(data):
    claim('Median time ratio', {
        'alternating (over objectives)':
            value_range([data.median_ratio(data.runs('alternating', o), TIME) for o in OBJECTIVES], 1),
        'repeated': f"{data.median_ratio(data.runs('repeated'), TIME):.2f}",
        'ordered': f"{data.median_ratio(data.runs('ordered'), TIME):.2f}"})
    claim('Cases below five cycles (of 30)', {
        approach: {o: count(data.ratio(data.runs(approach, o), LOSS) < 1) for o in OBJECTIVES}
        for approach in QUEUE_APPROACHES})


def conclusion(data):
    claim('Repeated: cases below five cycles (of 30)', count(data.ratio(data.runs('repeated'), LOSS) < 1))


SECTIONS = [('Ch. 1, Motivation (Fig. iteration-comparison)', introduction),
            ('Ch. 3, Alternating (Fig. knee-trace)', knee_figure),
            ('Ch. 4, Experimental Setup', setup),
            ('Ch. 4, Results: Queues', queues),
            ('Ch. 4, Results: Combined Schedules', combined_schedules),
            ('Ch. 4, Results: Repeated Cycles', repeated_cycles),
            ('Ch. 4, Results: runtime (last paragraph of Repeated Cycles)', runtime),
            ('Ch. 4, Discussion', discussion),
            ('Ch. 5, Conclusion', conclusion)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--data', type=Path, default=ROOT / 'data', help='the CSV files of run_experiments.py')
    data = Dataset(parser.parse_args().data)
    for title, section in SECTIONS:
        print(f'\n{title}')
        section(data)


if __name__ == '__main__':
    main()
