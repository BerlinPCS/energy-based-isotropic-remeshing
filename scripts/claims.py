#!/usr/bin/env python3
"""How each number of the thesis follows from the data.

Most numbers are matched ratios (Sec. 4.1): a run divided by five baseline cycles on the same mesh and target, and then
the median over the 30 mesh-target cases. shared/dataset.py loads the CSV files in data/ and lines the runs up for these
ratios. The script writes nothing.

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


def claim(statement, value):
    """Prints one claim. A dict, such as one value per objective, prints as 'key: value, ...'."""
    if isinstance(value, dict):
        value = ', '.join(f'{key}: {item}' for key, item in value.items())
    print(f'  {statement}\n      {value}')


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


def abstract(data):
    ordered = data.runs('ordered')
    claim('Ordered: median loss ratio to five cycles, cases below five cycles (of 30), median time ratio',
          f'{data.median_ratio(ordered, LOSS):.3f}, {count(data.ratio(ordered, LOSS) < 1)}, '
          f'{data.median_ratio(ordered, TIME):.2f}')


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
    claim('Edits until no acceptable candidate remains (Bunny, T = 1, dynamic queue, local average)',
          int(loss.index.max()))
    claim('Edits at which a knee-stopped stage would end, window K', f'{int(knee.end)}, {window}')
    claim('Further decrease of the average loss after the knee', percent(1 - loss.iloc[-1] / loss[int(knee.end)], 0))


def setup(data):
    core, inputs = data.core, data.inputs
    claim('Configurations of the main comparison, mesh-target cases',
          f"{core.groupby(['approach', 'objective']).ngroups}, {core.groupby(['input', 'target']).ngroups}")
    smallest = inputs.loc[['elk', 'stanford-bunny-low']]
    others = inputs.drop(smallest.index)
    claim('Vertices of the two smallest inputs (Elk, Bunny (low)), fewest of any other input',
          f"{', '.join(map(str, smallest.vertices))}, {others.vertices.min()}")
    ratios = [a / b for a in smallest.average_edge_length for b in others.average_edge_length]
    claim('Average edge length of the two smallest inputs relative to each other mesh', f'{value_range(ratios, 0)} x')

    # The baseline on seven reorderings of each input, against the original order: the largest change per case.
    orders = data.tiers['input-order'].set_index(['input', 'target'])
    change = (orders[LOSS] / data.reference[LOSS] - 1).abs()
    largest = change.groupby(level=['input', 'target']).max()
    mesh, target = largest.idxmax()
    claim('Baseline on reordered inputs: further orders; largest change of the loss per case, median and worst case',
          f'{orders.order.nunique()}; {percent(largest.median())}, {percent(largest.max())} '
          f'({MESH_NAMES[mesh]}, T = {target:g})')

    claim('Runs of the main comparison, failed runs in the whole dataset', f'{len(core)}, {len(data.failures)}')
    claim('Longest run of the main comparison in seconds (the safety cap is 900)',
          f'{core.algorithm_time_ms_max.max() / 1000:.1f}')
    claim('Why the runs of the main comparison stop (no run stops at the edit or cycle cap)',
          dict(core.stop_reason.value_counts()))
    claim('Most cycles or rounds of any run of the main comparison', int(core.cycles.max()))


def queues(data):
    for approach in ['dynamic-queue', 'batched']:
        for objective in OBJECTIVES:
            ratios = data.ratio(data.runs(approach, objective), LOSS)
            claim(f'Fig. queue-heatmap, {approach}, {objective}: median loss ratio',
                  {f'T = {t:g}': f'{at_target(ratios, t).median():.2f}' for t in TARGETS})

    def stopped_by_safeguard(approach):
        return {o: count(data.runs(approach, o).stop_reason == 'safeguard') for o in OBJECTIVES}

    claim('Dynamic queue: runs ended by the safeguard', stopped_by_safeguard('dynamic-queue'))
    queue_runs = data.core[data.core.approach.isin(['dynamic-queue', 'batched'])]
    claim('Dynamic and batched queues: runs that end with an empty queue (of 240)',
          count(queue_runs.stop_reason == 'no-candidate'))
    claim('Hybrid: runs ended by the safeguard', stopped_by_safeguard('hybrid'))

    bunny = ('stanford-bunny', 1.5)
    claim('Bunny, T = 1.5: loss relative to the baseline, dynamic queue',
          {o: f"{data.ratio(data.runs('dynamic-queue', o), LOSS).loc[bunny]:.1f} x"
           for o in ['local-average', 'maximum']})

    # scans.csv: per output, the edges outside the length band and, of the short ones, those whose collapse the length
    # check blocks.
    bunny = ('stanford-bunny', 0.75)
    claim('Bunny, T = 0.75: edges too long', {
        'dynamic queue with sum': percent(data.scan('dynamic-queue', 'sum').long_fraction.loc[bunny]),
        'baseline': percent(data.scan('baseline').long_fraction.loc[bunny])})
    failing = ['local-average', 'maximum']
    claim('T = 1.5, dynamic queue: edges too short (median over meshes)',
          {o: percent(at_target(data.scan('dynamic-queue', o).short_fraction, 1.5).median(), 0) for o in failing})

    def blocked(objective):
        scan = at_target(data.scan('dynamic-queue', objective), 1.5)
        return percent(scan.collapse_invalid_length.sum() / scan.collapse_candidates.sum(), 0)

    claim('T = 1.5, dynamic queue: short edges whose collapse the length check blocks (all meshes pooled)',
          {o: blocked(o) for o in failing})

    # The gains of Fig. horse-cap, such as the split's g_sum = -12.43, were measured once and are written into
    # figures/views_horse_cap.py.

    claim('Dynamic queue: outputs with an angle below 1 degree (of 30)',
          {o: count(data.runs('dynamic-queue', o).minimum_angle < 1) for o in OBJECTIVES})
    claim('Baseline: outputs with an angle below 1 degree', count(data.reference.minimum_angle < 1))
    claim('Dynamic queue, local average: outputs with an angle below 0.001 degrees',
          count(data.runs('dynamic-queue', 'local-average').minimum_angle < 0.001))

    def edit_ratio(approach, objective, operation):
        """The median over meshes at T = 1.5 of an operation count relative to the baseline's."""
        return f'{at_target(data.ratio(data.runs(approach, objective), operation), 1.5).median():.2f}'

    claim('T = 1.5, local average: collapses relative to the baseline',
          {a: edit_ratio(a, 'local-average', 'collapses') for a in ['dynamic-queue', 'batched']})
    claim('T = 1.5, sum: flips relative to the baseline',
          {a: edit_ratio(a, 'sum', 'flips') for a in ['dynamic-queue', 'batched']})


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
    claim('Hybrid: median loss ratio',
          {o: f"{data.median_ratio(data.runs('hybrid', o), LOSS):.2f}" for o in OBJECTIVES})
    claim('Alternating: cases below ten baseline cycles (of 30)',
          {o: count(data.ratio(alternating[o], LOSS, data.fixed(10)) < 1) for o in OBJECTIVES})
    claim('Alternating: median loss ratio, lowest to highest objective',
          value_range([data.median_ratio(alternating[o], LOSS) for o in OBJECTIVES]))
    claim('Alternating: cases below five baseline cycles (of 30)',
          {o: count(data.ratio(alternating[o], LOSS) < 1) for o in OBJECTIVES})

    # A run that undoes its last round logs a rollback in its trace.
    rollbacks = data.traces[(data.traces.approach == 'alternating') & (data.traces.event == 'rollback')]
    claim('Alternating: runs that undo their last round (of 30), fewest to most over the objectives',
          value_range([count(rollbacks.objective == o) for o in OBJECTIVES], 0))

    stages = alternating_stages(data)
    cycles = stages[stages.event == 'cycle']
    claim('Alternating: median change of the loss', {
        'first cycle': signed_percent(cycles[cycles['round'] == 1].change.median(), 0),
        'later cycles': signed_percent(cycles[cycles['round'] > 1].change.median(), 0)})
    later_stages = stages[(stages.event == 'stage') & (stages['round'] > 1)]
    claim('Alternating: median decrease of the loss in a queue stage after a later cycle, lowest to highest objective',
          value_range([-later_stages[later_stages.objective == o].change.median() for o in OBJECTIVES], 1, 100, '%'))

    losses = pd.DataFrame({o: alternating[o][LOSS] for o in OBJECTIVES})
    lowest = losses.idxmin(axis=1)  # the objective with the lowest loss, per case
    claim('Alternating: cases where each objective has the lowest loss', {o: count(lowest == o) for o in OBJECTIVES})
    claim('Alternating: faces below 0.001 degrees in all 30 outputs',
          {o: int(alternating[o].faces_below_0_001_degree.sum()) for o in OBJECTIVES})


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
    claim('Ten cycles: median decrease of the loss from five, cases lower (of 30)',
          f'{percent(1 - ten_to_five.median(), 0)}, {count(ten_to_five < 1)}')
    claim('Fifty cycles: median decrease of the loss from ten', percent(1 - fifty_to_ten.median(), 0))
    claim('Fifty cycles relative to ten at T = 1',
          {MESH_NAMES[mesh]: signed_percent(fifty_to_ten.loc[(mesh, 1.0)] - 1, 0) for mesh in ['horse', 'elk']})

    repeated, ordered = data.runs('repeated'), data.runs('ordered')
    claim('Repeated: median kept cycles, fewest, most',
          f'{repeated.cycles.median():g}, {int(repeated.cycles.min())}, {int(repeated.cycles.max())}')
    claim('Repeated: median decrease of the loss from five cycles', percent(1 - data.median_ratio(repeated, LOSS), 0))

    # Each rule against a fixed count at its median cycle count, in loss, time and case by case.
    stop, ordered_stop = int(repeated.cycles.median()), int(ordered.cycles.median())
    same_count = {f'Repeated against {stop} fixed cycles': (repeated, data.fixed(stop)),
                  f'Ordered against {ordered_stop} fixed ordered cycles':
                      (ordered, data.fixed(ordered_stop, ordered=True))}
    for name, (runs, fixed) in same_count.items():
        claim(f'{name}: median loss ratio to five cycles, median time ratio to five cycles',
              f'{data.median_ratio(runs, LOSS):.3f} against {data.median_ratio(fixed, LOSS):.3f}, '
              f'{data.median_ratio(runs, TIME):.2f} against {data.median_ratio(fixed, TIME):.2f}')
    differences = {}
    for name, (runs, fixed) in same_count.items():
        ratios = data.ratio(runs, LOSS, fixed)
        claim(f'{name}: cases lower, higher, equal', f'{count(ratios < 1)}, {count(ratios > 1)}, {count(ratios == 1)}')
        differences[name] = percent((ratios - 1).abs().median(), 2)
    claim('Each rule against its fixed count: median absolute difference of the loss', differences)

    rule = stagnation_rule(data, DELTA_C)
    assert (rule.kept == repeated.cycles.loc[rule.index]).all(), 'the replay disagrees with the repeated runs'
    claim('Repeated: runs where a later fixed cycle (up to 50) is more than 1% below the result (of 30)',
          count(rule.best_later < .99 * rule.kept_loss))
    claim('Repeated: runs that undo their last cycle (of 30), median rise of the loss it undoes',
          f'{count(rule.undone)}, {percent(rule[rule.undone].undone_rise.median())}')
    rules = {delta: stagnation_rule(data, delta) for delta in DELTA_C_TESTS}
    claim('Repeated, replayed with other delta_c: median loss ratio to five cycles',
          {f'delta_c = {delta:g}': f'{rules[delta].kept_ratio.median():.3f}' for delta in DELTA_C_TESTS})
    claim('Repeated, replayed with other delta_c: median kept cycles',
          {f'delta_c = {delta:g}': f'{rules[delta].kept.median():g}' for delta in DELTA_C_TESTS})

    # The median loss of the baseline over the original and the seven reordered inputs, per case.
    orders = pd.concat([data.tiers['input-order'].set_index(['input', 'target'])[LOSS], data.reference[LOSS]])
    order_median = orders.groupby(level=['input', 'target']).median()
    ordered_five = data.fixed(5, ordered=True)[LOSS] / order_median
    claim('Ordered, five cycles: cases below the median of the eight baseline orders (of 30), median ratio to it',
          f'{count(ordered_five < 1)}, {ordered_five.median():.3f}')
    ordered_to_repeated = data.ratio(ordered, LOSS, repeated)
    claim('Ordered against repeated: cases lower (of 30), median loss ratio',
          f'{count(ordered_to_repeated < 1)}, {ordered_to_repeated.median():.3f}')

    elk = ('elk', 1.0)
    claim('Elk, T = 1: loss after 25 and after 5 cycles',
          f'{data.fixed(25).loc[elk, LOSS]:.3f}, {data.reference.loc[elk, LOSS]:.3f}')
    claim('Elk, T = 1: deviation after 25 cycles relative to 5',
          f'{data.ratio(data.fixed(25), DEVIATION).loc[elk]:.1f}')
    deviation = data.ratio(repeated, DEVIATION)
    other_meshes = deviation.drop('elk', level='input')
    mesh, target = other_meshes.idxmax()
    claim('Repeated: largest deviation ratio', {
        'Elk': f"{deviation.xs('elk', level='input').max():.2f}",
        'any other mesh': f'{other_meshes.max():.2f} ({MESH_NAMES[mesh]}, T = {target:g})'})
    claim('Repeated: median change of the deviation', signed_percent(deviation.median() - 1, 0))
    for approach, k in [('dynamic-queue', 1), ('hybrid', 2), ('alternating', 5)]:
        medians = {o: f'{data.median_ratio(data.runs(approach, o), DEVIATION):.3f}' for o in OBJECTIVES}
        claim(f'{approach}: median deviation ratio, against the baseline after k = {k}',
              {**medians, f'k = {k}': f'{data.median_ratio(data.fixed(k), DEVIATION):.3f}'})


def runtime(data):
    for approach in QUEUE_APPROACHES:
        claim(f'{approach}: median time ratio to five cycles, lowest to highest objective',
              value_range([data.median_ratio(data.runs(approach, o), TIME) for o in OBJECTIVES], 1))
    claim('Alternating without the knee: median time relative to alternating, lowest to highest objective',
          value_range([data.median_ratio(data.no_knee(o), TIME, data.runs('alternating', o)) for o in OBJECTIVES]))
    repeated, ordered = data.runs('repeated'), data.runs('ordered')
    claim('Repeated, ordered: median time ratio to five cycles',
          f'{data.median_ratio(repeated, TIME):.2f}, {data.median_ratio(ordered, TIME):.2f}')
    claim('Repeated, ordered: median kept cycles', f'{repeated.cycles.median():g}, {ordered.cycles.median():g}')
    claim('Ten fixed cycles: median time ratio to five cycles', f'{data.median_ratio(data.fixed(10), TIME):.1f}')


def discussion(data):
    claim('Median time ratio to five cycles', {
        'alternating': value_range([data.median_ratio(data.runs('alternating', o), TIME) for o in OBJECTIVES], 1),
        'repeated': f"{data.median_ratio(data.runs('repeated'), TIME):.2f}",
        'ordered': f"{data.median_ratio(data.runs('ordered'), TIME):.2f}"})
    for approach in QUEUE_APPROACHES:
        claim(f'{approach}: cases below five cycles (of 30)',
              {o: count(data.ratio(data.runs(approach, o), LOSS) < 1) for o in OBJECTIVES})


def conclusion(data):
    claim('Repeated: cases below five cycles (of 30)', count(data.ratio(data.runs('repeated'), LOSS) < 1))


SECTIONS = [('Abstract', abstract), ('Ch. 1, Motivation (Fig. iteration-comparison)', introduction),
            ('Ch. 3, Alternating (Fig. knee-trace)', knee_figure), ('Ch. 4, Experimental Setup', setup),
            ('Ch. 4, Results: Queues', queues), ('Ch. 4, Results: Combined Schedules', combined_schedules),
            ('Ch. 4, Results: Repeated Cycles', repeated_cycles),
            ('Ch. 4, Results: runtime (last paragraph of Repeated Cycles)', runtime), ('Ch. 4, Discussion', discussion),
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
