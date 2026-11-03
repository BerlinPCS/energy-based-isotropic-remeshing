"""Appendix B (Evaluation Data): the absolute values of the five-cycle baseline (Table eval-baseline) and every
configuration relative to it (Table eval-all), as a whole chapter.

Writes B-evaluation-data.tex.
"""
from shared.dataset import ANGLE_TAIL, DEVIATION, LOSS, MIN_ANGLE, TARGETS, TIME, VALENCE
from shared.style import (ANGLE_TAIL_SYMBOL, DEVIATION_SYMBOL, LOSS_SYMBOL, MESH_NAMES, MIN_ANGLE_SYMBOL, TIME_SYMBOL,
                          VALENCE_SYMBOL, decimal_column)


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
    yield 'Until stagnation', [('Repeated', data.runs('repeated')), ('Ordered', data.runs('ordered'))]
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


def generate(data, out):
    lines = ['% Generated by scripts/figures/evaluation_data.py: edit the script, not this file.',
             r'\chapter{Evaluation Data}', r'\label{app:evaluation-data}', '',
             *table_baseline(data), '', *table_all_configurations(data)]
    (out / 'B-evaluation-data.tex').write_text('\n'.join(lines) + '\n')
