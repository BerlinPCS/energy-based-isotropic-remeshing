"""Fig. eval-cycles (Ch. 4): the average edge loss of the baseline (top) and the ordered phases (bottom) over the number
of cycles k, relative to five baseline cycles, for every mesh, with the cycle counts that the repeated and ordered runs
keep.

Writes cycles-combined.pdf.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from shared.dataset import LOSS, TARGETS
from shared.style import BLUE, ORANGE, PLOT_STYLE, TEXT_WIDTH_INCHES


def cycle_panel(ax, data, tier, stops, target, reference_tier=None):
    """The loss over k >= 4 for every mesh and its median, relative to five baseline cycles."""
    counts = list(range(4, 51))
    meshes = sorted(data.core.input.unique())
    reference = data.reference[LOSS].xs(target, level='target')

    def paths(rows):
        rows = rows[(rows.target == target) & (rows.cycles >= 4)]
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
    ax.set_xlim(4, 50)
    ax.set_ylim(.65, 1.15)
    ax.set_xticks([5, 10, 25, 50])
    ax.grid(axis='y', color='.92', lw=.5)


def generate(data, out):
    handles = [Line2D([], [], color='#b5bec6', lw=.8, label='Mesh'), Line2D([], [], color=BLUE, lw=1.6, label='Median'),
               Line2D([], [], color='.15', ls='--', lw=1, label='Baseline median'),
               Line2D([], [], ls='', marker='o', ms=3.2, mfc='white', mec='.15', mew=.6, label='Stagnation rule')]
    fixed, ordered = data.tiers['cycles'], data.tiers['ordered-cycles']
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH_INCHES, 3.75), sharex=True, sharey=True)
        fig.subplots_adjust(left=.09, right=.985, bottom=.155, top=.945, wspace=.1, hspace=.17)
        for column, target in enumerate(TARGETS):
            cycle_panel(axes[0, column], data, fixed, data.runs('repeated'), target)
            axes[0, column].set_title(f'Baseline, $T={target:g}$', pad=3)
            cycle_panel(axes[1, column], data, ordered, data.runs('ordered'), target, fixed)
            axes[1, column].set_title(f'Ordered, $T={target:g}$', pad=3)
            axes[1, column].set_xlabel('Cycles $k$')
        fig.text(.012, (.155 + .945) / 2, r'$\bar\lambda$ / baseline', rotation=90, ha='left', va='center')
        fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, fontsize=7, handlelength=1.6,
                   columnspacing=1.4, borderaxespad=.2)
        fig.savefig(out / 'cycles-combined.pdf')
        plt.close(fig)
