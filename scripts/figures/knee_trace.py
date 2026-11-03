"""Fig. knee-trace (Ch. 3): the knee rule on one run of the dynamic queue (Bunny, T = 1, local average). (a) The average
edge loss over the executed edits, (b) its relative decrease per window of K edits.

Writes knee-trace.pdf.
"""
import matplotlib.pyplot as plt

from shared.knee import KNEE_RATE_FRACTION, knee_figure_trace, knee_windows

KNEE_STYLE = {
    'font.family': 'serif', 'font.serif': ['Latin Modern Roman', 'CMU Serif', 'Computer Modern Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm', 'font.size': 7, 'axes.titlesize': 7.5, 'axes.labelsize': 7, 'xtick.labelsize': 6,
    'ytick.labelsize': 6, 'legend.fontsize': 6, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'axes.grid': True, 'grid.alpha': 0.25,
    'grid.linewidth': 0.5, 'legend.frameon': False, 'pdf.fonttype': 42,
}


def generate(data, out):
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
