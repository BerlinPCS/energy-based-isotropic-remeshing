"""Fig. queue-heatmap (Ch. 4): the median average edge loss of the dynamic and batched queues relative to five baseline
cycles, for each objective and target.

Writes queues-heatmap.pdf.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from shared.dataset import LOSS, OBJECTIVES, TARGETS
from shared.style import PLOT_STYLE, TEXT_WIDTH_INCHES


def generate(data, out):
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
        # The cells and the colour bar's gradient are rasterized at the save dpi; at the default 100 dpi they blur when
        # zoomed and sit up to half a pixel off the vector separator line and outline.
        fig.savefig(out / 'queues-heatmap.pdf', dpi=600)
        plt.close(fig)
