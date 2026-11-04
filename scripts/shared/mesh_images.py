"""What the mesh images of Ch. 1 and 4 share: captures with mesh-screenshot, composed into a grid of panels.

All panels of a figure share the camera and the image-space crop, the union of their non-background pixels, so that they
stay aligned and little white background is left.
"""
import io
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pikepdf
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from PIL import Image

import scan_meshes
from shared import render
from shared.style import TEXT_WIDTH_POINTS

ROOT = Path(__file__).resolve().parents[2]
POINT = 1 / 72.27
WIDTH = TEXT_WIDTH_POINTS * POINT
DPI = 600
IMAGE_STYLE = {'font.family': 'serif', 'font.serif': ['STIX Two Text', 'STIXGeneral'], 'font.size': 8,
               'mathtext.fontset': 'stix', 'pdf.fonttype': 42}
LONG_EDGE, SHORT_EDGE = (.10, .75, .20), (0, .80, .95)  # the out-of-band edge colours of apps/shared/src/visuals.cpp

# Vertex mean of the loss over [0, 0.12], wireframe width 0.8 (1 in the viewers reads too dark).
STYLE = ['--coloring', 'vertex', '--colormap', 'coolwarm', '--range', '0,0.12', '--edge-width', '0.8']
BAND = ['--coloring', 'vertex', '--colormap', 'magma_r', '--range', '0,0.12', '--edge-width', '0.8', '--out-of-band']
# Polyscope's light is a camera-fixed matcap; a higher exposure (default 1.1) brightens surfaces seen from the side, such
# as Max Planck's cheek and ear.
BRIGHT = [*STYLE, '--exposure', '1.5']


class Panel:
    """One capture: an input mesh, a run of the main comparison or a baseline of `cycles` cycles."""

    def __init__(self, data, runs, mesh, target, approach=None, objective='', cycles=None):
        self.mesh, self.target = mesh, target
        self.source, self.target_length = None, None
        target_name = f'T{target:g}'
        if approach is None:
            # An input is coloured against the target length of the baseline run.
            self.source = ROOT / 'meshes' / f'{mesh}.obj'
            self.target_length = data.reference.loc[(mesh, target)].target_length
            vertices, faces = scan_meshes.load_obj(self.source)
            edges = scan_meshes.unique_edges(faces)
            lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
            self.loss, self.deviation = float(scan_meshes.loss(lengths / self.target_length).mean()), None
            self.outside = None
            return
        if cycles is None:
            parts = [mesh, approach, *([objective] if objective else []), target_name]
            self.source = runs / 'core' / '__'.join(parts)
            row = data.runs(approach, objective).loc[(mesh, target)]
            self.outside = None
        else:
            self.source = runs / 'cycles' / f'{mesh}__baseline__{target_name}__k{cycles:02d}'
            row = data.fixed(cycles).loc[(mesh, target)]
            scan = data.scans[(data.scans.tier == 'cycles') & (data.scans.input == mesh) &
                              (data.scans.target == target) & (data.scans.cycles == cycles)]
            self.outside = 100 * float(scan.long_fraction.iloc[0] + scan.short_fraction.iloc[0]) if len(scan) else None
        self.loss, self.deviation = float(row.average_edge_loss), float(row.d_rms)


def capture(panels, camera, options, crop=None):
    """Sets panel.image for all panels. camera: the JSON of render.camera(). crop: (x0, y0, x1, y1) as fractions of the
    screenshot, before the shared crop."""
    with tempfile.TemporaryDirectory() as temporary:
        for index, panel in enumerate(panels):
            path = Path(temporary) / f'{index}.png'
            render.screenshot(panel.source, path, ['--camera', camera, *options], panel.target_length)
            panel.image = (plt.imread(path)[..., :3] * 255).round().astype(np.uint8)
    if crop:
        height, width = panels[0].image.shape[:2]
        for panel in panels:
            panel.image = panel.image[round(crop[1] * height):round(crop[3] * height),
                                      round(crop[0] * width):round(crop[2] * width)]
    # Pixels that differ from the white background in any panel, plus a small margin.
    mask = np.any([np.any(panel.image < 251, axis=2) for panel in panels], axis=0)
    ys, xs = np.nonzero(mask)
    pad = int(.01 * mask.shape[1])
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, mask.shape[1])
    for panel in panels:
        panel.image = panel.image[y0:y1, x0:x1]


def fit(image, width):
    """Downscale a screenshot (Lanczos) to DPI at its printed width; smaller ones stay as they are."""
    height, pixels = image.shape[:2]
    target = round(width * DPI)
    if pixels <= target:
        return image
    return np.asarray(Image.fromarray(image).resize((target, round(target * height / pixels)), Image.LANCZOS))


def loss_text(panel):
    return rf'$\bar\lambda={panel.loss:.3f}$'


def deviation_text(panel):
    return rf'$d_{{\mathrm{{RMS}}}}={1e3 * panel.deviation:.3g}\cdot 10^{{-3}}$'


def jpeg_panels(path, quality=95):
    """Re-encode the panels of a composed PDF as JPEG; matplotlib embeds images only losslessly, which makes the 600 ppi
    panels two to two and a half times larger. Full chroma resolution (4:4:4) keeps the colour of the thin out-of-band
    edges. Images under 200 px on a side, such as the colour bar's gradient, stay lossless, and so does a panel whose
    lossless stream is already smaller."""
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        for page in pdf.pages:
            for stream in page.get_images().values():
                image = pikepdf.PdfImage(stream)
                if min(image.width, image.height) < 200:
                    continue
                data = io.BytesIO()
                image.as_pil_image().convert('RGB').save(data, 'JPEG', quality=quality, subsampling=0)
                if len(data.getvalue()) < len(stream.read_raw_bytes()):
                    stream.write(data.getvalue(), filter=pikepdf.Name.DCTDecode)
        pdf.save(path)


def grid(path, cells, columns, colormap, titles=None, below=None, column_titles=None, row_titles=None, band=False,
         target=None, last_titles_below=False, arrows=None, overlay=None, max_height=None, row_title_width=45 * POINT):
    """cells: panels in row-major order. titles: one per cell, above the panel. below: one text per cell, under it.
    target: the common T of all panels, shown next to the colour bar. last_titles_below: the titles of the last row go
    below its panels. arrows: one label per column, for an arrow above the last row. overlay(ax, i) draws on panel i, in
    axes fractions. With max_height (inches) the panels shrink until the figure fits, narrower than the text."""
    gap = 3 * POINT
    rows = -(-len(cells) // columns)
    pixels_high, pixels_wide = cells[0].image.shape[:2]
    left = row_title_width if row_titles else 0
    title = (max(text.count('\n') for text in titles) + 1) * 8.4 * POINT + 2 * POINT if titles else 0
    under = 7.5 * POINT if below else 0
    head = 11 * POINT if column_titles else 0
    bar = 17 * POINT
    panel_width = (WIDTH - left - (columns - 1) * gap) / columns
    if max_height:
        panel_width = min(panel_width, (max_height - head - bar - rows * (title + under) - (rows - 1) * gap) / rows *
                          pixels_wide / pixels_high)
    panel_height = panel_width * pixels_high / pixels_wide
    figure_width = left + columns * panel_width + (columns - 1) * gap
    figure_height = head + rows * (title + panel_height + under) + (rows - 1) * gap + bar + \
        (title if last_titles_below else 0)

    with plt.rc_context(IMAGE_STYLE):
        fig = plt.figure(figsize=(figure_width, figure_height))
        for index, panel in enumerate(cells):
            row, column = divmod(index, columns)
            x = left + column * (panel_width + gap)
            y = figure_height - head - row * (title + panel_height + under + gap) - title - panel_height
            last_row = row == rows - 1
            ax = fig.add_axes([x / figure_width, y / figure_height, panel_width / figure_width,
                               panel_height / figure_height])
            # 'none' makes the PDF embed the panel exactly as fit() returns it.
            ax.imshow(fit(panel.image, panel_width), interpolation='none')
            ax.axis('off')
            if overlay:
                overlay(ax, index)
            if titles and last_titles_below and last_row:
                ax.text(.5, -1 * POINT / panel_height, titles[index], ha='center', va='top', transform=ax.transAxes,
                        fontsize=7, linespacing=1.2)
            elif titles:
                ax.text(.5, 1 + 1 * POINT / panel_height, titles[index], ha='center', va='bottom',
                        transform=ax.transAxes, fontsize=7, linespacing=1.2)
            if arrows and last_row:
                center, start, end = x + panel_width / 2, y + panel_height + title + gap - 1.5 * POINT, \
                    y + panel_height + 1.5 * POINT
                fig.add_artist(FancyArrowPatch((center / figure_width, start / figure_height),
                                               (center / figure_width, end / figure_height),
                                               transform=fig.transFigure, arrowstyle='-|>', mutation_scale=7, lw=.8,
                                               color='k', shrinkA=0, shrinkB=0))
                fig.text((center + 4 * POINT) / figure_width, (start + end) / 2 / figure_height, arrows[column],
                         ha='left', va='center', fontsize=7)
            if below:
                ax.text(.5, -.5 * POINT / panel_height, below[index], ha='center', va='top', transform=ax.transAxes,
                        fontsize=5.8)
            if row == 0 and column_titles:
                ax.text(.5, 1 + (title + 2 * POINT) / panel_height, column_titles[column], ha='center', va='bottom',
                        transform=ax.transAxes, fontsize=8)
            if column == 0 and row_titles:
                fig.text((left - 4 * POINT) / figure_width, (y + panel_height / 2) / figure_height, row_titles[row],
                         ha='right', va='center', fontsize=7.5, linespacing=1.15)

        # The colour bar along the bottom: label on its left, the common target or the edge legend on its right.
        bar_width = min(150 * POINT, .45 * figure_width)
        cax = fig.add_axes([(figure_width - bar_width) / 2 / figure_width, 10 * POINT / figure_height,
                            bar_width / figure_width, 4.5 * POINT / figure_height])
        colorbar = fig.colorbar(ScalarMappable(Normalize(0, .12), colormap), cax=cax, orientation='horizontal',
                                extend='max', extendfrac=.03)
        colorbar.set_ticks([0, .03, .06, .09, .12])
        colorbar.ax.tick_params(labelsize=6, length=2, width=.5, pad=1)
        colorbar.outline.set_linewidth(.5)
        cax.text(-.03, .5, 'Vertex loss', ha='right', va='center', transform=cax.transAxes, fontsize=6.5)
        if target:
            cax.text(1.06, .5, f'$T = {target:g}$', ha='left', va='center', transform=cax.transAxes, fontsize=7)
        if band:
            cax.legend(handles=[Line2D([], [], color=LONG_EDGE, lw=1.6, label=r'$>\frac{4}{3}\ell_*$'),
                                Line2D([], [], color=SHORT_EDGE, lw=1.6, label=r'$<\frac{4}{5}\ell_*$')],
                       loc='center left', bbox_to_anchor=(1.06, .5), frameon=False, fontsize=6.5, handlelength=1.3,
                       ncol=2, columnspacing=1, borderaxespad=0)
        fig.savefig(path, dpi=DPI)
        plt.close(fig)
    jpeg_panels(path)
