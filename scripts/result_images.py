"""The mesh images of the thesis: captures with mesh-screenshot, composed into figures. Used by generate_results.py.

All panels of a figure share the camera and the image-space crop, the union of their non-background pixels, so that they
stay aligned and little white background is left.
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from PIL import Image

import scan_meshes
from common import render
from common.style import TEXT_WIDTH_POINTS

ROOT = Path(__file__).resolve().parents[1]
POINT = 1 / 72.27
WIDTH = TEXT_WIDTH_POINTS * POINT
DPI = 600
IMAGE_STYLE = {'font.family': 'serif', 'font.serif': ['STIX Two Text', 'STIXGeneral'], 'font.size': 8,
               'mathtext.fontset': 'stix', 'pdf.fonttype': 42}
LONG_EDGE, SHORT_EDGE = (.10, .75, .20), (0, .80, .95)  # the out-of-band edge colours of apps/shared/src/visuals.cpp

CAMERAS = {
    'elk': render.camera([0.272816181182861, 0.307888865470886, 0.149749398231506],
                         [-0.997762441635132, 0.0, -0.0668581500649452, 0.282217770814896, -0.030678553506732,
                          0.888508439064026, 0.457833617925644, -0.333752542734146, 0.0594040341675282,
                          0.458860337734222, -0.886520445346832, -2.42347693443298, 0.0, 0.0, 0.0, 1.0]),
    'bunny-detail': render.camera([-0.0314967632293701, -0.188708603382111, -0.045610785484314],
                                  [0.619214057922363, 0.0, -0.785222172737122, -0.0163111686706543, 0.168047860264778,
                                   0.976830720901489, 0.132519945502281, 0.195673689246178, 0.767029225826263,
                                   -0.214013129472733, 0.604867398738861, -2.29503417015076, 0.0, 0.0, 0.0, 1.0]),
    'bunny-head': render.camera([-.62, .37, 0], [1, 0, 0, .62, 0, 1, 0, -.37, 0, 0, 1, -1.2, 0, 0, 0, 1]),
    'rocker-arm': render.camera([-0.026951789855957, 0.0627247095108032, 0.176916003227234],
                                [0.356904417276382, 0.0, 0.934140861034393, -0.155645221471786, -0.474825650453568,
                                 0.861178874969482, 0.181415215134621, -0.0989099442958832, -0.804462432861328,
                                 -0.508301973342896, 0.30735856294632, -1.72133219242096, 0.0, 0.0, 0.0, 1.0]),
    'horse-cap': render.camera([0.0764657258987427, -0.216618105769157, 0.0790727734565735],
                               [0.143999367952347, 0.0, -0.989577829837799, 0.0672376453876495, -0.00477570295333862,
                                0.999988436698914, -0.000694941030815244, 0.217035740613937, 0.989566266536713,
                                0.00482600042596459, 0.143997684121132, -0.504755198955536, 0.0, 0.0, 0.0, 1.0]),
    'max-planck': render.camera([0.13323563337326, -0.0250874683260918, -0.197578847408295],
                                [0.697087049484253, 0.0, -0.71698659658432, -0.23453825712204, -0.0854687839746475,
                                 0.992869675159454, -0.0830966457724571, 0.0198779292404652, 0.711874186992645,
                                 0.119205549359322, 0.692116498947144, -0.99624502658844, 0.0, 0.0, 0.0, 1.0]),
}
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


def capture(panels, camera, options, out, crop=None):
    """Sets panel.image for all panels. crop: (x0, y0, x1, y1) as fractions of the screenshot, before the shared crop."""
    with tempfile.TemporaryDirectory() as temporary:
        for index, panel in enumerate(panels):
            path = Path(temporary) / f'{index}.png'
            render.screenshot(panel.source, path, ['--camera', CAMERAS[camera], *options], panel.target_length)
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
        fig.savefig(path)
        plt.close(fig)


def bunny_iterations(data, runs, out):
    """Fig. iteration-comparison: the same cycle count leaves two targets at different stages of convergence."""
    targets, cycles = [1.0, 1.5], [1, 5, 25]
    panels = [Panel(data, runs, 'stanford-bunny', t, 'baseline', cycles=k) for t in targets for k in cycles]
    capture(panels, 'bunny-head', [*BAND, '--size', '900x700'], out)
    grid(out / 'views-bunny-iterations.pdf', panels, 3, 'magma_r',
         below=[rf'{loss_text(p)}, ${p.outside:.1f}\%$ outside band' for p in panels],
         column_titles=[f'$k = {k}$' for k in cycles], row_titles=[f'$T = {t:g}$' for t in targets], band=True,
         row_title_width=26 * POINT)


def bunny_queue_objectives(data, runs, out):
    targets = [0.75, 1.0, 1.5]
    configurations = [('baseline', ''), *(('dynamic-queue', o) for o in ['sum', 'maximum', 'local-average', 'global-average'])]
    panels = [Panel(data, runs, 'stanford-bunny', t, a, o) for a, o in configurations for t in targets]
    capture(panels, 'bunny-detail', BAND, out)
    grid(out / 'views-bunny-queue-objectives-4x3.pdf', panels, 3, 'magma_r', below=[loss_text(p) for p in panels],
         column_titles=[f'$T = {t:g}$' for t in targets], row_titles=['(a)', '(b)', '(c)', '(d)', '(e)'], band=True,
         max_height=540 * POINT)


def rocker_arm_objectives(data, runs, out):
    names = {'sum': 'Dynamic / sum', 'global-average': 'Dynamic / global avg.'}
    configurations = [(o, t) for o in names for t in [0.75, 1.0]]
    panels = [Panel(data, runs, 'rocker-arm', t, 'dynamic-queue', o) for o, t in configurations]
    capture(panels, 'rocker-arm', STYLE, out)
    titles = [f'{names[o]}\n$T = {t:g}$, {loss_text(p)}' for (o, t), p in zip(configurations, panels)]
    grid(out / 'views-rocker-arm-objectives.pdf', panels, 2, 'coolwarm', titles=titles)


def max_planck_hybrid(data, runs, out):
    """Left: the input and the dynamic queue started from it. Right: two baseline cycles and the hybrid's queue after them."""
    target = 0.75
    panels = [Panel(data, runs, 'max-planck', target), Panel(data, runs, 'max-planck', target, 'baseline', cycles=2),
              Panel(data, runs, 'max-planck', target, 'dynamic-queue', 'global-average'),
              Panel(data, runs, 'max-planck', target, 'hybrid', 'global-average')]
    capture(panels, 'max-planck', BRIGHT, out, crop=(0, .10, 1, .84))
    names = ['Input', 'Baseline, 2 cycles', 'Dynamic / global avg.', 'Hybrid / global avg.']
    grid(out / 'views-max-planck-hybrid.pdf', panels, 2, 'coolwarm',
         titles=[f'{name}, {loss_text(p)}' for name, p in zip(names, panels)], target=target, last_titles_below=True,
         arrows=['Queue', 'Queue'])


def elk_antlers(data, runs, out):
    target = 1.0
    panels = [Panel(data, runs, 'elk', target), Panel(data, runs, 'elk', target, 'baseline'),
              Panel(data, runs, 'elk', target, 'baseline', cycles=25)]
    capture(panels, 'elk', STYLE, out, crop=(0, .21, 1, .67))
    titles = ['Input', '\n'.join(['Baseline, $k=5$', deviation_text(panels[1])]),
              '\n'.join(['Baseline, $k=25$', deviation_text(panels[2])])]
    grid(out / 'views-elk-antlers-1x3.pdf', panels, 3, 'coolwarm', titles=titles, below=[loss_text(p) for p in panels],
         target=target)


def write_obj(path, vertices, faces):
    keep = sorted({vertex for face in faces for vertex in face})
    index = {vertex: position + 1 for position, vertex in enumerate(keep)}
    lines = [f'v {vertices[v][0]!r} {vertices[v][1]!r} {vertices[v][2]!r}\n' for v in keep]
    lines += [f'f {index[a]} {index[b]} {index[c]}\n' for a, b, c in faces]
    path.write_text(''.join(lines))


def horse_cap(data, runs, out):
    """A long edge as the base of two caps: no single edit improves it, flipping it and collapsing the new diagonal does.
    The last two panels apply these edits here by hand, no queue made them."""
    numbers = data.horse_cap
    corners = numbers['vertices']
    a, b, c, d = corners['A'], corners['B'], corners['C'], corners['D']
    target_length = numbers['target_length']
    run = runs / 'core' / scan_meshes.HORSE_CAP_RUN
    vertices, all_faces = scan_meshes.load_obj(run / 'remeshed.obj')
    stored = scan_meshes.StoredMesh(run / 'remeshed.obj', dict(target_length=target_length, epsilon=0.0))
    _, _, face_1, face_2, new_faces = stored.flip_quad(a, c)
    faces = [[int(x) for x in face] for index, face in enumerate(all_faces) if index not in (face_1, face_2)]
    moved = vertices.tolist()
    moved[b] = ((vertices[b] + vertices[d]) / 2).tolist()

    crop = (.30, .28, .57, .72)
    panels = []
    with tempfile.TemporaryDirectory() as temporary:
        flip, collapse = Path(temporary) / 'flip.obj', Path(temporary) / 'collapse.obj'
        write_obj(flip, vertices.tolist(), faces + new_faces)
        write_obj(collapse, moved, [[b if x == d else x for x in face] for face in faces])
        for index, source in enumerate([run / 'remeshed.obj', flip, collapse]):
            path = Path(temporary) / f'{index}.png'
            render.screenshot(source, path, ['--camera', CAMERAS['horse-cap'], *STYLE], target_length)
            image = (plt.imread(path)[..., :3] * 255).round().astype(np.uint8)
            height, width = image.shape[:2]
            panels.append(SimpleNamespace(image=image[round(crop[1] * height):round(crop[3] * height),
                                                      round(crop[0] * width):round(crop[2] * width)]))

    view = np.array(json.loads(CAMERAS['horse-cap'])['viewMat']).reshape(4, 4)
    size = np.tan(np.pi / 8)

    def in_panel(point):
        """The fraction of the cropped panel where a point of the mesh appears, with y upwards."""
        camera_space = view[:3, :3] @ point + view[:3, 3]
        x = camera_space[0] / -camera_space[2] / (size * render.WINDOW_WIDTH / render.WINDOW_HEIGHT) * .5 + .5
        y = camera_space[1] / -camera_space[2] / size * -.5 + .5
        return (x - crop[0]) / (crop[2] - crop[0]), 1 - (y - crop[1]) / (crop[3] - crop[1])

    def overlay(ax, index):
        edge = {0: (a, c, LONG_EDGE), 1: (b, d, SHORT_EDGE)}.get(index)
        if edge:
            ax.plot(*zip(in_panel(vertices[edge[0]]), in_panel(vertices[edge[1]])), color=edge[2], lw=1.1,
                    solid_capstyle='round', transform=ax.transAxes, zorder=3)

    def gain(operation, value):
        return rf'$g_{{\mathrm{{sum}}}}({operation})={value:+.2f}$'

    titles = [gain('s', numbers['split_gain_sum']), gain('f', numbers['flip_gain_sum']),
              rf"$g_{{\mathrm{{sum}}}}(f)+g_{{\mathrm{{sum}}}}(c)={numbers['net_gain_sum']:+.2f}$"]
    grid(out / 'views-horse-cap.pdf', panels, 3, 'coolwarm', titles=titles, target=1, overlay=overlay)


FIGURES = {'views-bunny-iterations': bunny_iterations, 'views-bunny-queue-objectives-4x3': bunny_queue_objectives,
           'views-rocker-arm-objectives': rocker_arm_objectives, 'views-horse-cap': horse_cap,
           'views-max-planck-hybrid': max_planck_hybrid, 'views-elk-antlers-1x3': elk_antlers}
