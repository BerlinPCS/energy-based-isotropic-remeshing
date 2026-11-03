"""Fig. iteration-comparison (Ch. 1): the Bunny after 1, 5 and 25 baseline cycles at T = 1 and 1.5. The same cycle count
leaves the two targets at different stages of convergence.

Writes views-bunny-iterations.pdf.
"""
from shared import render
from shared.mesh_images import BAND, POINT, Panel, capture, grid, loss_text

CAMERA = render.camera([-.62, .371205713, 0], [1, 0, 0, .62, 0, 1, 0, -.371205713, 0, 0, 1, -1.2, 0, 0, 0, 1])


def generate(data, runs, out):
    targets, cycles = [1.0, 1.5], [1, 5, 25]
    panels = [Panel(data, runs, 'stanford-bunny', t, 'baseline', cycles=k) for t in targets for k in cycles]
    capture(panels, CAMERA, [*BAND, '--size', '900x700'])
    grid(out / 'views-bunny-iterations.pdf', panels, 3, 'magma_r',
         below=[rf'{loss_text(p)}, ${p.outside:.1f}\%$ outside band' for p in panels],
         column_titles=[f'$k = {k}$' for k in cycles], row_titles=[f'$T = {t:g}$' for t in targets], band=True,
         row_title_width=26 * POINT)
