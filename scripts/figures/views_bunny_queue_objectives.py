"""Fig. views-bunny-queue-objectives-4x3 (Ch. 4): the baseline and the dynamic queue with each objective on the Bunny
at all three targets, with the edges outside the length band.

Writes views-bunny-queue-objectives-4x3.pdf.
"""
from shared import render
from shared.dataset import OBJECTIVES, TARGETS
from shared.mesh_images import BAND, POINT, Panel, capture, grid, loss_text

CAMERA = render.camera([-0.0314967632293701, -0.187502890382111, -0.045610785484314],
                       [0.619214057922363, 0.0, -0.785222172737122, -0.0163111686706543, 0.168047860264778,
                        0.976830720901489, 0.132519945502281, 0.194495911747188, 0.767029225826263,
                        -0.214013129472733, 0.604867398738861, -2.29477613173838, 0.0, 0.0, 0.0, 1.0])


def generate(data, runs, out):
    configurations = [('baseline', ''), *(('dynamic-queue', o) for o in OBJECTIVES)]
    panels = [Panel(data, runs, 'stanford-bunny', t, a, o) for a, o in configurations for t in TARGETS]
    capture(panels, CAMERA, BAND)
    grid(out / 'views-bunny-queue-objectives-4x3.pdf', panels, 3, 'magma_r', below=[loss_text(p) for p in panels],
         column_titles=[f'$T = {t:g}$' for t in TARGETS], row_titles=['(a)', '(b)', '(c)', '(d)', '(e)'], band=True,
         max_height=540 * POINT)
