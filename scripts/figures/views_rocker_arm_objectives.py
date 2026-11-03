"""Fig. views-rocker-arm-objectives (Ch. 4): coarse near-equilateral patches on the Rocker arm that the dynamic queue
with the sum and the global average leaves unsplit at T = 0.75 and 1.

Writes views-rocker-arm-objectives.pdf.
"""
from shared import render
from shared.mesh_images import STYLE, Panel, capture, grid, loss_text

CAMERA = render.camera([-0.026951789855957, 0.0627247095108032, 0.176916003227234],
                       [0.356904417276382, 0.0, 0.934140861034393, -0.155645221471786, -0.474825650453568,
                        0.861178874969482, 0.181415215134621, -0.0989099442958832, -0.804462432861328,
                        -0.508301973342896, 0.30735856294632, -1.72133219242096, 0.0, 0.0, 0.0, 1.0])


def generate(data, runs, out):
    names = {'sum': 'Dynamic / sum', 'global-average': 'Dynamic / global avg.'}
    configurations = [(o, t) for o in names for t in [0.75, 1.0]]
    panels = [Panel(data, runs, 'rocker-arm', t, 'dynamic-queue', o) for o, t in configurations]
    capture(panels, CAMERA, STYLE)
    titles = [f'{names[o]}\n$T = {t:g}$, {loss_text(p)}' for (o, t), p in zip(configurations, panels)]
    grid(out / 'views-rocker-arm-objectives.pdf', panels, 2, 'coolwarm', titles=titles)
