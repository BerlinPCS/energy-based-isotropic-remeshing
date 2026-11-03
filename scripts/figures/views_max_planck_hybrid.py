"""Fig. views-max-planck-hybrid (Ch. 4): Max Planck at T = 0.75. Left: the input and the dynamic queue started from it.
Right: two baseline cycles and the hybrid's queue after them.

Writes views-max-planck-hybrid.pdf.
"""
from shared import render
from shared.mesh_images import BRIGHT, Panel, capture, grid, loss_text

CAMERA = render.camera([0.127840348199161, 0.0164227426606098, -0.189578028114322],
                       [0.697087049484253, 0.0, -0.71698659658432, -0.225040791996742, -0.0854687839746475,
                        0.992869675159454, -0.0830966457724571, -0.021132587456986, 0.711874186992645,
                        0.119205549359322, 0.692116498947144, -0.960729917267107, 0.0, 0.0, 0.0, 1.0])


def generate(data, runs, out):
    target = 0.75
    panels = [Panel(data, runs, 'max-planck', target), Panel(data, runs, 'max-planck', target, 'baseline', cycles=2),
              Panel(data, runs, 'max-planck', target, 'dynamic-queue', 'global-average'),
              Panel(data, runs, 'max-planck', target, 'hybrid', 'global-average')]
    capture(panels, CAMERA, BRIGHT, crop=(0, .10, 1, .84))
    names = ['Input', 'Baseline, 2 cycles', 'Dynamic / global avg.', 'Hybrid / global avg.']
    grid(out / 'views-max-planck-hybrid.pdf', panels, 2, 'coolwarm',
         titles=[f'{name}, {loss_text(p)}' for name, p in zip(names, panels)], target=target, last_titles_below=True,
         arrows=['Queue', 'Queue'])
