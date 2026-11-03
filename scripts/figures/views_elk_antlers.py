"""Fig. views-elk-antlers-1x3 (Ch. 4): the Elk's antlers at T = 1. After 25 cycles, the average edge loss is as low as
after five, but most of the thin plate is gone.

Writes views-elk-antlers-1x3.pdf.
"""
from shared import render
from shared.mesh_images import STYLE, Panel, capture, deviation_text, grid, loss_text

CAMERA = render.camera([0.272816181182861, 0.307888865470886, 0.149749398231506],
                       [-0.997762441635132, 0.0, -0.0668581500649452, 0.282217770814896, -0.030678553506732,
                        0.888508439064026, 0.457833617925644, -0.333752542734146, 0.0594040341675282,
                        0.458860337734222, -0.886520445346832, -2.42347693443298, 0.0, 0.0, 0.0, 1.0])


def generate(data, runs, out):
    target = 1.0
    panels = [Panel(data, runs, 'elk', target), Panel(data, runs, 'elk', target, 'baseline'),
              Panel(data, runs, 'elk', target, 'baseline', cycles=25)]
    capture(panels, CAMERA, STYLE, crop=(0, .21, 1, .67))
    titles = ['Input', '\n'.join(['Baseline, $k=5$', deviation_text(panels[1])]),
              '\n'.join(['Baseline, $k=25$', deviation_text(panels[2])])]
    grid(out / 'views-elk-antlers-1x3.pdf', panels, 3, 'coolwarm', titles=titles, below=[loss_text(p) for p in panels],
         target=target)
