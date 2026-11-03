"""Fig. views-horse-cap (Ch. 4): a long edge as the base of two caps on the Horse (batched queue, local average, T = 1).
No single edit improves it, but flipping it and collapsing the new diagonal does. The last two panels apply these edits
here by hand, no queue made them. The gains were measured once on this output and are written below.

Writes views-horse-cap.pdf.
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import scan_meshes
from shared import render
from shared.mesh_images import LONG_EDGE, SHORT_EDGE, STYLE, grid

# The output, and its long edge A-C with the apexes B and D of its two triangles.
RUN = 'horse__batched__local-average__T1'
A, B, C, D = 7898, 7729, 21556, 21740
# Gains g_sum of the edits of A-C, measured once. Neither the split nor the flip passes any of the four objectives, but
# the collapse of B-D after the flip passes all four.
SPLIT_GAIN, FLIP_GAIN, FLIP_AND_COLLAPSE_GAIN = -12.43, -1.48, 0.66

CAMERA = render.camera([0.0764657258987427, -0.216618105769157, 0.0790727734565735],
                       [0.143999367952347, 0.0, -0.989577829837799, 0.0672376453876495, -0.00477570295333862,
                        0.999988436698914, -0.000694941030815244, 0.217035740613937, 0.989566266536713,
                        0.00482600042596459, 0.143997684121132, -0.504755198955536, 0.0, 0.0, 0.0, 1.0])


def write_obj(path, vertices, faces):
    keep = sorted({vertex for face in faces for vertex in face})
    index = {vertex: position + 1 for position, vertex in enumerate(keep)}
    lines = [f'v {vertices[v][0]!r} {vertices[v][1]!r} {vertices[v][2]!r}\n' for v in keep]
    lines += [f'f {index[a]} {index[b]} {index[c]}\n' for a, b, c in faces]
    path.write_text(''.join(lines))


def generate(data, runs, out):
    a, b, c, d = A, B, C, D
    run = runs / 'core' / RUN
    target_length = json.loads((run / 'summary.json').read_text())['target_length']
    vertices, all_faces = scan_meshes.load_obj(run / 'remeshed.obj')
    # The flip replaces the two triangles of A-C by two triangles of B-D.
    faces = [[int(x) for x in face] for face in all_faces if not (a in face and c in face)]
    new_faces = [[b, a, d], [d, c, b]]
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
            render.screenshot(source, path, ['--camera', CAMERA, *STYLE], target_length)
            image = (plt.imread(path)[..., :3] * 255).round().astype(np.uint8)
            height, width = image.shape[:2]
            panels.append(SimpleNamespace(image=image[round(crop[1] * height):round(crop[3] * height),
                                                      round(crop[0] * width):round(crop[2] * width)]))

    view = np.array(json.loads(CAMERA)['viewMat']).reshape(4, 4)
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

    titles = [gain('s', SPLIT_GAIN), gain('f', FLIP_GAIN),
              rf'$g_{{\mathrm{{sum}}}}(f)+g_{{\mathrm{{sum}}}}(c)={FLIP_AND_COLLAPSE_GAIN:+.2f}$']
    grid(out / 'views-horse-cap.pdf', panels, 3, 'coolwarm', titles=titles, target=1, overlay=overlay)
