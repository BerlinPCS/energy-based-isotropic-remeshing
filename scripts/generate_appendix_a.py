#!/usr/bin/env python3
"""Appendix A: the table of the benchmark meshes and their gallery, each in full and with a magnified detail.

The table comes from inputs.csv of the dataset. The images are rendered with mesh-screenshot, with the wireframe on, at
4800 px, which needs a display. The full view is a downscaled copy (dense regions read darker), the detail a
native-resolution crop of about DETAIL_EDGES average edge lengths, on the surface near the mesh's centre or at
Mesh.detail, and marked by a box in the full view. Mesh.detail_side instead fixes the window, centred on Mesh.detail,
which may then include background.

usage: generate_appendix_a.py [--data DATA] [--out OUT] [--table-only]
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from common import render
from common.style import decimal_column

ROOT = Path(__file__).resolve().parents[1]
RENDER_SIZE = 2400  # points; rendered with 2x supersampling
OVERVIEW_PX = 1400  # longest side of the full view
DETAIL_EDGES = 15  # side of the detail window in average edge lengths
BOX = (0xD7, 0x7D, 0x43)  # thesis orange (opBefore)
STANFORD = "Stanford~\\cite{stanford3dscanrep}"
COMMON_3D = "\\cite{jacobson2020common3d}"


@dataclass(frozen=True)
class Mesh:
    name: str  # file in meshes/
    label: str  # name in the thesis
    source: str  # where the original comes from
    note: str = ""  # preparation beyond cleaning and scaling
    view: str = ""  # mesh-screenshot --look-at (eye, target[, up]); empty = default camera
    detail: tuple = ()  # preferred detail centre as fractions (x, y) of the full view; empty = centre of mass
    camera: str = ""  # mesh-screenshot --camera; overrides view
    detail_side: float = 0.0  # fixed detail window side as a fraction of the longer side of the full view


ROCKER_CAMERA = render.camera([-0.026951789855957, 0.0627247095108032, 0.176916003227234],
                             [0.356904417276382, 0.0, 0.934140861034393, -0.155645221471786,
                              -0.474825650453568, 0.861178874969482, 0.181415215134621, -0.0989099442958832,
                              -0.804462432861328, -0.508301973342896, 0.30735856294632, -1.72133219242096,
                              0.0, 0.0, 0.0, 1.0])
HOMER_CAMERA = render.camera([1.19209289550781e-07, 6.55651092529297e-07, 7.15255737304688e-07],
                            [0.890308082103729, 0.0, 0.455358743667603, -4.76837158203125e-07,
                             0.127006903290749, 0.960315465927124, -0.248321309685707, -4.76837158203125e-07,
                             -0.437288045883179, 0.278916150331497, 0.85497659444809, -2.87254476547241,
                             0.0, 0.0, 0.0, 1.0])
ELK_CAMERA = render.camera([0.17758572101593, -0.0320601463317871, -0.0586133003234863],
                          [0.880911767482758, 0.0, -0.473280549049377, -0.184177875518799,
                           -0.353293687105179, 0.665409624576569, -0.657581567764282, 0.0455300807952881,
                           0.3149254322052, 0.746478378772736, 0.58616715669632, -3.30361127853394,
                           0.0, 0.0, 0.0, 1.0])
MESHES = [
    Mesh("stanford-bunny-low", "Bunny (low)", STANFORD + ", decimated", "5 holes filled"),
    # The zoom sits on the uniform grid of the back.
    Mesh("stanford-bunny", "Bunny", STANFORD, "5 holes filled", detail=(0.47, 0.46)),
    Mesh("armadillo-r128", "Armadillo", STANFORD, "marching cubes", "0.8,0.3,-3.5,0,0,0"),
    Mesh("stanford-dragon-r128", "Dragon", STANFORD, "marching cubes"),
    Mesh("fandisk", "Fandisk", "Hoppe et al.~\\cite{hoppe1994piecewise,pmp23}"),
    Mesh("max-planck", "Max Planck", "MPI~" + COMMON_3D, "1 hole filled", "1.2,0.5,-3.8,0,0,0,0,1,0"),
    Mesh("horse", "Horse", "Cyberware~" + COMMON_3D, "", "-4,0,0.5,0,0,0,0,0,1"),
    # The zoom sits on the very large triangles of the patch that the pure queues leave unsplit (Fig. rocker-arm-objectives).
    Mesh("rocker-arm", "Rocker arm", "INRIA~" + COMMON_3D, "", camera=ROCKER_CAMERA, detail=(0.393, 0.778),
         detail_side=0.273),
    Mesh("homer", "Homer", "unknown~" + COMMON_3D, camera=HOMER_CAMERA, detail=(0.079, 0.499), detail_side=0.165),
    # The zoom sits on the toothed rim of the antler plate, where both sides of the thin sheet are visible.
    Mesh("elk", "Elk", "CGAL~\\cite{cgaldata}", camera=ELK_CAMERA, detail=(0.737, 0.157), detail_side=0.175),
]


def on_white(image):
    white = Image.new("RGB", image.size, "white")
    white.paste(image, mask=image.split()[3])
    return white


def detail_window(alpha, side, prefer):
    """Top-left corner of the side x side window that lies on the surface and is closest to prefer (x, y)."""
    inside = (alpha > 200).astype(np.int64)
    integral = np.pad(inside.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    covered = (integral[side:, side:] - integral[:-side, side:] - integral[side:, :-side] + integral[:-side, :-side])
    ys, xs = np.nonzero(covered >= 0.995 * side * side)
    if not len(xs):
        return None
    best = np.argmin((xs + side / 2 - prefer[0]) ** 2 + (ys + side / 2 - prefer[1]) ** 2)
    return int(xs[best]), int(ys[best])


def render_mesh(mesh, average_edge_length, out):
    options = ["--target", "1", "--edge-width", "1", "--transparent", "--no-loss"]
    if mesh.camera:
        aspect = render.WINDOW_WIDTH / render.WINDOW_HEIGHT
        options += ["--camera", mesh.camera, "--size", f"{round(RENDER_SIZE * aspect)}x{RENDER_SIZE}"]
    else:
        options += ["--size", f"{RENDER_SIZE}x{RENDER_SIZE}"]
        if mesh.view:
            options += ["--look-at", mesh.view]
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "render.png"
        render.screenshot(ROOT / "meshes" / f"{mesh.name}.obj", path, options)
        image = Image.open(path)
        image.load()
    image = image.crop(image.getbbox())
    w, h = image.size
    alpha = np.asarray(image.split()[3])

    # The mesh's longest side (2) spans roughly the longer image side, which sets the edge length in pixels.
    side = int(DETAIL_EDGES * average_edge_length / 2 * max(w, h))
    side = max(200, min(side, int(0.4 * min(w, h))))
    if mesh.detail:
        prefer = (mesh.detail[0] * w, mesh.detail[1] * h)
    else:
        ys, xs = np.nonzero(alpha > 200)
        prefer = (xs.mean(), ys.mean())
    if mesh.detail_side:
        side = round(mesh.detail_side * max(w, h))
        x0, y0 = max(0, round(prefer[0] - side / 2)), max(0, round(prefer[1] - side / 2))
    else:
        while (corner := detail_window(alpha, side, prefer)) is None:
            side = int(0.85 * side)  # thin parts: shrink until a window fits on the surface
        x0, y0 = corner
    full = on_white(image)
    full.crop((x0, y0, x0 + side, y0 + side)).save(out / f"{mesh.name}-detail.jpg", quality=92)

    scale = OVERVIEW_PX / max(w, h)
    overview = full.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    box = [round(x0 * scale), round(y0 * scale), round((x0 + side) * scale), round((y0 + side) * scale)]
    ImageDraw.Draw(overview).rectangle(box, outline=BOX, width=max(3, OVERVIEW_PX // 250))
    overview.save(out / f"{mesh.name}.jpg", quality=92)


def write_table(inputs, out):
    rows = []
    for mesh in MESHES:
        row = inputs.loc[mesh.name]
        rows.append([mesh.label, mesh.source, f"{int(row.vertices)}", f"{int(row.faces)}",
                     f"{row.average_edge_length:.4f}", f"{row.edge_length_cv:.2f}", mesh.note])
    spec = "".join(decimal_column([row[index] for row in rows]) for index in range(2, 6))
    # As wide as the text: the space left over is spread over the gaps between the columns.
    lines = ["% Generated by scripts/generate_appendix_a.py; edit MESHES there, not this file.",
             "\\sisetup{group-digits=integer, group-minimum-digits=4}",
             f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}lc{spec}c@{{}}}}", "\\toprule",
             "Mesh & Source & {$|V|$} & {$|F|$} & {$\\ell_{\\mathrm{avg}}$} & {CV} & Preparation \\\\",
             "\\midrule", *(" & ".join(row) + " \\\\" for row in rows), "\\bottomrule", "\\end{tabular*}"]
    (out / "mesh-table.tex").write_text("\n".join(lines) + "\n")


def write_gallery(out):
    lines = ["% Generated by scripts/generate_appendix_a.py; edit MESHES there, not this file.",
             "\\setlength{\\tabcolsep}{0pt}",
             "\\begin{tabular}{@{}*{2}{m{.29\\textwidth}@{\\hspace{.02\\textwidth}}m{.16\\textwidth}}@{}}"]
    for start in range(0, len(MESHES), 2):
        row = MESHES[start:start + 2]
        cells = []
        for mesh in row:
            cells.append(f"\\centering\\includegraphics[width=.27\\textwidth,height=3.5cm,keepaspectratio]"
                         f"{{figures/meshes/{mesh.name}.jpg}} & "
                         f"\\includegraphics[width=.16\\textwidth]{{figures/meshes/{mesh.name}-detail.jpg}}")
        lines.append(" &\n".join(cells) + " \\\\")
        labels = [f"\\multicolumn{{2}}{{c}}{{\\small {mesh.label}}}" for mesh in row]
        lines.append(" & ".join(labels) + " \\\\[2ex]")
    lines.append("\\end{tabular}")
    (out / "mesh-gallery.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--table-only", action="store_true", help="only the table, which needs no display")
    args = parser.parse_args()
    inputs = pd.read_csv(args.data / "inputs.csv").set_index("mesh")
    args.out.mkdir(parents=True, exist_ok=True)
    write_table(inputs, args.out)
    if args.table_only:
        return
    write_gallery(args.out)
    images = args.out / "meshes"
    images.mkdir(exist_ok=True)
    for mesh in MESHES:
        render_mesh(mesh, inputs.loc[mesh.name].average_edge_length, images)
        print("rendered", mesh.name, flush=True)


if __name__ == "__main__":
    main()
