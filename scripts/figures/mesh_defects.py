"""Generate mesh-defects.tikz (Figure fig:mesh-manifoldness, Chapter 2).

Five small hand-built meshes, one per kind of defect, drawn in the style of small-operations.tikz:
(a) two closed triangle fans that meet only at their apex (non-manifold vertex),
(b) three triangles sharing one edge (non-manifold edge),
(c) a patch with a hole and a single triangle inside the hole (boundary edges, a second connected component),
(d) a flat strip that contains the same configuration twice: on the left the middle vertex lies on the opposite
    edge of its face (degenerate face), on the right it has moved beyond that edge (inverted face). The inverted
    face is drawn on top with an orange tint, and the part of the edge {t2, b2} that it covers is dotted,
(e) a single connected sheet whose left border continues as a flap that folds back over the sheet and passes
    through it (self-intersection); the script checks that it is one consistently oriented component in which no
    edge has more than two faces, so the self-intersection is its only defect.

Orthographic camera per panel (azimuth, elevation), two-sided flat shading from a light at the upper left,
painter's algorithm by face centroid depth. In (e) the flap faces are split at the plane of the sheet and drawn
in layers (below the sheet, the sheet, above the sheet), which is exact for a planar sheet seen from above; the
intersection is marked only where no face covers it. Vertex markers are drawn only where no opaque face covers
them. The script checks each defect before writing.
"""

import random
from collections import Counter, defaultdict
from math import cos, pi, radians, sin, sqrt

from shared.style import define_color

AMBIENT = 0.66
PITCH = 2.42  # cm between panel centres
BOX = (2.0, 1.6)  # cm, panel drawing area (width, height)
LABEL_Y = -0.95


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def norm(a):
    length = sqrt(dot(a, a))
    return tuple(x / length for x in a)


def normal(mesh, face):
    a, b, c = (mesh["v"][i] for i in face)
    return cross(sub(b, a), sub(c, a))


def edges(faces):
    return Counter(tuple(sorted((f[i], f[(i + 1) % 3]))) for f in faces for i in range(3))


def components(faces):
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            x = parent[x]
        return x

    for f in faces:
        for x in f[1:]:
            parent[find(x)] = find(f[0])
    return len({find(x) for f in faces for x in f})


def orient(faces):
    """Reorder the faces of a connected mesh so that adjacent faces traverse their shared edge in opposite
    directions; fails if the mesh is not orientable."""
    faces = list(faces)
    incident = defaultdict(list)
    for k, f in enumerate(faces):
        for i in range(3):
            incident[tuple(sorted((f[i], f[(i + 1) % 3])))].append(k)
    done, stack = {0}, [0]
    while stack:
        f = faces[stack.pop()]
        for i in range(3):
            a, b = f[i], f[(i + 1) % 3]
            for m in incident[tuple(sorted((a, b)))]:
                if m in done:
                    continue
                g = faces[m]
                if any((g[j], g[(j + 1) % 3]) == (a, b) for j in range(3)):
                    faces[m] = (g[0], g[2], g[1])
                done.add(m)
                stack.append(m)
    assert len(done) == len(faces)
    assert max(Counter((f[i], f[(i + 1) % 3]) for f in faces for i in range(3)).values()) == 1, "not orientable"
    return faces


def camera(az, el):
    """Screen right, screen up and the direction towards the viewer, in world coordinates."""
    a, e = radians(az), radians(el)
    right = (cos(a), sin(a), 0.0)
    forward = (-sin(a), cos(a), 0.0)  # horizontal viewing direction
    up = tuple(sin(e) * f + cos(e) * z for f, z in zip(forward, (0, 0, 1)))
    toward = tuple(-cos(e) * f + sin(e) * z for f, z in zip(forward, (0, 0, 1)))
    return right, up, toward


def clip(poly, side):
    """The part of a convex polygon with side * z >= 0 (Sutherland-Hodgman against the plane z = 0)."""
    out = []
    for i, p in enumerate(poly):
        q = poly[(i + 1) % len(poly)]
        dp, dq = side * p[2], side * q[2]
        if dp >= 0:
            out.append(p)
        if (dp >= 0) != (dq >= 0):
            t = dp / (dp - dq)
            out.append(tuple(a + t * (b - a) for a, b in zip(p, q)))
    return out


def lattice(rings, spacing):
    coords = {}
    for i in range(-rings, rings + 1):
        for j in range(-rings, rings + 1):
            if max(abs(i), abs(j), abs(i + j)) <= rings:
                coords[(i, j)] = (spacing * (i + 0.5 * j), spacing * j * sqrt(3) / 2)
    tris = []
    for (i, j) in coords:
        for tri in (((i, j), (i + 1, j), (i, j + 1)), ((i + 1, j), (i + 1, j + 1), (i, j + 1))):
            if all(t in coords for t in tri):
                tris.append(tri)
    return coords, tris


# ---------------------------------------------------------------------------------------------- panels

def hourglass():
    n, r, h = 7, 0.95, 0.75
    v = [(0.0, 0.0, 0.0)]
    v += [(r * cos(2 * pi * k / n + 0.2), r * sin(2 * pi * k / n + 0.2), h) for k in range(n)]
    v += [(0.85 * r * cos(2 * pi * k / n + 0.5), 0.85 * r * sin(2 * pi * k / n + 0.5), -h) for k in range(n)]
    f = [(0, 1 + k, 1 + (k + 1) % n) for k in range(n)]
    f += [(0, 1 + n + (k + 1) % n, 1 + n + k) for k in range(n)]
    mesh = dict(v=v, f=f, hv={0}, he=set(), az=8, el=20)
    counts = edges(f)
    assert sum(1 for face in f if 0 in face) == 2 * n and max(counts.values()) <= 2
    return mesh


def book():
    v = [(-0.75, 0.0, 0.0), (0.75, 0.0, 0.0)]
    for x, angle in ((0.15, 80), (-0.3, 200), (0.35, 315)):
        v.append((x, 0.95 * cos(radians(angle)), 0.95 * sin(radians(angle))))
    f = [(0, 1, 2), (1, 0, 3), (0, 1, 4)]
    assert edges(f)[(0, 1)] == 3
    return dict(v=v, f=f, hv={0, 1}, he={(0, 1)}, az=34, el=14)


def holed_patch():
    coords, tris = lattice(2, 1.0)
    keys = sorted(coords)
    index = {k: n for n, k in enumerate(keys)}
    v = []
    for k in keys:
        x, y = coords[k]
        jitter = 0.07 * sin(2.3 * k[0] + 1.3 * k[1] + 0.4)
        v.append((x + jitter, y - jitter, 0.35 - 0.09 * (x * x + y * y)))
    f = [tuple(index[t] for t in tri) for tri in tris if (0, 0) not in tri]
    counts = edges(f)
    hole = {e for e, c in counts.items() if c == 1 and all(max(abs(a) for a in (*keys[p], keys[p][0] + keys[p][1])) == 1 for p in e)}
    # A single triangle inside the hole, not connected to the rest of the mesh. Its plane faces the camera but is
    # turned away from the light, so that it stays large on screen and is shaded darker than the patch.
    az, el = 12, 52
    right, up, toward = camera(az, el)
    n = norm(tuple(t + 0.3 * (0.45 * r - 0.65 * u) for r, u, t in zip(right, up, toward)))
    base = len(v)
    for a in (80, 200, 320):
        d = (cos(radians(a)), sin(radians(a)), 0.0)
        d = norm(sub(d, tuple(dot(d, n) * x for x in n)))
        v.append(tuple(c + 0.45 * x for c, x in zip((0.0, 0.0, 0.33), d)))
    f.append((base, base + 1, base + 2))
    island = {(base, base + 1), (base + 1, base + 2), (base, base + 2)}
    assert components(f) == 2
    assert sum(1 for c in edges(f).values() if c == 1) == len([e for e, c in counts.items() if c == 1]) + 3
    he = hole | island
    return dict(v=v, f=f, hv={p for e in he for p in e}, he=he, az=az, el=el)


def degenerate_and_flipped():
    """The configuration (a, c, b) twice: b on the edge {a, c} (left), b beyond it (right)."""
    pos = {"a": (-1.45, 0.05), "c": (0.0, -0.03), "t1": (-0.75, 0.95), "s1": (-0.7, -0.95),
           "c2": (1.45, 0.05), "b2": (0.72, -0.5), "t2": (0.75, 0.95), "s2": (0.78, -0.97)}
    pos["b"] = tuple(a + 0.5 * (c - a) for a, c in zip(pos["a"], pos["c"]))  # exactly on the edge {a, c}
    names = list(pos)
    index = {k: n for n, k in enumerate(names)}
    tris = [("a", "b", "t1"), ("b", "c", "t1"), ("a", "c", "b"), ("c", "a", "s1"),  # left: degenerate
            ("c", "t2", "t1"), ("c", "s1", "s2"),  # connecting faces
            ("c", "b2", "t2"), ("b2", "c2", "t2"), ("c", "c2", "b2"), ("c2", "c", "s2")]  # right: flipped
    v = [(*pos[k], 0.0) for k in names]
    f = [tuple(index[x] for x in t) for t in tris]
    mesh = dict(v=v, f=f, az=0, el=70)
    degenerate, flipped = 2, 8
    areas = [normal(mesh, face)[2] for face in f]
    assert abs(areas[degenerate]) < 1e-12, "face (a, c, b) is not degenerate"
    assert areas[flipped] < 0 and all(a > 0 for k, a in enumerate(areas) if k not in (degenerate, flipped))
    assert max(edges(f).values()) <= 2
    he = {tuple(sorted((index[x], index[y]))) for x, y in
          (("a", "b"), ("b", "c"), ("a", "c"), ("c", "c2"), ("c2", "b2"), ("b2", "c"))}
    # The flipped face lies on top. It covers the part of the edge {t2, b2} below the edge {c, c2}, which is dotted.
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = (pos[k] for k in ("t2", "b2", "c", "c2"))
    u = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / ((x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3))
    assert 0 < u < 1
    crossing = (x1 + u * (x2 - x1), y1 + u * (y2 - y1), 0.0)
    hidden = [(crossing, v[index["b2"]])]
    return dict(mesh, hv={index["b"], index["b2"]}, he=he, hidden=hidden, skip_fill={degenerate}, flip={flipped})


def folded_sheet():
    """A sheet whose straight left border continues as a flap: it rises over the sheet and folds back down
    through it. The sheet is a jittered grid with randomly chosen diagonals."""
    rng = random.Random(19)  # chosen so that no two visible vertex markers nearly overlap
    nx, ny = 5, 4  # grid vertices along x and y
    xs = [-1.0 + 2.0 * i / (nx - 1) for i in range(nx)]
    ys = [-0.75 + 1.5 * j / (ny - 1) for j in range(ny)]
    grid = {}
    v = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            amp = 0.03 if i == 0 else 0.09
            grid[i, j] = len(v)
            v.append((x + amp * rng.uniform(-1, 1), y + amp * rng.uniform(-1, 1), 0.0))
    f = []
    for j in range(ny - 1):
        for i in range(nx - 1):
            a, b, c, d = grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]
            f += [(a, b, c), (a, c, d)] if rng.random() < 0.5 else [(a, b, d), (b, c, d)]
    border = [grid[0, j] for j in range(ny)]  # the left border, from front to back
    ridge, tip = [], []
    for row, (x, z) in ((ridge, RIDGE), (tip, TIP)):
        for b in border:
            row.append(len(v))
            v.append((x + 0.07 * rng.uniform(-1, 1), v[b][1] + 0.07 * rng.uniform(-1, 1), z + 0.07 * rng.uniform(-1, 1)))
    split = set()
    for lower, upper in ((border, ridge), (ridge, tip)):
        for k in range(ny - 1):
            split |= {len(f), len(f) + 1}
            f += [(lower[k], lower[k + 1], upper[k + 1]), (lower[k], upper[k + 1], upper[k])]
    f = orient(f)
    assert components(f) == 1 and max(edges(f).values()) <= 2
    assert all(v[r][2] > 0 > v[t][2] for r, t in zip(ridge, tip))  # the fold passes through the sheet
    return dict(v=v, f=f, hv=set(), he=set(), az=AZ, el=EL, split=split)


RIDGE, TIP, AZ, EL = (-0.3, 0.7), (0.75, -0.65), 25, 50



PANELS = (
    ("non-manifold\\\\vertex", hourglass),
    ("non-manifold\\\\edge", book),
    ("hole and\\\\separate component", holed_patch),
    ("degenerate and\\\\inverted face", degenerate_and_flipped),
    ("self-\\\\intersection", folded_sheet),
)


def bary(p, tri):
    """Barycentric coordinates of the 2D point p in the triangle tri, or None if p lies outside."""
    (x, y), ((x1, y1), (x2, y2), (x3, y3)) = p, tri
    det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(det) < 1e-12:
        return None
    l1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
    l2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
    l3 = 1 - l1 - l2
    return (l1, l2, l3) if min(l1, l2, l3) > -1e-9 else None


def polygons(mesh, toward):
    """Drawable polygons (layer, points, face index, flipped) in drawing order; flipped faces come last."""
    split = mesh.get("split", set())
    polys = []
    for k, face in enumerate(mesh["f"]):
        if k in mesh.get("skip_fill", set()):
            continue
        points = [mesh["v"][i] for i in face]
        if k in split:  # flap face: the parts below the sheet (layer 0) and above it (layer 3)
            for side, layer in ((-1, 0), (1, 3)):
                part = clip(points, side)
                if len(part) >= 3:
                    polys.append((layer, part, k, False))
        elif k in mesh.get("flip", set()):
            polys.append((9, points, k, True))
        else:
            polys.append((mesh["layer"][k] if "layer" in mesh else 2 if split else 0, points, k, False))
    return sorted(polys, key=lambda p: (p[0], sum(dot(q, toward) for q in p[1]) / len(p[1])))


def draw(out, panels):
    lines = [
        "% Generated by scripts/figures/mesh_defects.py; edit the generator, not this file.",
        r"\begingroup",
        define_color("opInk"),
        define_color("opNeutral"),
        define_color("opBefore"),
        r"\begin{tikzpicture}[",
        r"  md edge/.style={draw=opInk,line width=.45pt,line join=round},",
        r"  md hidden/.style={draw=opInk,line width=.9pt,line cap=round,dash pattern=on 0pt off 1.8pt},",
        r"  md mark/.style={draw=opBefore,line width=1.5pt,line cap=round},",
        r"  md vertex/.style={draw=opInk,fill=white,line width=.45pt},",
        r"  md hot/.style={draw=black,fill=opBefore,line width=.5pt},",
        r"  md label/.style={font=\footnotesize,text=opInk,align=center,anchor=north}]",
        rf"\path[use as bounding box] ({-PITCH / 2:.2f},{LABEL_Y - 0.62:.2f}) rectangle ({PITCH * 4.5:.2f},{BOX[1] / 2 + 0.05:.2f});",
    ]
    for col, (label, build) in enumerate(panels):
        mesh = build()
        right, up, toward = camera(mesh["az"], mesh["el"])
        light = norm(tuple(-0.45 * r + 0.65 * u + 0.6 * t for r, u, t in zip(right, up, toward)))
        screen = [(dot(p, right), dot(p, up)) for p in mesh["v"]]
        xs, ys = [s[0] for s in screen], [s[1] for s in screen]
        scale = min(BOX[0] / (max(xs) - min(xs)), BOX[1] / (max(ys) - min(ys)))
        cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2

        def project(p):
            return ((dot(p, right) - cx) * scale + PITCH * col, (dot(p, up) - cy) * scale)

        def fmt(p):
            q = project(p)
            return f"({q[0]:.4f},{q[1]:.4f})"

        polys = polygons(mesh, toward)

        def covered(p, skip):
            """Whether an opaque polygon of a face k with not skip(k) lies in front of the point p."""
            p2, depth = project(p), dot(p, toward)
            for layer, points, k, flipped in polys:
                if skip(k):
                    continue
                q2 = [project(q) for q in points]
                for j in range(1, len(points) - 1):
                    w = bary(p2, (q2[0], q2[j], q2[j + 1]))
                    if w and sum(wj * dot(q, toward) for wj, q in zip(w, (points[0], points[j], points[j + 1]))) > depth + 1e-6:
                        return True
            return False

        items = []  # (layer, kind, line); kind orders polygons before marks before vertex markers
        for layer, points, k, flipped in polys:
            path = " -- ".join(fmt(p) for p in points)
            shade = AMBIENT + (1 - AMBIENT) * abs(dot(norm(normal(mesh, mesh["f"][k])), light))
            if flipped:  # flipped face: tinted orange, its edges are marked separately
                items.append((layer, 0, rf"\fill[fill=opBefore!40!opNeutral!{100 * shade:.1f}!black] {path} -- cycle;"))
                continue
            items.append((layer, 0, rf"\filldraw[md edge,fill=opNeutral!{100 * shade:.1f}!black] {path} -- cycle;"))
        marks = [(mesh["v"][p], mesh["v"][q]) for p, q in sorted(mesh["he"])]
        for k in sorted(mesh.get("split", set())):  # visible parts of the intersection with the sheet
            points = [mesh["v"][i] for i in mesh["f"][k]]
            if not (min(q[2] for q in points) < 0 < max(q[2] for q in points)):
                continue
            cut = [q for q in clip(points, 1) if abs(q[2]) < 1e-12]
            samples = [tuple(a + t / 40 * (b - a) for a, b in zip(*cut)) for t in range(41)]
            visible = [not covered(q, lambda m: m == k or m not in mesh["split"]) for q in samples]
            run = None
            for q, seen in zip(samples + [None], visible + [False]):
                if seen and run is None:
                    run = q
                elif not seen and run is not None:
                    marks.append((run, last))
                    run = None
                last = q
        for p, q in marks:
            items.append((50, 1, rf"\draw[md mark] {fmt(p)} -- {fmt(q)};"))
        for p, q in mesh.get("hidden", []):
            items.append((50, 1, rf"\draw[md hidden] {fmt(p)} -- {fmt(q)};"))
        # A vertex marker is drawn if no opaque polygon of another face lies in front of it.
        used = {i for face in mesh["f"] for i in face}
        for i, p in enumerate(mesh["v"]):
            if i not in used:
                continue
            if covered(p, lambda k: i in mesh["f"][k]):
                continue
            p2 = project(p)
            style = "md hot" if i in mesh["hv"] else "md vertex"
            radius = 1.6 if i in mesh["hv"] else 1.25
            # Sheet vertices in (e) are drawn before the parts of the flap above the sheet.
            split = mesh.get("split", set())
            layer = 2 if split and not any(i in mesh["f"][k] for k in split) else 99
            items.append((layer, 2, rf"\draw[{style}] ({p2[0]:.4f},{p2[1]:.4f}) circle[radius={radius}pt];"))
        lines.append(f"% panel {chr(97 + col)}")
        lines += [line for _, _, line in sorted(items, key=lambda item: item[:2])]
        lines.append(rf"\node[md label] at ({PITCH * col:.2f},{LABEL_Y:.2f}) {{({chr(97 + col)}) {label}}};")
    lines += [r"\end{tikzpicture}", r"\endgroup"]
    out.write_text("\n".join(line + "%" for line in lines) + "\n")


def generate(out):
    draw(out / "mesh-defects.tikz", PANELS)
