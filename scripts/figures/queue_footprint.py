"""Generate the re-evaluation footprint figure of Chapter 3 (Section sec:queue).

Writes queue-footprint.tikz. A two-ring patch of a slightly perturbed triangulation whose centre vertex v
has valence 4, and each panel highlights the candidates of one operation type that the dynamic queue re-evaluates
after an edit at v. Two spokes of the regular lattice are flipped away and the interior relaxed; v lies at the
midpoint of two opposite neighbours. The patch can therefore be the result of relocating v, of splitting the edge
between those neighbours (v is the new vertex) or of a collapse into v. The sets follow the precise footprints of
the dynamic queue (src/remeshing/dynamic_queue.cpp), which are the same for these three
edits, with v as the only anchor:

  split and edge-loss flip: the edges of the faces incident to v (their evaluations read both incident triangles);
  collapse:                 the edges incident to v or to a neighbour of v (a collapse reads both one-rings);
  relocation:               v and its neighbours (a relocation reads its one-ring).

"""

from math import cos, sin, sqrt

from shared.style import define_color

RINGS = 2
SCALE = 0.62  # cm per lattice unit
GAP = 4.05  # cm between panel centres
CENTRE_FLIPS = ((0, 1), (0, -1))  # spokes of v that are flipped away, so that v has valence 4
SPLIT_ENDS = ((1, 0), (-1, 0))  # opposite neighbours of v; v is placed at their midpoint
RELAX_ITERATIONS = 200


def lattice(rings=RINGS):
    """Vertices of a regular triangulation within hex distance `rings` of the origin, and its faces."""
    coords = {}
    for i in range(-rings, rings + 1):
        for j in range(-rings, rings + 1):
            if max(abs(i), abs(j), abs(i + j)) <= rings:
                coords[(i, j)] = (i + 0.5 * j, j * sqrt(3) / 2)
    faces = []
    for (i, j) in coords:
        for tri in (((i, j), (i + 1, j), (i, j + 1)), ((i + 1, j), (i + 1, j + 1), (i, j + 1))):
            if all(t in coords for t in tri):
                faces.append(tri)
    return coords, faces


def perturb(coords):
    """Deterministic small jitter so the patch does not look like a perfect lattice; the centre stays put."""
    out = {}
    for (i, j), (x, y) in coords.items():
        if (i, j) == (0, 0):
            out[(i, j)] = (x, y)
            continue
        angle = 2.4 * i + 1.7 * j + 0.3 * i * j
        radius = 0.09 if max(abs(i), abs(j), abs(i + j)) == RINGS else 0.06
        out[(i, j)] = (x + radius * cos(angle), y + radius * sin(angle))
    return out


def flip_edge(faces, a, b):
    """Replaces the two faces at edge {a, b} by the two faces at the other diagonal, keeping the orientation."""
    f1 = next(f for f in faces if any(f[k] == a and f[(k + 1) % 3] == b for k in range(3)))
    f2 = next(f for f in faces if any(f[k] == b and f[(k + 1) % 3] == a for k in range(3)))
    c = next(v for v in f1 if v not in (a, b))
    d = next(v for v in f2 if v not in (a, b))
    return [f for f in faces if f not in (f1, f2)] + [(a, d, c), (d, b, c)]


def relax(pos, faces, movable):
    """Uniform Laplacian smoothing of the movable vertices; the others stay in place."""
    neighbours = {v: set() for v in pos}
    for a, b in edges_of(faces):
        neighbours[a].add(b)
        neighbours[b].add(a)
    for _ in range(RELAX_ITERATIONS):
        pos = {v: (sum(pos[u][0] for u in neighbours[v]) / len(neighbours[v]),
                   sum(pos[u][1] for u in neighbours[v]) / len(neighbours[v])) if v in movable else xy
               for v, xy in pos.items()}
    return pos


def edges_of(faces):
    edges = set()
    for f in faces:
        for k in range(3):
            edges.add(tuple(sorted((f[k], f[(k + 1) % 3]))))
    return edges


def hex_distance(a, b):
    di, dj = a[0] - b[0], a[1] - b[1]
    return max(abs(di), abs(dj), abs(di + dj))


def check_orientation(pos, faces):
    """The drawing must not fold a triangle."""
    for f in faces:
        (ax, ay), (bx, by), (cx, cy) = (pos[v] for v in f)
        assert (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 0, f


def relocation_row():
    coords, faces = lattice()
    centre = (0, 0)
    for spoke in CENTRE_FLIPS:
        faces = flip_edge(faces, centre, spoke)
    edges = edges_of(faces)
    neighbours = {u for e in edges if centre in e for u in e if u != centre}
    assert len(neighbours) == 4 and set(SPLIT_ENDS) <= neighbours
    # The one-ring of v becomes a square: the split ends stay on the lattice, the other two neighbours move
    # perpendicular to them, and the remaining inner vertices are relaxed around it.
    jittered = perturb(coords)
    pos = dict(coords)
    a, b = SPLIT_ENDS
    for c in neighbours - set(SPLIT_ENDS):
        side = 1 if coords[c][1] > 0 else -1
        pos[c] = (0.0, side * coords[a][0])
    inner = {v for v in coords if 0 < hex_distance(v, centre) < RINGS}
    pos = {v: jittered[v] if v not in inner and v != centre else xy for v, xy in pos.items()}
    pos = relax(pos, faces, inner - neighbours)
    # Add the jitter of the inner vertices; v lies at the midpoint of the split ends.
    pos = {v: (x + jittered[v][0] - coords[v][0], y + jittered[v][1] - coords[v][1]) if v in inner else (x, y)
           for v, (x, y) in pos.items()}
    pos[centre] = ((pos[a][0] + pos[b][0]) / 2, (pos[a][1] + pos[b][1]) / 2)

    incident_faces = [f for f in faces if centre in f]
    split_flip = edges_of(incident_faces)
    collapse = {e for e in edges if e[0] in neighbours | {centre} or e[1] in neighbours | {centre}}
    smooth = neighbours | {centre}
    check_orientation(pos, faces)

    def over_vertices(dx, dy):
        cx, cy = pos[centre]
        return [rf"\node[font=\scriptsize,text=opInk,anchor=south west,inner sep=1pt] at "
                rf"({cx * SCALE + dx + 0.04:.4f},{cy * SCALE + dy + 0.04:.4f}) {{$v$}};"]

    return {
        "pos": pos, "faces": faces, "edges": edges, "shaded": set(incident_faces), "dark_vertices": {centre},
        "over_vertices": over_vertices,
        "panels": (
            ("(a) split and flip", split_flip, set()),
            ("(b) collapse", collapse, set()),
            ("(c) relocation", set(), smooth),
        ),
    }


def point(pos, v, dx, dy=0.0):
    x, y = pos[v]
    return f"({x * SCALE + dx:.4f},{y * SCALE + dy:.4f})"


def draw_row(lines, row, title_y):
    pos = row["pos"]
    for k, (title, hit_edges, hit_vertices) in enumerate(row["panels"]):
        dx = k * GAP
        px, py = dx, 0.0
        lines.append(f"% panel {k}")
        lines.append(rf"\node[fp label] at ({dx:.3f},{title_y:.3f}) {{{title}}};")
        for f in row["faces"]:
            fill = "opAfterFill" if f in row["shaded"] else "opNeutral"
            lines.append(rf"\fill[{fill}] " + " -- ".join(point(pos, v, px, py) for v in f) + " -- cycle;")
        for e in sorted(row["edges"]):
            style = "fp hit" if e in hit_edges else "fp edge"
            lines.append(rf"\draw[{style}] {point(pos, e[0], px, py)} -- {point(pos, e[1], px, py)};")
        for v in sorted(pos):
            if v in row["dark_vertices"]:
                style = "fp vertex,fill=opInk"
            elif v in hit_vertices:
                style = "fp vertex,fill=opAfter"
            else:
                style = "fp vertex"
            size = "2.1pt" if v in hit_vertices or v in row["dark_vertices"] else "1.5pt"
            lines.append(rf"\draw[{style}] {point(pos, v, px, py)} circle[radius={size}];")
        lines += row["over_vertices"](px, py)


def generate(out):
    lines = [
        "% Generated by scripts/figures/queue_footprint.py; edit the generator for changes.",
        r"\begingroup",
        define_color("opInk"),
        define_color("opNeutral"),
        define_color("opAfter"),
        define_color("opAfterFill"),
        r"\begin{tikzpicture}[",
        r"  fp edge/.style={draw=opInk!55,line width=.45pt,line cap=round},",
        r"  fp hit/.style={draw=opAfter,line width=1.25pt,line cap=round},",
    ]
    lines += [
        r"  fp vertex/.style={draw=opInk,fill=white,line width=.45pt},",
        r"  fp label/.style={font=\footnotesize\bfseries,text=opInk}]",
    ]
    top = relocation_row()
    radius = (RINGS + 0.25) * SCALE
    draw_row(lines, top, radius + 0.25)

    lines += [r"\end{tikzpicture}", r"\endgroup"]
    (out / "queue-footprint.tikz").write_text("\n".join(line + "%" for line in lines) + "\n")
