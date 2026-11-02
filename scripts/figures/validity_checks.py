"""Generate the normal-check figure of Chapter 3 (validity checks in sec:shared-operations).

Writes validity-checks.tikz from the OBJ pairs in validity_checks_meshes/, one per checked operation: `flip.obj` and
`flip_result.obj`, and so on. They hold only the triangles the edit replaces and the new ones, stored as Blender's default
OBJ export of a z-up model. Everything else follows from the two files: which edge or vertex the edit acts on, the angle
between each triangle's normal and n, the summed area-weighted normal of the replaced triangles, and which triangles
turn over. So the files can be edited or replaced, as long as each pair shows the operation of its column, no replaced
triangle lies 60 degrees or more from n, and at least one new triangle does, which the script checks. In the shipped
pairs more new than old triangles lie outside 60 degrees, so the edit is rejected (src/remeshing/validity.cpp; for the
flip the strict test, for the others the ring-bounded rule).

The top row shows the replaced triangles, the bottom row the new ones, seen obliquely from above in perspective (a
turntable camera per patch, as in Blender's viewport). Faces are lit by a light fixed to the camera (upper left, in
front), like Blender's default studio light.
"""

from math import cos, pi, sin, sqrt
from pathlib import Path

from common.edit_pairs import EditPair, read_obj, signature
from common.style import define_color
from common.tikz3d import Camera, angle, edges_of, normal

MESHES = Path(__file__).resolve().parent / "validity_checks_meshes"
# The columns of the figure: operation, camera azimuth and elevation in degrees.
COLUMNS = (("flip", -110, 40), ("collapse", 134, 40), ("relocation", 34, 42))
LIMIT = 60  # degrees between a triangle's normal and n
SCALE = 0.95  # cm per model unit
GAP = 3.55  # cm between column centres
ROW = 2.35  # cm between row centres
LIGHT = (-0.35, 0.55, 0.75)  # camera coordinates (right, up, towards the viewer)
AMBIENT = 0.55
DISTANCE = 3.0  # camera distance in patch radii (perspective strength)


def label_size(text):
    """Estimated width and height in cm of an angle label (\\tiny in the 10 pt thesis, with inner sep)."""
    return 0.092 * sum(ch.isdigit() for ch in text) + 0.1, 0.17


def inside(p, tri, margin=0.0):
    """Whether p lies inside the 2D triangle, at least `margin` cm from its edges."""
    (ax, ay), (bx, by), (cx, cy) = tri
    area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if abs(area) < 1e-12:
        return False
    sign = 1 if area > 0 else -1
    for (px, py), (qx, qy) in (((ax, ay), (bx, by)), ((bx, by), (cx, cy)), ((cx, cy), (ax, ay))):
        cross_ = sign * ((qx - px) * (p[1] - py) - (qy - py) * (p[0] - px))
        if cross_ < margin * sqrt((qx - px) ** 2 + (qy - py) ** 2):
            return False
    return True


def touches(c, w, h, points):
    """Whether the label box (centre c, size w x h) comes within 0.03 cm of a vertex disc (x, y, radius)."""
    for x, y, r in points:
        dx, dy = max(abs(c[0] - x) - w / 2, 0.0), max(abs(c[1] - y) - h / 2, 0.0)
        if dx * dx + dy * dy < (r + 0.03) ** 2:
            return True
    return False


def edge_distance(p, tri):
    """Signed distance in cm from p to the nearest edge of the 2D triangle, positive inside."""
    (ax, ay), (bx, by), (cx, cy) = tri
    sign = 1 if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) > 0 else -1
    return min(sign * ((qx - px) * (p[1] - py) - (qy - py) * (p[0] - px)) / sqrt((qx - px) ** 2 + (qy - py) ** 2)
               for (px, py), (qx, qy) in (((ax, ay), (bx, by)), ((bx, by), (cx, cy)), ((cx, cy), (ax, ay))))


def rect_samples(c, w, h):
    return [(c[0] + fx * w / 2, c[1] + fy * h / 2) for fx in (-1, 0, 1) for fy in (-1, 0, 1)]


def place_labels(tris, texts, points, over, thick=()):
    """Label centres (cm) for the triangles `tris` (2D screen triangles). A label goes inside its triangle, at the spot
    farthest from its edges, the triangles drawn over it (`over[i]`) and the vertices, if it fits there; otherwise just outside one of the triangle's
    edges, at the candidate that overlaps the fewest other triangles, labels and vertices (`points`, as discs), so preferably
    outside the mesh."""
    centres, boxes, leaders = [None] * len(tris), [], {}
    for i, (tri, text) in enumerate(zip(tris, texts)):
        w, h = label_size(text)
        (ax, ay), (bx, by), (cx, cy) = tri
        la, lb, lc = (sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2) for p, q in ((tri[1], tri[2]), (tri[2], tri[0]), (tri[0], tri[1])))
        c = ((la * ax + lb * bx + lc * cx) / (la + lb + lc), (la * ay + lb * by + lc * cy) / (la + lb + lc))
        # Candidates: the incenter and a barycentric grid. The digits' ink (without the node's inner sep) has to
        # fit inside the triangle, outside the triangles drawn over it and clear of the vertices; take the candidate
        # with the largest clearance from all three.
        grid = [((u * ax + v * bx + (40 - u - v) * cx) / 40, (u * ay + v * by + (40 - u - v) * cy) / 40)
                for u in range(1, 39) for v in range(1, 40 - u)]
        iw, ih = 0.045 * len(text) + 0.045, 0.07  # measured: digits 0.09 cm, degree sign 0.085 cm, height 0.14 cm
        best = None
        for c in [c] + grid:
            corners = [(c[0] + fx * iw, c[1] + fy * ih) for fx in (-1, 1) for fy in (-1, 1)]
            clearance = min(edge_distance(q, tri) for q in corners)
            for j in over[i]:
                clearance = min(clearance, min(-edge_distance(q, tris[j]) for q in corners))
            # Vertices as discs; the highlighted thick edges as chains of discs of their half width.
            for x, y, r in points + [(p[0] + t / 20 * (q[0] - p[0]), p[1] + t / 20 * (q[1] - p[1]), 0.023 - 0.009)
                                     for p, q in thick for t in range(21)]:
                dx, dy = max(abs(c[0] - x) - iw, 0.0), max(abs(c[1] - y) - ih, 0.0)
                clearance = min(clearance, sqrt(dx * dx + dy * dy) - r)
            if clearance >= 0.025 and (best is None or clearance > best[0]):
                best = (clearance, c)
        if best is not None:
            centres[i] = best[1]
            boxes.append((best[1], w, h))
    for i, (tri, text) in enumerate(zip(tris, texts)):
        if centres[i] is not None:
            continue
        w, h = label_size(text)
        best = None
        for k in range(3):
            p, q, r = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
            ex, ey = q[0] - p[0], q[1] - p[1]
            length = sqrt(ex * ex + ey * ey)
            nx, ny = ey / length, -ex / length
            if nx * (r[0] - p[0]) + ny * (r[1] - p[1]) > 0:  # point away from the opposite vertex
                nx, ny = -nx, -ny
            d = 0.03 + abs(nx) * w / 2 + abs(ny) * h / 2
            for t in (0.5, 0.35, 0.65, 0.2, 0.8):
                c = (p[0] + t * ex + nx * d, p[1] + t * ey + ny * d)
                samples = rect_samples(c, w, h)
                cost = 10 * sum(inside(sp, other) for j, other in enumerate(tris) if j != i for sp in samples)
                cost += 100 * sum(abs(c[0] - o[0]) < (w + ow) / 2 + 0.02 and abs(c[1] - o[1]) < (h + oh) / 2 + 0.02
                                  for o, ow, oh in boxes)
                cost += 50 * touches(c, w, h, points)
                cost += abs(t - 0.5)
                if best is None or cost < best[0]:
                    best = (cost, c)
        if best[0] >= 10:
            # Beside the triangle it would cover another one: put it where it covers nothing, with a leader line
            # from the point of the triangle farthest from its edges.
            (ax, ay), (bx, by), (cx, cy) = tri
            anchor = max((((u * ax + v * bx + (40 - u - v) * cx) / 40, (u * ay + v * by + (40 - u - v) * cy) / 40)
                          for u in range(1, 39) for v in range(1, 40 - u)), key=lambda g: edge_distance(g, tri))
            for a in range(24):
                ux, uy = cos(2 * pi * a / 24), sin(2 * pi * a / 24)
                for d in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                    c = (anchor[0] + ux * d, anchor[1] + uy * d)
                    samples = rect_samples(c, w, h)
                    cost = 10 * sum(inside(sp, other) for other in tris for sp in samples)
                    cost += 100 * sum(abs(c[0] - o[0]) < (w + ow) / 2 + 0.02 and abs(c[1] - o[1]) < (h + oh) / 2 + 0.02
                                      for o, ow, oh in boxes)
                    cost += 50 * touches(c, w, h, points)
                    cost += d
                    if cost < best[0]:
                        best = (cost, c)
                        leaders[i] = anchor
        centres[i] = best[1]
        boxes.append((best[1], w, h))
    return centres, leaders


def read_column(operation):
    """The edit of one column, and the angles of its replaced and new triangles to n."""
    pair = EditPair(operation, read_obj(MESHES / f"{operation}.obj", y_up=True),
                    read_obj(MESHES / f"{operation}_result.obj", y_up=True))
    if pair.common_faces:
        raise ValueError(f"{operation}: the OBJ files must hold only the triangles the edit replaces and the new ones")
    (old_positions, old_faces), (new_positions, new_faces) = pair.before, pair.after
    n = (0.0, 0.0, 0.0)
    for face in old_faces:
        n = tuple(a + b for a, b in zip(n, normal(old_positions, face)))
    old_angles = [angle(normal(old_positions, face), n) for face in old_faces]
    new_angles = [angle(normal(new_positions, face), n) for face in new_faces]
    if max(old_angles) >= LIMIT:
        raise ValueError(f"{operation}: a replaced triangle lies {max(old_angles):.0f} degrees from n")
    if max(new_angles) < LIMIT:
        raise ValueError(f"{operation}: no new triangle turns over, the largest angle to n is {max(new_angles):.0f} degrees")
    return pair, old_angles, new_angles


class Panel:
    """One mesh of a column: its triangles, how they are coloured, and the elements the edit acts on or creates."""

    def __init__(self, column, row, mesh, angles, labels, fill, ink, thick_edges, marked_vertices, camera, centre):
        self.column, self.row = column, row
        self.positions, self.faces = mesh
        self.angles, self.labels, self.fill, self.ink = angles, labels, fill, ink
        self.thick_edges, self.marked_vertices = thick_edges, marked_vertices
        self.camera, self.centre = camera, centre
        self.offset = column * GAP, -row * ROW
        # Edges shared with a triangle that passes are drawn in its color only, not red and not dashed.
        self.passing = set(edges_of([f for f, a in zip(self.faces, angles) if a < LIMIT]))

    def screen(self, point):
        x, y, _ = self.camera.view(point)
        return (x - self.centre[0]) * SCALE + self.offset[0], (y - self.centre[1]) * SCALE + self.offset[1]

    def at(self, v):
        return self.screen(self.positions[v])

    def on_edge(self, e, t):
        return self.screen(tuple(a + t * (b - a) for a, b in zip(self.positions[e[0]], self.positions[e[1]])))

    def edge_style(self, e, failing):
        if signature(e, self.labels) in self.thick_edges:
            return "draw=opOperand,line width=1.3pt" if self.row == 0 else "draw=opAfter,line width=1.3pt"
        return "draw=opFail,line width=.8pt" if failing and e not in self.passing else f"draw={self.ink}"

    def draw(self, lines):
        self.draw_faces(lines)
        self.draw_turned_faces(lines)
        self.draw_vertices(lines)
        self.draw_angles(lines)

    def draw_faces(self, lines):
        # Painter's algorithm: each face draws its fill and its edges, back to front, so that nearer faces cover the
        # fills, edges and red outlines behind them. A shared edge is drawn by both faces; the nearer one wins.
        for i in self.camera.back_to_front(self.positions, self.faces):
            failing = self.angles[i] >= LIMIT
            brightness = self.camera.brightness(normal(self.positions, self.faces[i]))
            style = "fill=opFail!30" if failing else f"fill={self.fill}!{brightness:.1f}!black"
            lines.append(rf"\fill[{style}] " + polygon(self.at(v) for v in self.faces[i]))
            for e in edges_of([self.faces[i]]):
                lines.append(rf"\draw[vc edge,{self.edge_style(e, failing)}] {coordinate(self.at(e[0]))} -- {coordinate(self.at(e[1]))};")

    def draw_turned_faces(self, lines):
        # The turned triangles may lie behind others (the flip's lies entirely behind cdb): repeat them on top with a
        # slight tint, and dash the parts of their edges that lie behind other faces (hidden lines).
        dashed = []  # (edge, t0, t1)
        for face, a in zip(self.faces, self.angles):
            if a >= LIMIT:
                lines.append(r"\fill[opFail,fill opacity=.15] " + polygon(self.at(v) for v in face))
                for e in edges_of([face]):
                    dashed += [(e, t0, t1) for t0, t1 in self.camera.edge_runs(self.positions, self.faces, e, True)
                               if (e, t0, t1) not in dashed]
        for e, t0, t1 in dashed:
            lines.append(rf"\draw[vc edge,vc hidden] {coordinate(self.on_edge(e, t0))} -- {coordinate(self.on_edge(e, t1))};")
        # Redraw the visible parts of the edges that run close to or across a dashed line, in the style their nearest
        # face drew them with, so that the dashes stay below them.
        style_of = {}
        for i in self.camera.back_to_front(self.positions, self.faces):
            for e in edges_of([self.faces[i]]):
                style_of[e] = self.edge_style(e, self.angles[i] >= LIMIT)
        for e, style in style_of.items():
            for t0, t1 in self.camera.edge_runs(self.positions, self.faces, e, False):
                if self.runs_near_dashes(e, t0, t1, dashed):
                    lines.append(rf"\draw[vc edge,{style}] {coordinate(self.on_edge(e, t0))} -- {coordinate(self.on_edge(e, t1))};")

    def runs_near_dashes(self, e, t0, t1, dashed):
        for dashed_edge, s0, s1 in dashed:
            (px, py), (qx, qy) = self.on_edge(dashed_edge, s0), self.on_edge(dashed_edge, s1)
            ex, ey = qx - px, qy - py
            for step in range(1, 40):
                x, y = self.on_edge(e, t0 + (t1 - t0) * step / 40)
                t = max(0.0, min(1.0, ((x - px) * ex + (y - py) * ey) / max(ex * ex + ey * ey, 1e-12)))
                if (x - px - t * ex) ** 2 + (y - py - t * ey) ** 2 < 0.05 ** 2:
                    return True
        return False

    def is_marked(self, v):
        return self.labels[v] in self.marked_vertices

    def used_vertices(self):
        return sorted({v for f in self.faces for v in f})

    def draw_vertices(self, lines):
        for v in self.used_vertices():
            if self.is_marked(v):
                style, size = ("vc vertex,fill=opOperand" if self.row == 0 else "vc vertex,fill=opAfter"), "2pt"
            else:
                style, size = "vc vertex", "1.4pt"
            lines.append(rf"\draw[{style}] {coordinate(self.at(v))} circle[radius={size}];")

    def draw_angles(self, lines):
        faces, angles = self.faces, self.angles
        texts = [f"{a:.0f}" for a in angles]
        depth = [self.camera.face_depth(self.positions, face) for face in faces]
        # Drawn over face i: the nearer faces, and the turned faces, which are repeated on top.
        over = [[j for j in range(len(faces)) if j != i and (depth[j] > depth[i] or angles[j] >= LIMIT)]
                for i in range(len(faces))]
        discs = [(*self.at(v), 0.07 if self.is_marked(v) else 0.05) for v in self.used_vertices()]  # 2pt and 1.4pt radii
        thick = [(self.at(a), self.at(b)) for a, b in edges_of(faces) if signature((a, b), self.labels) in self.thick_edges]
        centres, leaders = place_labels([tuple(self.at(v) for v in f) for f in faces], texts, discs, over, thick)
        name = f"l{self.column}{self.row}"
        for i in range(len(faces)):
            failing = angles[i] >= LIMIT
            lines.append(rf"\node[{'vc fail' if failing else 'vc angle'}] ({name}{i}) at {coordinate(centres[i])} {{${texts[i]}^\circ$}};")
            if i in leaders:
                lines.append(rf"\draw[{'opFail' if failing else 'opInk'},line width=.3pt] {coordinate(leaders[i])} -- ({name}{i});")


def coordinate(point):
    return f"({point[0]:.4f},{point[1]:.4f})"


def polygon(points):
    return " -- ".join(coordinate(p) for p in points) + " -- cycle;"


def generate(out):
    lines = [
        "% Generated by scripts/figures/validity_checks.py from the OBJ files in validity_checks_meshes.",
        r"\begingroup",
        define_color("opInk"),
        define_color("opBefore"),
        define_color("opOperand"),
        define_color("opBeforeFill"),
        define_color("opAfter"),
        define_color("opAfterFill"),
        define_color("opFail"),
        r"\begin{tikzpicture}[",
        r"  vc edge/.style={line width=.5pt,line join=round,line cap=round},",
        r"  vc vertex/.style={draw=opInk,fill=white,line width=.45pt},",
        r"  vc angle/.style={font=\tiny,text=opInk,inner sep=.5pt},",
        r"  vc fail/.style={font=\tiny\bfseries,text=opFail,inner sep=.5pt},",
        r"  vc hidden/.style={draw=opFail,line width=.6pt,dash pattern=on 1.6pt off 1.1pt},",
        r"  vc label/.style={font=\footnotesize\bfseries,text=opInk}]",
    ]
    left = 0.0
    for column, (operation, azimuth, elevation) in enumerate(COLUMNS):
        pair, old_angles, new_angles = read_column(operation)
        points = pair.before[0] + pair.after[0]
        target = tuple((min(p[i] for p in points) + max(p[i] for p in points)) / 2 for i in range(3))
        radius = max(sqrt((p[0] - target[0]) ** 2 + (p[1] - target[1]) ** 2) for p in points)
        camera = Camera(target, azimuth, elevation, DISTANCE * radius, LIGHT, AMBIENT)
        views = [camera.view(p) for p in points]
        centre = (min(p[0] for p in views) + max(p[0] for p in views)) / 2, (min(p[1] for p in views) + max(p[1] for p in views)) / 2
        top = (max(p[1] for p in views) - centre[1]) * SCALE

        operand_vertices = pair.removed_vertices | {v for edge in pair.operand_edges for v in edge}
        for row, panel in enumerate((
                Panel(column, 0, pair.before, old_angles, pair.labels_before, "opBeforeFill", "opBefore", pair.operand_edges,
                      operand_vertices, camera, centre),
                Panel(column, 1, pair.after, new_angles, pair.labels_after, "opAfterFill", "opAfter", pair.new_edges,
                      pair.added_vertices, camera, centre))):
            lines.append(f"% {operation}, {'after' if row else 'before'}")
            panel.draw(lines)
            left = min([left] + [panel.at(v)[0] for v in panel.used_vertices()])
        lines.append(rf"\node[vc label] at ({column * GAP:.3f},{top + 0.35:.3f}) {{({'abc'[column]}) {operation}}};")
    for row, text in enumerate(("before", "after")):
        lines.append(rf"\node[font=\footnotesize,text=opInk,anchor=east] at ({left - 0.25:.3f},{-row * ROW:.3f}) "
                     rf"{{{text}}};")
    lines += [r"\end{tikzpicture}", r"\endgroup"]
    (out / "validity-checks.tikz").write_text("\n".join(line + "%" for line in lines) + "\n")
