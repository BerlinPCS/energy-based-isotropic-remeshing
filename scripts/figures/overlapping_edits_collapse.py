"""Generate overlapping-edits-collapse.tikz (alternative to Figure "Overlapping candidate edits", Ch. 2).

A flat strip with a long edge {a,b} (candidate split o1) and a short edge {b,t} that
shares the endpoint b and lies in the same triangle (candidate midpoint collapse o2). Both
improve the summed edge loss sum_e log2(l(e)/L)^2 (L = 1) on their own. After the split,
the collapse still improves. After the collapse, b has moved and the split edge has a
different opposite vertex, and the split no longer improves. The two orders therefore end
in different meshes. Positions were found by a random search for this sign pattern; all
gains printed in the figure are computed here.

A collinear arrangement (short edge continuing the long one) does not work: there the two
edits always reinforce each other, so their gains change but never their sign.
"""

import math

from common.style import define_color

L = 1.0
SHIFT = -0.28  # centres the strip horizontally

MID = ["p0", "a", "b", "c", "p1"]
TOP = ["t0", "t", "t2", "t3"]
BOT = ["b0", "b1", "b2", "b3"]
POS = {
    "p0": (-1.93, 0.0), "a": (-0.95, 0.0), "b": (0.73, -0.07), "c": (1.54, 0.0), "p1": (2.5, 0.0),
    "t0": (-1.59, 0.92), "t": (0.48, 0.73), "t2": (0.91, 0.93), "t3": (1.83, 0.8),
    "b0": (-1.35, -0.83), "b1": (0.03, -0.8), "b2": (1.24, -0.77), "b3": (1.87, -0.8),
}
POS = {k: (x + SHIFT, y) for k, (x, y) in POS.items()}
FACES = (
    [(MID[i], MID[i + 1], TOP[i]) for i in range(4)]
    + [(MID[i + 1], TOP[i + 1], TOP[i]) for i in range(3)]
    + [(MID[i + 1], MID[i], BOT[i]) for i in range(4)]
    + [(MID[i + 1], BOT[i], BOT[i + 1]) for i in range(3)]
    + [("t0", "t", "t2")]  # makes t interior, so the collapse leaves the patch border in place
)


def edges(faces):
    return {frozenset(e) for f in faces for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}


def energy(pos, faces):
    return sum(math.log2(math.dist(*(pos[x] for x in e)) / L) ** 2 for e in edges(faces))


def signed_area(pos, f):
    (ax, ay), (bx, by), (cx, cy) = (pos[x] for x in f)
    return 0.5 * ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))


def rotate_to(f, a, b):
    for i in range(3):
        g = f[i:] + f[:i]
        if {g[0], g[1]} == {a, b}:
            return g
    raise ValueError


def neighbors(faces, v):
    return {x for f in faces if v in f for x in f} - {v}


def split(pos, faces, a, b, m="m"):
    """Midpoint split of {a,b}: each incident triangle (p,q,r) becomes (p,m,r), (m,q,r)."""
    pos = dict(pos)
    pos[m] = tuple((pa + pb) / 2 for pa, pb in zip(pos[a], pos[b]))
    out = []
    for f in faces:
        if a in f and b in f:
            p, q, r = rotate_to(f, a, b)
            out += [(p, m, r), (m, q, r)]
        else:
            out.append(f)
    return pos, out


def collapse(pos, faces, a, b):
    """Midpoint collapse of {a,b}; the merged vertex keeps the name a."""
    incident = [f for f in faces if a in f and b in f]
    assert len(neighbors(faces, a) & neighbors(faces, b)) == len(incident), "link condition"
    pos = dict(pos)
    pos[a] = tuple((pa + pb) / 2 for pa, pb in zip(pos[a], pos[b]))
    del pos[b]
    return pos, [tuple(a if x == b else x for x in f) for f in faces if f not in incident]


def gain(mesh, op):
    after = op(*mesh)
    return energy(*mesh) - energy(*after), after


def valid(mesh):
    pos, faces = mesh
    return all(signed_area(pos, f) > 0 for f in faces)


SPLIT = lambda pos, faces: split(pos, faces, "a", "b")
COLLAPSE = lambda pos, faces: collapse(pos, faces, "b", "t")

M = (POS, FACES)
assert valid(M)
g1, after_split = gain(M, SPLIT)
g2, after_collapse = gain(M, COLLAPSE)
g2_after, split_then_collapse = gain(after_split, COLLAPSE)
g1_after, collapse_then_split = gain(after_collapse, SPLIT)
assert all(valid(m) for m in (after_split, after_collapse, split_then_collapse, collapse_then_split))
assert g1 > 0 and g2 > 0 and g2_after > 0 and g1_after < 0


def panel(mesh, x0, title, highlights, new_vertex, new_edges, note):
    """highlights: list of (edge, style, label, label options)."""
    pos, faces = mesh
    pt = lambda n: f"({pos[n][0]:.3f},{pos[n][1]:.3f})"
    s = [f"\\begin{{scope}}[shift={{({x0:.2f},0)}}]%", f"\\node[ov title] at (0,1.3) {{{title}}};%"]
    for f in faces:
        s.append("\\fill[opNeutral] " + " -- ".join(pt(x) for x in f) + " -- cycle;%")
    special = {frozenset(e[:2]) for e, *_ in highlights} | {frozenset(e) for e in new_edges}
    for e in sorted(edges(faces), key=sorted):
        if e not in special:
            a, b = sorted(e)
            s.append(f"\\draw[ov edge] {pt(a)} -- {pt(b)};%")
    for a, b in new_edges:
        s.append(f"\\draw[ov new] {pt(a)} -- {pt(b)};%")
    for (a, b), style, label, opts in highlights:
        s.append(f"\\draw[{style}] {pt(a)} -- node[ov label,{opts}] {{{label}}} {pt(b)};%")
    for name in sorted({x for f in faces for x in f}):
        fill = "opAfter" if name == new_vertex else "white"
        s.append(f"\\draw[ov vertex,fill={fill}] {pt(name)} circle[radius=1.6pt];%")
    s.append(f"\\node[ov label,align=center] at (0,-1.4) {{{note}}};%")
    s.append("\\end{scope}%")
    return s



def generate(out):
    lines = [
        "% Generated by scripts/figures/overlapping_edits_collapse.py (DRAFT); edit the generator, not this file.%",
        f"% Summed edge loss, L = 1: E(M) = {energy(*M):.3f}; g(split) = {g1:.3f}, g(collapse) = {g2:.3f};%",
        f"% after the split: g(collapse) = {g2_after:.3f}; after the collapse: g(split) = {g1_after:.3f}.%",
        "\\begingroup%",
        define_color("opInk") + "%",
        define_color("opNeutral") + "%",
        define_color("opBefore") + "%",
        define_color("opAfter") + "%",
        "\\begin{tikzpicture}[x=0.75cm,y=0.75cm,%",
        "  ov edge/.style={draw=opInk,line width=.5pt,line cap=round},%",
        "  ov new/.style={draw=opAfter,line width=1.2pt,line cap=round},%",
        "  ov cand/.style={draw=opBefore,line width=1.6pt,line cap=round},%",
        "  ov dead/.style={draw=opBefore,line width=1.2pt,dashed,line cap=round},%",
        "  ov vertex/.style={draw=opInk,line width=.5pt},%",
        "  ov label/.style={font=\\footnotesize,text=opInk,inner sep=1.2pt},%",
        "  ov title/.style={font=\\footnotesize\\bfseries,text=opInk}]%",
    ]
    lines += panel(
        M, 0.0, "$\\mathcal{M}$",
        [(("a", "b"), "ov cand", "$o_1$", "below=1pt,fill=opNeutral"),
         (("b", "t"), "ov cand", "$o_2$", "left=1pt,fill=opNeutral")],
        None, [], f"split: $g(o_1)={g1:.2f}$\\\\collapse: $g(o_2)={g2:.2f}$",
    )
    lines += panel(
        after_collapse, -4.9, "after the collapse $o_2$",
        [(("a", "b"), "ov dead", "$o_1$", "below=1pt,fill=opNeutral")],
        "b", [], f"split: $g(o_1)={g1_after:.2f}$",
    )
    lines += panel(
        after_split, 4.9, "after the split $o_1$",
        [(("b", "t"), "ov cand", "$o_2$", "pos=0.35,left=1pt,fill=opNeutral")],
        "m", [("a", "m"), ("m", "b"), ("m", "t"), ("m", "b1")], f"collapse: $g(o_2)={g2_after:.2f}$",
    )
    lines += ["\\end{tikzpicture}%", "\\endgroup%"]
    (out / "overlapping-edits-collapse.tikz").write_text("\n".join(lines) + "\n")
