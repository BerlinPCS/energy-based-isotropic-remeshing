"""An edit given as two OBJ files, the patch before and after it, as they come from Blender or PMP. What the edit changed
is worked out from the two files, so the files can be replaced without touching the figures that draw them."""
from collections import Counter

TOLERANCE = 1e-5  # for the decimal precision of OBJ files
# How many vertices and faces each operation adds.
OPERATION_CHANGES = {"split": (1, 2), "collapse": (-1, -2), "flip": (0, 0), "smooth": (0, 0), "relocation": (0, 0)}


def read_obj(path, y_up=False):
    """Vertices and triangles. y_up: the file is stored as (x, z, -y), which is Blender's default export of a z-up model;
    the vertices come back z-up."""
    vertices, faces = [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            x, y, z = map(float, fields[1:4])
            vertices.append((x, -z, y) if y_up else (x, y, z))
        elif fields[0] == "f":
            ids = [int(field.split("/")[0]) for field in fields[1:]]
            face = tuple(i - 1 if i > 0 else len(vertices) + i for i in ids)
            if len(face) != 3 or len(set(face)) != 3:
                raise ValueError(f"Expected nondegenerate triangles: {path}")
            if any(i < 0 or i >= len(vertices) for i in face):
                raise ValueError(f"Invalid vertex index: {path}")
            faces.append(face)
    return vertices, faces


def edges(faces):
    return Counter(tuple(sorted((f[i], f[(i + 1) % 3]))) for f in faces for i in range(3))


def correspondence(before, after):
    """Identify unchanged 3D positions with tolerance for OBJ decimal precision. Vertex order is not assumed to persist
    across exports."""
    labels_before = list(range(len(before)))
    labels_after = []
    for j, q in enumerate(after):
        matches = [i for i, p in enumerate(before) if max(abs(a - b) for a, b in zip(p, q)) < TOLERANCE]
        if len(matches) > 1:
            raise ValueError("Ambiguous coincident vertices")
        labels_after.append(matches[0] if matches else len(before) + j)
    return labels_before, labels_after


def signature(element, labels):
    """An edge or face, independent of the vertex order of its file."""
    return tuple(sorted(labels[i] for i in element))


def signatures(elements, labels):
    return {signature(element, labels) for element in elements}


class EditPair:
    """before and after are (vertices, faces). A vertex has the same label in both meshes if its position is unchanged, so
    a relocated vertex counts as removed and added. Edges and faces below are sets of signatures."""

    def __init__(self, operation, before, after):
        self.operation, self.before, self.after = operation, before, after
        self.labels_before, self.labels_after = correspondence(before[0], after[0])
        self.common_faces = signatures(before[1], self.labels_before) & signatures(after[1], self.labels_after)
        edges_before, edges_after = edges(before[1]), edges(after[1])
        signatures_before = signatures(edges_before, self.labels_before)
        signatures_after = signatures(edges_after, self.labels_after)
        self.common_edges = signatures_before & signatures_after
        self.removed_vertices = set(self.labels_before) - set(self.labels_after)
        self.added_vertices = set(self.labels_after) - set(self.labels_before)

        # The edge the operation acts on: the one that disappears (flip, split), or the one between two removed vertices
        # (collapse). A relocation has none, it acts on the removed vertex.
        removed_edges = signatures_before - self.common_edges
        if self.removed_vertices:
            self.operand_edges = {e for e in removed_edges if set(e) <= self.removed_vertices}
        else:
            self.operand_edges = removed_edges
        # The edge a flip creates; the other new edges all end at an added vertex.
        self.new_edges = {e for e in signatures_after - self.common_edges if not set(e) & self.added_vertices}

        # Smoothing retains connectivity even though its central position changes.
        if operation in ("smooth", "relocation"):
            if len(self.removed_vertices) != 1 or len(self.added_vertices) != 1:
                raise ValueError(f"Expected exactly one moved vertex: {operation}")
            (added,), (removed,) = self.added_vertices, self.removed_vertices
            if signatures(after[1], [removed if label == added else label for label in self.labels_after]) != \
                    signatures(before[1], self.labels_before):
                raise ValueError("Relocation changed connectivity")
        if not (self.operand_edges or self.removed_vertices) or len(self.operand_edges) > 1:
            raise ValueError(f"Unexpected operand: {operation}")
        change = tuple(len(after[i]) - len(before[i]) for i in (0, 1))
        if change != OPERATION_CHANGES[operation] or any(n > 2 for n in (*edges_before.values(), *edges_after.values())):
            raise ValueError(f"Unexpected operation topology: {operation}")
