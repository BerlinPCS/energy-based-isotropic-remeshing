"""Small 3D patches as TikZ: a perspective camera with flat shading, painter's order and hidden parts of edges."""
from math import acos, cos, degrees, radians, sin, sqrt


def sub(p, q):
    return tuple(a - b for a, b in zip(p, q))


def cross(u, v):
    return (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])


def dot(u, v):
    return sum(a * b for a, b in zip(u, v))


def unit(u):
    length = sqrt(dot(u, u))
    return tuple(x / length for x in u)


def normal(pos, f):
    a, b, c = (pos[v] for v in f)
    return cross(sub(b, a), sub(c, a))


def angle(n, ref):
    return degrees(acos(max(-1.0, min(1.0, dot(n, ref) / sqrt(dot(n, n) * dot(ref, ref))))))


def edges_of(faces):
    return sorted({tuple(sorted((f[k], f[(k + 1) % 3]))) for f in faces for k in range(3)})


class Camera:
    """Turntable camera looking at `target` from azimuth `az` (degrees from +x towards +y) and elevation `el` (degrees
    above the horizon), at `distance`, without roll; perspective projection. `light` is in camera coordinates (right,
    up, towards the viewer), and `ambient` the share of the brightness that does not depend on the light."""

    def __init__(self, target, az, el, distance, light, ambient):
        az, el = radians(az), radians(el)
        self.back = (cos(el) * cos(az), cos(el) * sin(az), sin(el))  # towards the viewer
        self.right = (-sin(az), cos(az), 0.0)
        self.up = cross(self.back, self.right)
        self.eye = tuple(t + distance * b for t, b in zip(target, self.back))
        self.distance = distance
        self.light, self.ambient = light, ambient

    def view(self, p):
        """Screen coordinates (model units at the target's distance) and depth (larger is nearer) of a point."""
        q = sub(p, self.eye)
        along = -dot(q, self.back)
        return dot(q, self.right) * self.distance / along, dot(q, self.up) * self.distance / along, -along

    def brightness(self, n):
        """Lambertian shading in percent with an ambient term; faces seen from behind are shaded from their back side,
        as in Blender."""
        light = unit(tuple(self.light[0] * r + self.light[1] * u + self.light[2] * b
                           for r, u, b in zip(self.right, self.up, self.back)))
        n = unit(n)
        if dot(n, self.back) < 0:
            n = tuple(-x for x in n)
        return 100 * (self.ambient + (1 - self.ambient) * max(0.0, dot(n, light)))

    def face_depth(self, pos, face):
        return sum(self.view(pos[v])[2] for v in face) / 3

    def back_to_front(self, pos, faces):
        """The indices of the faces in painter's order, so that nearer faces cover what lies behind them."""
        return sorted(range(len(faces)), key=lambda i: self.face_depth(pos, faces[i]))

    def point_covered(self, pos, faces, point, skip=()):
        """Whether a 3D point lies behind one of the faces, other than those that contain all vertices in `skip`."""
        x, y, z = self.view(point)
        for f in faces:
            if skip and set(skip) <= set(f):
                continue
            (ax, ay, az), (bx, by, bz), (cx, cy, cz) = (self.view(pos[v]) for v in f)
            den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(den) < 1e-12:
                continue
            l1 = ((by - cy) * (x - cx) + (cx - bx) * (y - cy)) / den
            l2 = ((cy - ay) * (x - cx) + (ax - cx) * (y - cy)) / den
            if min(l1, l2, 1 - l1 - l2) > 0.01 and l1 * az + l2 * bz + (1 - l1 - l2) * cz > z:
                return True
        return False

    def edge_runs(self, pos, faces, e, hidden, samples=80):
        """The parameter intervals of edge e that are hidden (or visible) behind other faces."""
        flags = [self.point_covered(pos, faces, tuple(a + (k + 0.5) / samples * (b - a) for a, b in zip(pos[e[0]], pos[e[1]])),
                                    skip=e) == hidden for k in range(samples)]
        runs, start = [], None
        for k, flag in enumerate(flags + [False]):
            if flag and start is None:
                start = k
            elif not flag and start is not None:
                runs.append((start / samples, k / samples))
                start = None
        return runs
