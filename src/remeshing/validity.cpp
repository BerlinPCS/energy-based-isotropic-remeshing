#include "remeshing/validity.h"

#include "remeshing/parameters.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <utility>

namespace remeshing {

namespace {

Normal triangle_normal(const Point& p0, const Point& p1, const Point& p2) { return pmp::cross(p1 - p0, p2 - p0); }

// Area-weighted normal of a triangle, with the vertex `moved` at `position`.
Normal face_normal(const Mesh& mesh, Face f, Vertex moved, const Point& position) {
	std::array<Point, 3> corners;
	int i = 0;
	for (Vertex v : mesh.vertices(f)) corners[i++] = v == moved ? position : mesh.position(v);
	return pmp::cross(corners[1] - corners[0], corners[2] - corners[0]);
}

// Within 60 degrees of the reference; a triangle of (numerically) zero area never is.
bool within_cone(const Normal& reference, double reference_norm, const Normal& normal) {
	const double norm = pmp::norm(normal);
	if (!std::isfinite(reference_norm) || !std::isfinite(norm)) return false;
	if (reference_norm <= FLT_EPSILON || norm <= FLT_EPSILON * reference_norm) return false;

	const double cosine = pmp::dot(reference, normal) / (reference_norm * norm);
	return std::isfinite(cosine) && cosine > NORMAL_TEST_COSINE;
}

// Test fails if more new than old triangles lie outside 60 degrees of their summed normal, or if a new triangle 
// deviates further from it than the most deviating old one.
bool ring_passes_normal_test(const std::vector<Normal>& old_normals, const std::vector<Normal>& new_normals) {
	Normal reference(0, 0, 0);
	for (const Normal& n : old_normals) reference += n;
	const double reference_norm = pmp::norm(reference);

	int new_outside = 0;
	for (const Normal& n : new_normals)
		if (!within_cone(reference, reference_norm, n)) new_outside++;
	if (new_outside == 0) return true;
	int old_outside = 0;
	for (const Normal& n : old_normals)
		if (!within_cone(reference, reference_norm, n)) old_outside++;
	if (new_outside > old_outside) return false;

	const auto cosine = [&](const Normal& n) {
		const double norm = pmp::norm(n);
		return norm <= FLT_EPSILON * reference_norm ? 0.0 : pmp::dot(n, reference) / (norm * reference_norm);
	};
	double bound = NORMAL_TEST_COSINE;
	for (const Normal& n : old_normals) bound = std::min(bound, cosine(n));
	for (const Normal& n : new_normals)
		if (pmp::norm(n) <= FLT_EPSILON * reference_norm || cosine(n) < bound) return false;
	return true;
}

Normal vertex_normal(const Mesh& mesh, Vertex v) {
	Normal normal(0, 0, 0);
	for (Face f : mesh.faces(v)) normal += face_normal(mesh, f, Vertex(), Point());
	return pmp::normalize(normal);
}

} // namespace

Point edge_midpoint(const Mesh& mesh, Edge e) {
	return 0.5 * (mesh.position(mesh.vertex(e, 0)) + mesh.position(mesh.vertex(e, 1)));
}

bool split_valid(const Mesh& mesh, Edge e, double ell_max) {
	return !mesh.is_deleted(e) && edge_length(mesh, e) > ell_max;
}

bool collapse_legal(const Mesh& mesh, Edge e, double ell_min, double ell_max, PlannedCollapse& collapse) {
	if (mesh.is_deleted(e) || edge_length(mesh, e) >= ell_min) return false;

	collapse.halfedge = mesh.halfedge(e, 0);
	const Vertex from = mesh.from_vertex(collapse.halfedge);
	const Vertex to = mesh.to_vertex(collapse.halfedge);
	collapse.position = 0.5 * (mesh.position(from) + mesh.position(to));

	for (Vertex v : mesh.vertices(from)) {
		if (v == to) continue;
		if (pmp::distance(collapse.position, mesh.position(v)) > ell_max) return false;
	}
	for (Vertex v : mesh.vertices(to)) {
		if (v == from) continue;
		if (pmp::distance(collapse.position, mesh.position(v)) > ell_max) return false;
	}
	return mesh.is_collapse_ok(collapse.halfedge);
}

bool collapse_passes_normal_test(const Mesh& mesh, const PlannedCollapse& collapse) {
	const Vertex from = mesh.from_vertex(collapse.halfedge);
	const Vertex to = mesh.to_vertex(collapse.halfedge);
	const Face shared0 = mesh.face(collapse.halfedge);
	const Face shared1 = mesh.face(mesh.opposite_halfedge(collapse.halfedge));

	static thread_local std::vector<Normal> old_normals, new_normals; 
	old_normals.clear();
	new_normals.clear();
	for (Face f : mesh.faces(from)) {
		old_normals.push_back(face_normal(mesh, f, Vertex(), collapse.position));
		if (f != shared0 && f != shared1) new_normals.push_back(face_normal(mesh, f, from, collapse.position));
	}
	for (Face f : mesh.faces(to)) {
		if (f == shared0 || f == shared1) continue;
		old_normals.push_back(face_normal(mesh, f, Vertex(), collapse.position));
		new_normals.push_back(face_normal(mesh, f, to, collapse.position));
	}
	return ring_passes_normal_test(old_normals, new_normals);
}

bool flip_valid(const Mesh& mesh, Edge e) {
	if (mesh.is_deleted(e) || !mesh.is_flip_ok(e)) return false;

	const Point a = mesh.position(mesh.from_vertex(mesh.halfedge(e, 0)));
	const Point b = mesh.position(mesh.to_vertex(mesh.halfedge(e, 0)));
	const Point c = mesh.position(mesh.to_vertex(mesh.next_halfedge(mesh.halfedge(e, 0))));
	const Point d = mesh.position(mesh.to_vertex(mesh.next_halfedge(mesh.halfedge(e, 1))));

	const Normal reference = triangle_normal(a, b, c) + triangle_normal(b, a, d);
	const double reference_norm = pmp::norm(reference);
	return within_cone(reference, reference_norm, triangle_normal(c, d, b)) &&
		   within_cone(reference, reference_norm, triangle_normal(d, c, a));
}

bool relocation_valid(const Mesh& mesh, Vertex v, const Point& position) {
	static thread_local std::vector<Normal> old_normals, new_normals;
	old_normals.clear();
	new_normals.clear();
	for (Face f : mesh.faces(v)) {
		old_normals.push_back(face_normal(mesh, f, Vertex(), position));
		new_normals.push_back(face_normal(mesh, f, v, position));
	}
	return ring_passes_normal_test(old_normals, new_normals);
}

pmp::vec3 tangential_step(const Mesh& mesh, Vertex v) {
	Point center(0, 0, 0);
	unsigned int valence = 0;
	for (Vertex neighbor : mesh.vertices(v)) {
		center += mesh.position(neighbor);
		valence++;
	}
	if (valence == 0) return pmp::vec3(0, 0, 0);
	center /= valence;

	const pmp::vec3 direction = center - mesh.position(v);
	const Normal normal = vertex_normal(mesh, v);
	return direction - pmp::dot(direction, normal) * normal;
}

std::vector<Edge> flips_connecting(const Mesh& mesh, Vertex a, Vertex b) {
	std::vector<Edge> edges;
	for (Halfedge h : mesh.halfedges(a)) {
		const Halfedge opposite_a = mesh.next_halfedge(h);
		const Halfedge across = mesh.opposite_halfedge(opposite_a);
		if (mesh.to_vertex(mesh.next_halfedge(across)) == b) edges.push_back(mesh.edge(opposite_a));
	}
	return edges;
}

} // namespace remeshing
