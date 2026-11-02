#include "remeshing/metrics.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace remeshing {

namespace {

constexpr double IDEAL_VALENCE = 6.0; // closed meshes only 

// The corner angle at `apex`. atan2 stays accurate for the nearly flat triangles of the queues, where acos does not.
double angle_degrees(const Point& apex, const Point& b, const Point& c) {
	const pmp::dvec3 u(pmp::dvec3(b) - pmp::dvec3(apex));
	const pmp::dvec3 v(pmp::dvec3(c) - pmp::dvec3(apex));
	return std::atan2(pmp::norm(pmp::cross(u, v)), pmp::dot(u, v)) * 180.0 / std::acos(-1.0);
}

} // namespace

Metrics measure(const Mesh& mesh, double ell_star) {
	Metrics metrics;

	double energy = 0.0;
	for (Edge e : mesh.edges()) energy += edge_loss(edge_length(mesh, e), ell_star);
	metrics.average_edge_loss = energy / static_cast<double>(mesh.n_edges());

	std::vector<double> minimum_angles;
	minimum_angles.reserve(mesh.n_faces());
	for (Face f : mesh.faces()) {
		std::array<Point, 3> p;
		int i = 0;
		for (Vertex v : mesh.vertices(f)) p[i++] = mesh.position(v);
		minimum_angles.push_back(std::min(
			{angle_degrees(p[0], p[1], p[2]), angle_degrees(p[1], p[0], p[2]), angle_degrees(p[2], p[0], p[1])}));
	}
	std::sort(minimum_angles.begin(), minimum_angles.end());
	metrics.minimum_angle = minimum_angles.front();
	metrics.angle_p05 =
		minimum_angles[static_cast<size_t>(std::floor(0.05 * static_cast<double>(minimum_angles.size() - 1)))];
	const auto count_below = [&](double degrees) {
		return static_cast<size_t>(std::lower_bound(minimum_angles.begin(), minimum_angles.end(), degrees) -
								   minimum_angles.begin());
	};
	metrics.faces_below_1_degree = count_below(1.0);
	metrics.faces_below_0_001_degree = count_below(0.001);

	double squared_error = 0.0;
	for (Vertex v : mesh.vertices()) {
		const double error = static_cast<double>(mesh.valence(v)) - IDEAL_VALENCE;
		squared_error += error * error;
	}
	metrics.valence_error = std::sqrt(squared_error / static_cast<double>(mesh.n_vertices()));
	return metrics;
}

} // namespace remeshing
