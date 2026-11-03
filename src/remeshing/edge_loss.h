#pragma once

// Sec. 3.1.2: the per-edge loss and the cached energy of a mesh.

#include <cmath>
#include <cstddef>
#include <pmp/surface_mesh.h>

namespace remeshing {

using Mesh = pmp::SurfaceMesh;
using pmp::Edge;
using pmp::Face;
using pmp::Halfedge;
using pmp::Normal;
using pmp::Point;
using pmp::Vertex;

inline double edge_length(const Mesh& mesh, Edge e) {
	return pmp::distance(mesh.position(mesh.vertex(e, 0)), mesh.position(mesh.vertex(e, 1)));
}

inline double average_edge_length(const Mesh& mesh) {
	double sum = 0.0;
	for (Edge e : mesh.edges()) sum += edge_length(mesh, e);
	return sum / mesh.n_edges();
}

// A (numerically) zero-length edge would have an infinite loss; this finite stand-in keeps the running sums usable.
inline constexpr double ZERO_LENGTH_EDGE_LOSS = 1e6;

// lambda(e) = log2(length / ell_*)^2
inline double edge_loss(double length, double ell_star) {
	double ratio = length / ell_star;
	double loss = std::log2(ratio);
	loss = loss * loss;
	return std::isfinite(loss) ? loss : ZERO_LENGTH_EDGE_LOSS;
}

// The loss of every edge and their sum E are cached to prevent repeated recomputation, 
// since computing the loss is one of the most essential and frequent operations.
class EdgeLosses {
public:
	EdgeLosses(Mesh& mesh, double ell_star)
		: mesh_(mesh), ell_star_(ell_star), losses_(mesh.edge_property<double>("e:edge_loss", 0.0)) {
		rebuild();
	}
	~EdgeLosses() { mesh_.remove_edge_property(losses_); }
	EdgeLosses(const EdgeLosses&) = delete;
	EdgeLosses& operator=(const EdgeLosses&) = delete;

	// Recomputes the sum from scratch, which also discards its accumulated rounding error.
	void rebuild() {
		energy_ = 0.0;
		for (Edge e : mesh_.edges()) add(e);
	}
	// When copying a mesh.
	void reattach() {
		losses_ = mesh_.edge_property<double>("e:edge_loss", 0.0);
		rebuild();
	}
	void remove(Edge e) { energy_ -= losses_[e]; }
	void add(Edge e) {
		const double loss = edge_loss(edge_length(mesh_, e), ell_star_);
		losses_[e] = loss;
		energy_ += loss;
	}
	double operator[](Edge e) const { return losses_[e]; }
	double energy() const { return energy_; }

private:
	Mesh& mesh_;
	double ell_star_;
	double energy_ = 0.0;
	pmp::EdgeProperty<double> losses_;
};

} // namespace remeshing
