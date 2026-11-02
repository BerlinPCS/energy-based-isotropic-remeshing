#pragma once

// Sec. 3.1.3: Operation validity

#include "remeshing/edge_loss.h"

#include <vector>

namespace remeshing {

Point edge_midpoint(const Mesh& mesh, Edge e);

// A collapse merges the two ends of `halfedge` at their midpoint and keeps the vertex `halfedge` points to.
struct PlannedCollapse {
	Halfedge halfedge;
	Point position;
};

// A split is possible on an edge longer than ell_max and needs no normal test.
bool split_valid(const Mesh& mesh, Edge e, double ell_max);

// The edge is shorter than ell_min, no resulting edge is longer than ell_max and the collapse is legal.
bool collapse_legal(const Mesh& mesh, Edge e, double ell_min, double ell_max, PlannedCollapse& collapse);

// The normal test of a collapse.
bool collapse_passes_normal_test(const Mesh& mesh, const PlannedCollapse& collapse);

// A legal flip with both new triangles within 60 degrees of the summed normal of the two old ones.
bool flip_valid(const Mesh& mesh, Edge e);

// The normal test of a relocation.
bool relocation_valid(const Mesh& mesh, Vertex v, const Point& position);

// The step from v to its tangential smoothing target p_v'.
pmp::vec3 tangential_step(const Mesh& mesh, Vertex v);

// The edges whose flip would connect a and b.
std::vector<Edge> flips_connecting(const Mesh& mesh, Vertex a, Vertex b);

} // namespace remeshing
