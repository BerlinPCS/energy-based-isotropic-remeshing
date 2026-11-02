#pragma once

// Sec. 3.1.3: the four base operations, applied to the state of one remeshing run.

#include "remeshing/edge_loss.h"
#include "remeshing/remesh.h"

#include <atomic>
#include <cstdint>

namespace remeshing {

// The state all schedules share.
struct Run {
	Run(Mesh& mesh, Objective objective, double ell_star, std::atomic<uint64_t>* edits_done);

	Mesh& mesh;
	Objective objective;
	double ell_star;
	double ell_max;
	double ell_min;
	double epsilon = 0.0; // Sec. 3.2.2: fixed from the input's average edge loss before the first edit
	EdgeLosses losses;
	EditCounts edits;
	std::atomic<uint64_t>* edits_done; // live count for a user interface, may be null
	float rho = 0.0f;				   // Sec. 3.2.3: the knee rule's rho
	std::ostream* trace = nullptr;	   // may be null, see trace_event
	uint64_t trace_interval = 1000;

	// lambda-bar, the average edge loss of the mesh
	double average_loss() const;

	// Removes deleted elements, which renumbers the rest.
	void collect_garbage();
};

// Each returns the vertex the edit created or kept where the caller needs it.
Vertex split_edge(Run& run, Edge e, const Point& position);
Vertex collapse_edge(Run& run, Halfedge h, const Point& position);
void flip_edge(Run& run, Edge e);
void relocate_vertex(Run& run, Vertex v, const Point& position);

// Create a trace of the current mesh state. Not called by core dataset runs.
void trace_event(const Run& run, const char* event);

} // namespace remeshing
