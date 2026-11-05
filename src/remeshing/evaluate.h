#pragma once

// Sec. 3.2.2: Evaluate checks a candidate's validity on the current mesh and computes its gain under the objective.

#include "remeshing/edits.h"
#include "remeshing/validity.h"

namespace remeshing {

// The four candidate types of C(M). Flips are evaluated by edge loss here; the valence flips of the cycles are not
// candidates of a queue (see cycles.cpp).
enum class Operation { Split, Collapse, Flip, Relocation };

struct Evaluation {
	bool passes = false; // valid and accepted by the objective
	double gain = 0.0;	 // g(o) under the run's objective
	double g_sum = 0.0;	 // decrease of the total loss
	int d_o = 0;		 // change of the edge count
	Point position;		 // the new vertex of a split, the merged vertex of a collapse, the target of a relocation
	Halfedge collapse;	 // the halfedge a collapse removes
};

// g_gavg(o) = lambda-bar(M) - lambda-bar(o(M)) as a function of the current energy and edge count.
double global_average_gain(double energy, int edge_count, double g_sum, int d_o);

// Whether a global-average gain passes its threshold, epsilon / |A|.
bool global_average_accepts(double gain, double epsilon, int edge_count_after);

// `retain` is set only by the dynamic queue under the global average (see dynamic_queue.cpp): `passes` then also holds
// for valid collapses that do not pass yet, but may once the average loss has dropped.
Evaluation evaluate_split(const Run& run, Edge e, bool retain = false);
Evaluation evaluate_collapse(const Run& run, Edge e, bool retain = false);
Evaluation evaluate_flip(const Run& run, Edge e);
Evaluation evaluate_relocation(const Run& run, Vertex v);

} // namespace remeshing
