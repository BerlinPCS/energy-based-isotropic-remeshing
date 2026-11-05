#include "remeshing/evaluate.h"

#include "remeshing/parameters.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <vector>

namespace remeshing {

namespace {

// The losses of the affected edges: E^-_o before and E^+_o after the edit.
struct AffectedLosses {
	double sum_before = 0.0;
	double sum_after = 0.0;
	double max_before = 0.0;
	double max_after = 0.0;
	int count_before = 0;
	int count_after = 0;
};

// Fills gain, g_sum and d_o and returns whether the objective accepts the edit.
bool apply_objective(const Run& run, const AffectedLosses& affected, Evaluation& result) {
	result.g_sum = affected.sum_before - affected.sum_after;
	result.d_o = affected.count_after - affected.count_before;
	switch (run.objective) {
	case Objective::Sum:
		result.gain = result.g_sum;
		return result.gain > run.epsilon;
	case Objective::Maximum:
		result.gain = affected.max_before - affected.max_after;
		return result.gain > run.epsilon;
	case Objective::LocalAverage:
		result.gain = affected.sum_before / static_cast<double>(affected.count_before) -
					  affected.sum_after / static_cast<double>(affected.count_after);
		return result.gain > run.epsilon / static_cast<double>(affected.count_after);
	case Objective::GlobalAverage: {
		const int edge_count = static_cast<int>(run.mesh.n_edges());
		result.gain = global_average_gain(run.losses.energy(), edge_count, result.g_sum, result.d_o);
		return global_average_accepts(result.gain, run.epsilon, edge_count + result.d_o);
	}
	}
	return false;
}

bool may_pass_later(const Evaluation& evaluation) { return evaluation.d_o < 0 && evaluation.g_sum > 0.0; }

} // namespace

double global_average_gain(double energy, int edge_count, double g_sum, int d_o) {
	const int after_count = edge_count + d_o;
	const double before_avg = energy / static_cast<double>(edge_count);
	const double after_avg = (energy - g_sum) / static_cast<double>(after_count);
	return before_avg - after_avg;
}

bool global_average_accepts(double gain, double epsilon, int edge_count_after) {
	return gain > epsilon / static_cast<double>(edge_count_after);
}

Evaluation evaluate_split(const Run& run, Edge e, bool retain) {
	const Mesh& mesh = run.mesh;
	Evaluation result;
	if (!split_valid(mesh, e, run.ell_max)) return result;

	const double length = edge_length(mesh, e);
	const double before = run.losses[e];
	result.position = edge_midpoint(mesh, e);
	const Point opposite0 = mesh.position(mesh.to_vertex(mesh.next_halfedge(mesh.halfedge(e, 0))));
	const Point opposite1 = mesh.position(mesh.to_vertex(mesh.next_halfedge(mesh.halfedge(e, 1))));

	std::array<double, 4> new_losses;
	new_losses[0] = new_losses[1] = edge_loss(0.5 * length, run.ell_star);
	new_losses[2] = edge_loss(pmp::distance(result.position, opposite0), run.ell_star);
	new_losses[3] = edge_loss(pmp::distance(result.position, opposite1), run.ell_star);

	AffectedLosses affected;
	affected.sum_before = before;
	affected.sum_after = std::accumulate(new_losses.begin(), new_losses.end(), 0.0);
	affected.max_before = before;
	affected.max_after = *std::max_element(new_losses.begin(), new_losses.end());
	affected.count_before = 1;
	affected.count_after = 4;
	const bool accepted = apply_objective(run, affected, result);
	result.passes = accepted;
	return result;
}

Evaluation evaluate_collapse(const Run& run, Edge e, bool retain) {
	const Mesh& mesh = run.mesh;
	Evaluation result;
	PlannedCollapse collapse;
	if (!collapse_legal(mesh, e, run.ell_min, run.ell_max, collapse)) return result;

	const Vertex from = mesh.from_vertex(collapse.halfedge);
	const Vertex to = mesh.to_vertex(collapse.halfedge);

	AffectedLosses affected;
	affected.sum_before = affected.max_before = run.losses[e];
	affected.count_before = 1;
	static thread_local std::vector<Vertex> neighbors;
	neighbors.clear();
	for (Halfedge h : mesh.halfedges(from)) {
		const Vertex v = mesh.to_vertex(h);
		if (v == to) continue;
		neighbors.push_back(v);
		const double loss = run.losses[mesh.edge(h)];
		affected.sum_before += loss;
		affected.max_before = std::max(affected.max_before, loss);
		affected.count_before++;
	}
	for (Halfedge h : mesh.halfedges(to)) {
		const Vertex v = mesh.to_vertex(h);
		if (v == from) continue;
		const double loss = run.losses[mesh.edge(h)];
		affected.sum_before += loss;
		affected.max_before = std::max(affected.max_before, loss);
		affected.count_before++;
		if (std::find(neighbors.begin(), neighbors.end(), v) == neighbors.end()) neighbors.push_back(v);
	}
	for (Vertex v : neighbors) {
		const double loss = edge_loss(pmp::distance(collapse.position, mesh.position(v)), run.ell_star);
		affected.sum_after += loss;
		affected.max_after = std::max(affected.max_after, loss);
	}
	affected.count_after = static_cast<int>(neighbors.size());
	result.position = collapse.position;
	result.collapse = collapse.halfedge;
	const bool accepted = apply_objective(run, affected, result);
]	result.passes =
		(accepted || (retain && may_pass_later(result))) && collapse_passes_normal_test(mesh, collapse);
	return result;
}

Evaluation evaluate_flip(const Run& run, Edge e) {
	const Mesh& mesh = run.mesh;
	Evaluation result;
	if (mesh.is_deleted(e)) return result;

	const Vertex opposite0 = mesh.to_vertex(mesh.next_halfedge(mesh.halfedge(e, 0)));
	const Vertex opposite1 = mesh.to_vertex(mesh.next_halfedge(mesh.halfedge(e, 1)));

	AffectedLosses affected;
	affected.sum_before = affected.max_before = run.losses[e];
	affected.sum_after = affected.max_after =
		edge_loss(pmp::distance(mesh.position(opposite0), mesh.position(opposite1)), run.ell_star);
	affected.count_before = affected.count_after = 1;
	result.passes = apply_objective(run, affected, result) && flip_valid(mesh, e);
	return result;
}

Evaluation evaluate_relocation(const Run& run, Vertex v) {
	const Mesh& mesh = run.mesh;
	if (mesh.is_deleted(v)) return {};

	const pmp::vec3 step = tangential_step(mesh, v);
	if (pmp::norm(step) == 0.0) return {};

	AffectedLosses affected;
	static thread_local std::vector<Point> neighbor_positions;
	neighbor_positions.clear();
	for (Halfedge h : mesh.halfedges(v)) {
		const double loss = run.losses[mesh.edge(h)];
		neighbor_positions.push_back(mesh.position(mesh.to_vertex(h)));
		affected.sum_before += loss;
		affected.max_before = std::max(affected.max_before, loss);
	}
	affected.count_before = affected.count_after = static_cast<int>(neighbor_positions.size());

	double scale = 1.0;
	for (int i = 0; i < RELOCATION_STEP_SIZES; ++i, scale *= 0.5) {
		Evaluation result;
		const Point position = mesh.position(v) + step * scale;
		// A small step can round back to the current position of a float coordinate.
		if (pmp::norm(position - mesh.position(v)) == 0.0) continue;
		affected.sum_after = 0.0;
		affected.max_after = 0.0;
		for (const Point& neighbor : neighbor_positions) {
			const double loss = edge_loss(pmp::distance(position, neighbor), run.ell_star);
			affected.sum_after += loss;
			affected.max_after = std::max(affected.max_after, loss);
		}
		result.position = position;
		if (!apply_objective(run, affected, result) || !relocation_valid(mesh, v, position)) continue;
		result.passes = true;
		return result;
	}
	return {};
}

} // namespace remeshing
