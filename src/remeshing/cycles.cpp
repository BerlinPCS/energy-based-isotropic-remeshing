// Sec. 3.2.1: the cycles of Botsch and Kobbelt, in index order (baseline, repeated) or with split and collapse
// candidates sorted by length (ordered).

#include "remeshing/schedules.h"
#include "remeshing/validity.h"

#include <cfloat>
#include <cmath>
#include <limits>
#include <vector>

namespace remeshing {

namespace {

struct EdgeByLength {
	Edge edge;
	double length;
};

// Floating point issues
bool split_would_create_zero_length_edge(const Mesh& mesh, Edge e) {
	const Point midpoint = edge_midpoint(mesh, e);
	const Vertex opposite0 = mesh.to_vertex(mesh.next_halfedge(mesh.halfedge(e, 0)));
	const Vertex opposite1 = mesh.to_vertex(mesh.next_halfedge(mesh.halfedge(e, 1)));
	return pmp::distance(mesh.position(opposite0), midpoint) < FLT_EPSILON ||
		   pmp::distance(mesh.position(opposite1), midpoint) < FLT_EPSILON;
}

void split_phase(Run& run, CandidateOrder order) {
	std::vector<EdgeByLength> candidates;
	candidates.reserve(run.mesh.n_edges() / 4);
	for (Edge e : run.mesh.edges()) {
		const double length = edge_length(run.mesh, e);
		if (length > run.ell_max && !split_would_create_zero_length_edge(run.mesh, e))
			candidates.push_back({e, length});
	}
	if (order == CandidateOrder::Length)
		std::sort(candidates.begin(), candidates.end(), [](const EdgeByLength& lhs, const EdgeByLength& rhs) {
			if (lhs.length != rhs.length) return lhs.length > rhs.length;
			return lhs.edge.idx() < rhs.edge.idx();
		});

	for (const EdgeByLength& candidate : candidates)
		split_edge(run, candidate.edge, edge_midpoint(run.mesh, candidate.edge));
}

void collapse_phase(Run& run, CandidateOrder order) {
	std::vector<EdgeByLength> candidates;
	candidates.reserve(run.mesh.n_edges() / 4);
	for (Edge e : run.mesh.edges()) {
		const double length = edge_length(run.mesh, e);
		if (length < run.ell_min) candidates.push_back({e, length});
	}
	if (order == CandidateOrder::Length)
		std::sort(candidates.begin(), candidates.end(), [](const EdgeByLength& lhs, const EdgeByLength& rhs) {
			if (lhs.length != rhs.length) return lhs.length < rhs.length;
			return lhs.edge.idx() < rhs.edge.idx();
		});

	for (const EdgeByLength& candidate : candidates) {
		PlannedCollapse collapse;
		if (collapse_legal(run.mesh, candidate.edge, run.ell_min, run.ell_max, collapse) &&
			collapse_passes_normal_test(run.mesh, collapse))
			collapse_edge(run, collapse.halfedge, collapse.position);
	}
}

int valence_deviation(int valence) { return (valence - 6) * (valence - 6); }

// A flip takes one edge from each end of the edge and gives one to each opposite vertex.
bool flip_lowers_valence_deviation(int v0, int v1, int w0, int w1) {
	const int before = valence_deviation(v0) + valence_deviation(v1) + valence_deviation(w0) + valence_deviation(w1);
	const int after =
		valence_deviation(v0 - 1) + valence_deviation(v1 - 1) + valence_deviation(w0 + 1) + valence_deviation(w1 + 1);
	return after < before;
}

void flip_phase(Run& run) {
	Mesh& mesh = run.mesh;
	std::vector<int> valences(mesh.vertices_size(), 0);
	for (Vertex v : mesh.vertices()) valences[v.idx()] = mesh.valence(v);

	for (Edge e : mesh.edges()) {
		const int v0 = mesh.vertex(e, 0).idx();
		const int v1 = mesh.vertex(e, 1).idx();
		const int w0 = mesh.to_vertex(mesh.next_halfedge(mesh.halfedge(e, 0))).idx();
		const int w1 = mesh.to_vertex(mesh.next_halfedge(mesh.halfedge(e, 1))).idx();
		if (!flip_lowers_valence_deviation(valences[v0], valences[v1], valences[w0], valences[w1]) ||
			!flip_valid(mesh, e))
			continue;
		flip_edge(run, e);
		valences[v0]--;
		valences[v1]--;
		valences[w0]++;
		valences[w1]++;
	}
}

// Each vertex takes the largest valid step towards its target, and later vertices see the new positions.
void relocation_phase(Run& run) {
	for (Vertex v : run.mesh.vertices()) {
		const pmp::vec3 step = tangential_step(run.mesh, v);
		if (pmp::norm(step) == 0.0) continue;

		double scale = 1.0;
		for (int i = 0; i < RELOCATION_STEP_SIZES; ++i, scale *= 0.5) {
			const pmp::vec3 scaled_step = step * scale;
			if (!relocation_valid(run.mesh, v, run.mesh.position(v) + scaled_step)) continue;
			const Point position = run.mesh.position(v) + scaled_step;
			relocate_vertex(run, v, position);
			break;
		}
	}
}

} // namespace

void run_cycle(Run& run, CandidateOrder order) {
	split_phase(run, order);
	collapse_phase(run, order);
	flip_phase(run);
	relocation_phase(run);
	run.collect_garbage();
	trace_event(run, "cycle");
}

Result run_fixed_cycles(Run& run, CandidateOrder order, int cycles) {
	for (int cycle = 0; cycle < cycles; ++cycle) run_cycle(run, order);
	Result result;
	result.cycles = cycles;
	result.stop = StopReason::FixedCycles;
	return result;
}

Result run_repeated_cycles(Run& run, CandidateOrder order) {
	Result result;
	result.stop = StopReason::CycleCap;
	double previous = run.average_loss();
	for (int cycle = 0; cycle < MAX_CYCLES; ++cycle) {
		const Checkpoint before(run);
		run_cycle(run, order);
		result.cycles++;
		const double current = run.average_loss();
		const double relative_decrease =
			(previous - current) / std::max(std::abs(previous), std::numeric_limits<double>::epsilon());
		if (relative_decrease < DELTA_C) {
			result.stop = StopReason::Stagnation;
			if (current > previous) {
				before.restore(run);
				result.cycles--;
			}
			break;
		}
		previous = current;
	}
	return result;
}

} // namespace remeshing
