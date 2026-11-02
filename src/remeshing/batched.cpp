// Sec. 3.2.2: batched processing (Algorithm 3.2).

#include "remeshing/evaluate.h"
#include "remeshing/schedules.h"

#include <vector>

namespace remeshing {

namespace {

struct Candidate {
	double gain;
	Operation operation;
	int element; // index of the edge or vertex
};

Evaluation evaluate(const Run& run, Operation operation, int element) {
	switch (operation) {
	case Operation::Split:
		return evaluate_split(run, Edge(element));
	case Operation::Collapse:
		return evaluate_collapse(run, Edge(element));
	case Operation::Flip:
		return evaluate_flip(run, Edge(element));
	case Operation::Relocation:
		return evaluate_relocation(run, Vertex(element));
	}
	return {};
}

void apply(Run& run, const Candidate& candidate, const Evaluation& evaluation) {
	switch (candidate.operation) {
	case Operation::Split:
		split_edge(run, Edge(candidate.element), evaluation.position);
		break;
	case Operation::Collapse:
		collapse_edge(run, evaluation.collapse, evaluation.position);
		break;
	case Operation::Flip:
		flip_edge(run, Edge(candidate.element));
		break;
	case Operation::Relocation:
		relocate_vertex(run, Vertex(candidate.element), evaluation.position);
		break;
	}
}

// The accepted candidates of the current mesh, sorted by decreasing gain.
std::vector<Candidate> evaluate_all(const Run& run) {
	std::vector<Candidate> candidates;
	candidates.reserve(2 * run.mesh.n_edges() + run.mesh.n_vertices());
	const auto consider = [&](Operation operation, int element) {
		const Evaluation evaluation = evaluate(run, operation, element);
		if (evaluation.passes) candidates.push_back({evaluation.gain, operation, element});
	};
	for (Edge e : run.mesh.edges()) {
		consider(Operation::Split, e.idx());
		consider(Operation::Collapse, e.idx());
		consider(Operation::Flip, e.idx());
	}
	for (Vertex v : run.mesh.vertices()) consider(Operation::Relocation, v.idx());
	std::sort(candidates.begin(), candidates.end(),
			  [](const Candidate& lhs, const Candidate& rhs) { return lhs.gain > rhs.gain; });
	return candidates;
}

} // namespace

StopReason run_batched(Run& run) {
	const uint64_t cap = edit_cap(run.mesh.n_edges() + run.mesh.n_vertices());
	LocalAverageSafeguard safeguard(run);
	uint64_t executed = 0;
	StopReason reason = StopReason::NoCandidate;

	bool stopped = false;
	while (!stopped) {
		if (executed >= cap) {
			reason = StopReason::EditCap;
			break;
		}

		const std::vector<Candidate> candidates = evaluate_all(run);
		if (candidates.empty()) break;

		for (const Candidate& candidate : candidates) {
			if (executed >= cap) break;
			const Evaluation evaluation = evaluate(run, candidate.operation, candidate.element);
			if (!evaluation.passes) continue;
			apply(run, candidate, evaluation);
			executed++;
			if (safeguard.stops(run, executed)) {
				reason = StopReason::Safeguard;
				stopped = true;
				break;
			}
		}
	}
	run.collect_garbage();
	return reason;
}

} // namespace remeshing
