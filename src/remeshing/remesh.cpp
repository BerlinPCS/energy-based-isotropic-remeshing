#include "remeshing/remesh.h"

#include "remeshing/schedules.h"

#include <chrono>
#include <stdexcept>

namespace remeshing {

std::string input_problem(const Mesh& mesh) {
	if (mesh.n_faces() == 0) return "the mesh has no faces";
	if (!mesh.is_triangle_mesh()) return "the mesh is not a triangle mesh";
	for (Vertex v : mesh.vertices()) {
		if (mesh.is_isolated(v)) return "the mesh has isolated vertices";
		if (mesh.is_boundary(v)) return "the mesh is not closed";
	}
	return "";
}

Result remesh(Mesh& mesh, const Settings& settings, std::atomic<uint64_t>* edits_done) {
	if (const std::string problem = input_problem(mesh); !problem.empty()) throw std::invalid_argument(problem);

	const auto start = std::chrono::steady_clock::now();

	// ell_* has single precision like the coordinates.
	const float ell_star = static_cast<float>(average_edge_length(mesh) * settings.target);
	Run run(mesh, settings.objective, ell_star, edits_done);
	run.epsilon = std::max<double>(EPSILON_FLOOR, EPSILON_SCALE * run.average_loss());
	run.rho = settings.no_knee ? 0.0f : KNEE_RHO;
	run.trace = settings.trace;
	run.trace_interval = settings.trace_interval;
	trace_event(run, "initial");

	Result result;
	switch (settings.approach) {
	case Approach::Baseline:
		result = run_fixed_cycles(run, CandidateOrder::Index, settings.cycles.value_or(DEFAULT_CYCLES));
		break;
	case Approach::Repeated:
		result = run_repeated_cycles(run, CandidateOrder::Index);
		break;
	case Approach::Ordered:
		result = settings.cycles ? run_fixed_cycles(run, CandidateOrder::Length, *settings.cycles)
								 : run_repeated_cycles(run, CandidateOrder::Length);
		break;
	case Approach::DynamicQueue:
		result.stop = run_queue_stage(run, StageEnd::UntilStopped);
		break;
	case Approach::Batched:
		result.stop = run_batched(run);
		break;
	case Approach::Hybrid:
		result = run_hybrid(run);
		break;
	case Approach::Alternating:
		result = run_alternating(run);
		break;
	}

	trace_event(run, "final");
	result.ell_star = run.ell_star;
	result.epsilon = run.epsilon;
	result.edits = run.edits;
	result.algorithm_time_ms =
		std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
	return result;
}

bool uses_objective(Approach approach) {
	return approach != Approach::Baseline && approach != Approach::Repeated && approach != Approach::Ordered;
}

std::string to_string(Approach approach) {
	switch (approach) {
	case Approach::Baseline:
		return "baseline";
	case Approach::Repeated:
		return "repeated";
	case Approach::Ordered:
		return "ordered";
	case Approach::DynamicQueue:
		return "dynamic-queue";
	case Approach::Batched:
		return "batched";
	case Approach::Hybrid:
		return "hybrid";
	case Approach::Alternating:
		return "alternating";
	}
	return "";
}

std::string to_string(Objective objective) {
	switch (objective) {
	case Objective::Sum:
		return "sum";
	case Objective::Maximum:
		return "maximum";
	case Objective::LocalAverage:
		return "local-average";
	case Objective::GlobalAverage:
		return "global-average";
	}
	return "";
}

std::string to_string(StopReason reason) {
	switch (reason) {
	case StopReason::FixedCycles:
		return "fixed-cycles";
	case StopReason::Stagnation:
		return "stagnation";
	case StopReason::NoCandidate:
		return "no-candidate";
	case StopReason::Safeguard:
		return "safeguard";
	case StopReason::Knee:
		return "knee";
	case StopReason::CycleCap:
		return "cycle-cap";
	case StopReason::EditCap:
		return "edit-cap";
	}
	return "";
}

} // namespace remeshing
