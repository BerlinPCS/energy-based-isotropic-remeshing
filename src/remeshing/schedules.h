#pragma once

// The schedules of Sec. 3.2 and what they share.

#include "remeshing/edits.h"
#include "remeshing/parameters.h"

#include <algorithm>
#include <cstdint>

namespace remeshing {

// cycles.cpp, Sec. 3.2.1
enum class CandidateOrder { Index, Length };
void run_cycle(Run& run, CandidateOrder order);
Result run_fixed_cycles(Run& run, CandidateOrder order, int cycles);
Result run_convergent_cycles(Run& run, CandidateOrder order);

// dynamic_queue.cpp, Sec. 3.2.2, Algorithm 3.1. A stage runs until it stops, or also ends at the knee.
enum class StageEnd { UntilStopped, AtKnee };
StopReason run_queue_stage(Run& run, StageEnd end);

// batched.cpp, Sec. 3.2.2, Algorithm 3.2
StopReason run_batched(Run& run);

// combined.cpp, Sec. 3.2.3
Result run_hybrid(Run& run);
Result run_alternating(Run& run); // Algorithm 3.3

// Sec. 4.1: the practical limit of a queue stage with `elements` = |E| + |V| at its start.
inline uint64_t edit_cap(uint64_t elements) {
	return std::max<uint64_t>(EDIT_CAP_MIN, elements * static_cast<uint64_t>(EDIT_CAP_FACTOR));
}

// Sec. 3.2.2: the safeguard of the local average. It stops a stage once a window of W = |E| + |V| executed edits has
// lowered the average loss by less than delta_q of its value at the start of the window.
class LocalAverageSafeguard {
public:
	explicit LocalAverageSafeguard(const Run& run)
		: window_(std::max<uint64_t>(1, run.mesh.n_edges() + run.mesh.n_vertices())),
		  window_start_loss_(run.average_loss()) {}

	bool stops(const Run& run, uint64_t executed) {
		if (run.objective != Objective::LocalAverage || executed - window_start_ < window_) return false;
		const double loss = run.average_loss();
		const double improvement = window_start_loss_ - loss;
		const double required_improvement = DELTA_Q * window_start_loss_;
		if (improvement < required_improvement) return true;
		window_start_loss_ = loss;
		window_start_ = executed;
		return false;
	}

private:
	uint64_t window_;
	uint64_t window_start_ = 0;
	double window_start_loss_;
};

} // namespace remeshing
