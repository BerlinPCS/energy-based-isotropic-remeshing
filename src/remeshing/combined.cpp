// Sec. 3.2.3: baseline cycles combined with the dynamic queue.

#include "remeshing/schedules.h"

#include <cmath>
#include <limits>

namespace remeshing {

Result run_hybrid(Run& run) {
	for (int cycle = 0; cycle < HYBRID_CYCLES; ++cycle) run_cycle(run, CandidateOrder::Index);
	Result result;
	result.cycles = HYBRID_CYCLES;
	result.stop = run_queue_stage(run, StageEnd::UntilStopped);
	return result;
}

Result run_alternating(Run& run) {
	Result result;
	double previous = run.average_loss();
	for (int round = 0; round < MAX_CYCLES; ++round) {
		run_cycle(run, CandidateOrder::Index);
		run_queue_stage(run, StageEnd::AtKnee);
		result.cycles++;
		const double current = run.average_loss();
		const double relative_decrease =
			(previous - current) / std::max(std::abs(previous), std::numeric_limits<double>::epsilon());
		previous = current;
		if (relative_decrease < DELTA_C) break;
	}
	result.stop = run_queue_stage(run, StageEnd::UntilStopped);
	return result;
}

} // namespace remeshing
