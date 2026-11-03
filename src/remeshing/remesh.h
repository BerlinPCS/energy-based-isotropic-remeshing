#pragma once

#include <atomic>
#include <cstdint>
#include <optional>
#include <ostream>
#include <pmp/surface_mesh.h>
#include <string>

namespace remeshing {

// Sec. 3.2: phase-based (baseline, repeated, ordered), queue-based (dynamic queue, batched), combined (hybrid,
// alternating).
enum class Approach { Baseline, Repeated, Ordered, DynamicQueue, Batched, Hybrid, Alternating };

// Sec. 3.2.2: the objectives of the queue-based and combined approaches.
enum class Objective { Sum, Maximum, LocalAverage, GlobalAverage };

enum class StopReason {
	FixedCycles, // Sec. 3.1.4: the k cycles have run
	Stagnation,	 // Sec. 3.1.4: a cycle lowered the average loss by less than delta_c
	NoCandidate, // Sec. 3.1.4: no valid candidate passes the acceptance test
	Safeguard,	 // Sec. 3.2.2: the local average's window of W edits lowered the average loss by less than delta_q
	Knee,		 // Sec. 3.2.3: the end of a queue stage within the alternating schedule, never the end of a run
	CycleCap,	 // Sec. 4.1: practical limit
	EditCap,	 // Sec. 4.1: practical limit
};

struct Settings {
	Approach approach = Approach::Baseline;
	Objective objective = Objective::Sum; // only read by the queue-based and combined approaches
	double target = 1.0;				  // T
	std::optional<int> cycles; // k: the baseline's cycle count (default 5); makes ordered cycles run k cycles
	bool no_knee = false;	   // alternating only: rho = 0, a queue stage ends only at a window that raises the loss
	std::ostream* trace = nullptr;	// if given, receives the trace_event rows of edits.h
	uint64_t trace_interval = 1000; // executed edits between two "edits" rows
};

struct EditCounts {
	uint64_t splits = 0;
	uint64_t collapses = 0;
	uint64_t flips = 0;
	uint64_t relocations = 0;

	uint64_t total() const { return splits + collapses + flips + relocations; }
};

struct Result {
	double ell_star = 0.0;
	double epsilon = 0.0;
	EditCounts edits;
	int cycles = 0; // cycles of a phase-based run or a hybrid, rounds of an alternating run
	StopReason stop = StopReason::NoCandidate;
	double algorithm_time_ms = 0.0;
};

// Check input mesh
std::string input_problem(const pmp::SurfaceMesh& mesh);

Result remesh(pmp::SurfaceMesh& mesh, const Settings& settings, std::atomic<uint64_t>* edits_done = nullptr);

bool uses_objective(Approach approach);
std::string to_string(Approach approach);
std::string to_string(Objective objective);
std::string to_string(StopReason reason);
inline constexpr Approach APPROACHES[] = {Approach::Baseline,	  Approach::Repeated, Approach::Ordered,
										  Approach::DynamicQueue, Approach::Batched,  Approach::Hybrid,
										  Approach::Alternating};
inline constexpr Objective OBJECTIVES[] = {Objective::Sum, Objective::Maximum, Objective::LocalAverage,
										   Objective::GlobalAverage};

} // namespace remeshing
