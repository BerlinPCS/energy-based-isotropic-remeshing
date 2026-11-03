// Runs one approach on one mesh and writes the remeshed mesh and a summary.json with the measures of Sec. 4.1.

#include "args/args.hxx"
#include "remeshing/metrics.h"
#include "remeshing/parameters.h"
#include "remeshing/remesh.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <pmp/io/io.h>

namespace fs = std::filesystem;
using namespace remeshing;

namespace {

template <typename Enum, size_t N> Enum parse(const std::string& name, const Enum (&values)[N], const char* flag) {
	for (Enum value : values)
		if (to_string(value) == name) return value;
	throw std::invalid_argument("unknown value for " + std::string(flag) + ": " + name);
}

int run(int argc, char** argv) {
	args::ArgumentParser parser("Run one approach of the thesis on one mesh.",
								"Writes remeshed.<extension of INPUT> and summary.json to OUTPUT_DIR.");
	args::HelpFlag help(parser, "help", "Show this help message", {'h', "help"});
	args::Positional<std::string> input_arg(parser, "INPUT", "Closed triangle mesh (OBJ, OFF, STL or PMP)");
	args::Positional<std::string> output_arg(parser, "OUTPUT_DIR", "Directory for the results");
	args::ValueFlag<std::string> approach_arg(
		parser, "NAME", "baseline, repeated, ordered, dynamic-queue, batched, hybrid or alternating", {"approach"});
	args::ValueFlag<std::string> objective_arg(
		parser, "NAME", "sum, maximum, local-average or global-average", {"objective"});
	args::ValueFlag<double> target_arg(parser, "T", "Target edge length as a multiple of the input's average edge",
									   {"target"});
	args::ValueFlag<int> cycles_arg(parser, "k", "Cycle count of baseline (default 5) and of fixed-count ordered",
									{"cycles"});
	args::Flag no_knee_arg(parser, "no-knee", "Alternating with rho = 0", {"no-knee"});
	args::ValueFlag<std::string> trace_arg(parser, "FILE", "Write a CSV row per cycle, stage and trace interval",
										   {"trace"});
	args::ValueFlag<uint64_t> trace_interval_arg(parser, "N", "Executed edits between two trace rows (default 1000)",
												 {"trace-interval"});

	try {
		parser.ParseCLI(argc, argv);
	} catch (const args::Help&) {
		std::cout << parser;
		return 0;
	} catch (const args::Error& error) {
		std::cerr << error.what() << "\n" << parser;
		return 2;
	}
	if (!input_arg || !output_arg || !approach_arg || !target_arg) {
		std::cerr << "INPUT, OUTPUT_DIR, --approach and --target are required.\n" << parser;
		return 2;
	}

	Settings settings;
	settings.approach = parse(args::get(approach_arg), APPROACHES, "--approach");
	settings.target = args::get(target_arg);
	if (settings.target <= 0.0) throw std::invalid_argument("--target must be positive");

	if (uses_objective(settings.approach)) {
		if (!objective_arg) throw std::invalid_argument(to_string(settings.approach) + " needs --objective");
		settings.objective = parse(args::get(objective_arg), OBJECTIVES, "--objective");
	} else if (objective_arg) {
		throw std::invalid_argument(to_string(settings.approach) + " has no objective");
	}

	if (cycles_arg) {
		if (settings.approach != Approach::Baseline && settings.approach != Approach::Ordered)
			throw std::invalid_argument("--cycles applies to baseline and ordered");
		if (args::get(cycles_arg) < 1) throw std::invalid_argument("--cycles must be at least 1");
		settings.cycles = args::get(cycles_arg);
	}
	if (no_knee_arg) {
		if (settings.approach != Approach::Alternating) throw std::invalid_argument("--no-knee applies to alternating");
		settings.no_knee = true;
	}
	if (trace_interval_arg) {
		if (!trace_arg) throw std::invalid_argument("--trace-interval needs --trace");
		if (args::get(trace_interval_arg) < 1) throw std::invalid_argument("--trace-interval must be at least 1");
		settings.trace_interval = args::get(trace_interval_arg);
	}
	std::ofstream trace_file;
	if (trace_arg) {
		trace_file.open(args::get(trace_arg));
		if (!trace_file) throw std::invalid_argument("cannot write " + args::get(trace_arg));
		trace_file << "event,edits,average_edge_loss,edges,vertices\n";
		settings.trace = &trace_file;
	}

	const fs::path input_path = args::get(input_arg);
	const fs::path output_dir = args::get(output_arg);
	Mesh mesh;
	pmp::read(mesh, input_path);
	if (const std::string problem = input_problem(mesh); !problem.empty())
		throw std::invalid_argument(input_path.string() + ": " + problem);

	const Result result = remesh(mesh, settings);

	fs::create_directories(output_dir);
	pmp::write(mesh, output_dir / ("remeshed" + input_path.extension().string()));

	const Metrics metrics = measure(mesh, result.ell_star);
	const nlohmann::json summary = {
		{"input", fs::absolute(input_path).lexically_normal().string()},
		{"approach", to_string(settings.approach)},
		{"objective",
		 uses_objective(settings.approach) ? nlohmann::json(to_string(settings.objective)) : nlohmann::json()},
		{"target", settings.target},
		{"target_length", result.ell_star},
		{"epsilon", result.epsilon},
		{"cycles", result.cycles},
		{"no_knee", settings.no_knee},
		{"stop_reason", to_string(result.stop)},
		{"average_edge_loss", metrics.average_edge_loss},
		{"minimum_angle", metrics.minimum_angle},
		{"angle_p05", metrics.angle_p05},
		{"faces_below_1_degree", metrics.faces_below_1_degree},
		{"faces_below_0_001_degree", metrics.faces_below_0_001_degree},
		{"valence_error", metrics.valence_error},
		{"algorithm_time_ms", result.algorithm_time_ms},
		{"splits", result.edits.splits},
		{"collapses", result.edits.collapses},
		{"flips", result.edits.flips},
		{"relocations", result.edits.relocations},
		{"vertices", mesh.n_vertices()},
		{"edges", mesh.n_edges()},
		{"faces", mesh.n_faces()},
	};
	std::ofstream(output_dir / "summary.json") << summary.dump(2) << "\n";

	std::cout << to_string(settings.approach) << ": " << result.edits.total() << " edits, average edge loss "
			  << metrics.average_edge_loss << ", " << result.algorithm_time_ms << " ms, " << to_string(result.stop)
			  << "\n";
	return 0;
}

} // namespace

int main(int argc, char** argv) {
	try {
		return run(argc, argv);
	} catch (const std::exception& error) {
		std::cerr << "remesher-cli: " << error.what() << "\n";
		return 1;
	}
}
