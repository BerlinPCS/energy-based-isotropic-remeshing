// Runs the approaches of Ch. 3 interactively. Does not save results

#include "remeshing/metrics.h"
#include "remeshing/parameters.h"
#include "remeshing/remesh.h"
#include "shared/visuals.h"

#include "imgui.h"
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <pmp/io/io.h>
#include <polyscope/polyscope.h>
#include <polyscope/view.h>
#include <portable-file-dialogs.h>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace remeshing;

namespace {

const float WIREFRAME_WIDTH = 0.8f;

template <typename Enum, size_t N> void enum_combo(const char* label, Enum& value, const Enum (&values)[N]) {
	if (!ImGui::BeginCombo(label, to_string(value).c_str())) return;
	for (Enum candidate : values)
		if (ImGui::Selectable(to_string(candidate).c_str(), candidate == value)) value = candidate;
	ImGui::EndCombo();
}

class App {
public:
	App() {
		if (fs::is_directory(MESHES_DIR))
			for (const auto& entry : fs::directory_iterator(MESHES_DIR))
				if (entry.path().extension() == ".obj") mesh_paths_.push_back(entry.path());
		std::sort(mesh_paths_.begin(), mesh_paths_.end());
	}

	void run() {
		polyscope::options::programName = "Energy-Based Isotropic Remeshing";
		polyscope::options::usePrefsFile = false;
		polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
		polyscope::options::shadowBlurIters = 6;
		polyscope::init();
		visuals::register_extra_color_maps();
		polyscope::state::userCallback = [this] { draw_panel(); };
		if (!mesh_paths_.empty()) load(mesh_paths_.front());
		polyscope::show();
		if (running_) std::_Exit(0);
		if (worker_.joinable()) worker_.join();
	}

private:
	std::vector<fs::path> mesh_paths_;
	fs::path mesh_path_;
	Mesh input_;
	Mesh shown_;
	float shown_ell_star_ = 1.0f;
	std::string error_;

	Settings settings_;
	bool ordered_fixed_cycles_ = false;
	int cycles_ = DEFAULT_CYCLES;

	std::thread worker_;
	std::atomic<bool> running_{false};
	std::atomic<bool> finished_{false};
	std::atomic<uint64_t> edits_done_{0};
	Mesh worker_mesh_;
	Result result_;
	Metrics metrics_;
	bool has_result_ = false;
	std::string worker_error_;

	visuals::LossVisuals style_;

	void load(const fs::path& path) {
		Mesh mesh;
		try {
			pmp::read(mesh, path);
		} catch (const std::exception& error) {
			error_ = error.what();
			return;
		}
		if (const std::string problem = input_problem(mesh); !problem.empty()) {
			error_ = path.filename().string() + ": " + problem;
			return;
		}
		error_.clear();
		mesh_path_ = path;
		input_ = std::move(mesh);
		show_input();
		polyscope::view::resetCameraToHomeView();
	}

	void show_input() {
		shown_ = input_;
		shown_ell_star_ = static_cast<float>(average_edge_length(input_) * settings_.target);
		has_result_ = false;
		draw();
	}

	void draw() { visuals::draw_mesh(shown_, shown_ell_star_, style_)->setEdgeWidth(WIREFRAME_WIDTH); }

	void start_remesh() {
		if (worker_.joinable()) worker_.join();
		Settings settings = settings_;
		settings.cycles.reset();
		if (settings.approach == Approach::Baseline ||
			(settings.approach == Approach::Ordered && ordered_fixed_cycles_))
			settings.cycles = cycles_;
		worker_mesh_ = input_;
		edits_done_ = 0;
		finished_ = false;
		running_ = true;
		worker_error_.clear();
		worker_ = std::thread([this, settings] {
			try {
				result_ = remesh(worker_mesh_, settings, &edits_done_);
				metrics_ = measure(worker_mesh_, result_.ell_star);
			} catch (const std::exception& error) {
				worker_error_ = error.what();
			}
			running_ = false;
			finished_ = true;
		});
	}

	void take_result() {
		finished_ = false;
		worker_.join();
		if (!worker_error_.empty()) {
			error_ = worker_error_;
			return;
		}
		shown_ = worker_mesh_;
		shown_ell_star_ = static_cast<float>(result_.ell_star);
		has_result_ = true;
		draw();
	}

	void draw_panel() {
		if (finished_) take_result();
		const bool busy = running_;

		ImGui::PushItemWidth(180.0f);
		ImGui::BeginDisabled(busy);
		if (ImGui::BeginCombo("Mesh", mesh_path_.empty() ? "" : mesh_path_.filename().c_str())) {
			for (const fs::path& path : mesh_paths_)
				if (ImGui::Selectable(path.filename().c_str(), path == mesh_path_)) load(path);
			ImGui::EndCombo();
		}
		ImGui::SameLine();
		if (ImGui::Button("Open...")) {
			const auto paths = pfd::open_file("Open a closed triangle mesh", "",
											  {"Meshes (*.obj *.off *.stl *.pmp)", "*.obj *.off *.stl *.pmp"})
								   .result();
			if (!paths.empty()) load(paths.front());
		}

		enum_combo("Approach", settings_.approach, APPROACHES);
		if (uses_objective(settings_.approach)) enum_combo("Objective", settings_.objective, OBJECTIVES);
		float target = static_cast<float>(settings_.target);
		if (ImGui::InputFloat("Target T", &target, 0.0f, 0.0f, "%.3f") && target > 0.0f) settings_.target = target;
		if (settings_.approach == Approach::Ordered) ImGui::Checkbox("Fixed cycle count", &ordered_fixed_cycles_);
		const bool fixed_cycles = settings_.approach == Approach::Baseline ||
								  (settings_.approach == Approach::Ordered && ordered_fixed_cycles_);
		if (fixed_cycles && ImGui::InputInt("Cycles k", &cycles_)) cycles_ = std::max(cycles_, 1);

		ImGui::BeginDisabled(mesh_path_.empty());
		if (ImGui::Button("Remesh")) start_remesh();
		ImGui::SameLine();
		if (ImGui::Button("Reset")) show_input();
		ImGui::EndDisabled();
		ImGui::EndDisabled();
		ImGui::PopItemWidth();

		if (busy) ImGui::Text("Remeshing... %llu edits", static_cast<unsigned long long>(edits_done_.load()));
		if (!error_.empty()) ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f), "%s", error_.c_str());

		ImGui::Separator();
		ImGui::Text("%zu vertices, %zu edges, %zu faces", shown_.n_vertices(), shown_.n_edges(), shown_.n_faces());
		if (has_result_) {
			ImGui::Text("Average edge loss: %.5f", metrics_.average_edge_loss);
			ImGui::Text("Minimum angle: %.2f", metrics_.minimum_angle);
			ImGui::Text("5th percentile angle: %.2f", metrics_.angle_p05);
			ImGui::Text("Faces below 1 degree: %zu", metrics_.faces_below_1_degree);
			ImGui::Text("Valence error: %.3f", metrics_.valence_error);
			ImGui::Text("Algorithm time: %.0f ms", result_.algorithm_time_ms);
			ImGui::Text("Edits: %llu splits, %llu collapses,", static_cast<unsigned long long>(result_.edits.splits),
						static_cast<unsigned long long>(result_.edits.collapses));
			ImGui::Text("%llu flips, %llu relocations", static_cast<unsigned long long>(result_.edits.flips),
						static_cast<unsigned long long>(result_.edits.relocations));
			ImGui::Text("Cycles or rounds: %d", result_.cycles);
			ImGui::Text("Stopped by: %s", to_string(result_.stop).c_str());
		}

		ImGui::Separator();
		if (visuals::draw_loss_controls(style_)) draw();
		visuals::draw_camera_controls();
	}
};

} // namespace

int main() {
	App app;
	app.run();
	return 0;
}
