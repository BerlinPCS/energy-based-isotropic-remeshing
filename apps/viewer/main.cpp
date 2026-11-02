// Browses the results of remesher-cli.

#include "remeshing/edge_loss.h"
#include "shared/visuals.h"

#include "imgui.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <pmp/io/io.h>
#include <polyscope/polyscope.h>
#include <polyscope/view.h>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;

namespace {

const float WIREFRAME_WIDTH = 0.8f;

struct Measure {
	const char* name;
	const char* key;
	const char* format;
};

// Sec. 4.1
const Measure MEASURES[] = {
	{"Avg. loss", "average_edge_loss", "%.5f"},
	{"Min. angle", "minimum_angle", "%.2f"},
	{"5% angle", "angle_p05", "%.2f"},
	{"Valence err.", "valence_error", "%.3f"},
	{"d_RMS", "d_rms", "%.2e"},
	{"Time (ms)", "algorithm_time_ms", "%.0f"},
};

struct ResultEntry {
	fs::path folder;
	json summary;
	std::string label;
	bool five_cycle_baseline = false;
};

std::string describe(const json& summary) {
	std::string label = summary.at("approach").get<std::string>();
	if (!summary.at("objective").is_null()) label += " / " + summary.at("objective").get<std::string>();
	if (summary.at("stop_reason") == "fixed-cycles")
		label += ", k = " + std::to_string(summary.at("cycles").get<int>());
	if (summary.at("no_knee").get<bool>()) label += ", no knee";
	return label;
}

class Viewer {
public:
	explicit Viewer(const fs::path& root) {
		for (const auto& entry : fs::recursive_directory_iterator(root)) {
			if (entry.path().filename() != "summary.json") continue;
			std::ifstream input(entry.path());
			json summary = json::parse(input);
			summary.erase("d_rms");
			if (std::ifstream deviation(entry.path().parent_path() / "deviation.json"); deviation)
				summary["d_rms"] = json::parse(deviation).at("d_rms");
			const std::string mesh = fs::path(summary.at("input").get<std::string>()).stem().string();
			const double target = summary.at("target").get<double>();
			ResultEntry result{entry.path().parent_path(), summary, describe(summary)};
			result.five_cycle_baseline = result.label == "baseline, k = 5";
			results_[mesh][target].push_back(std::move(result));
		}
		for (auto& [mesh, by_target] : results_)
			for (auto& [target, results] : by_target)
				std::sort(results.begin(), results.end(),
						  [](const ResultEntry& a, const ResultEntry& b) { return a.label < b.label; });
	}

	bool empty() const { return results_.empty(); }

	void run() {
		polyscope::options::programName = "Result Viewer";
		polyscope::options::usePrefsFile = false;
		polyscope::options::groundPlaneMode = polyscope::GroundPlaneMode::ShadowOnly;
		polyscope::options::shadowBlurIters = 6;
		polyscope::options::rightGuiPaneWidth = 760; 
		polyscope::init();
		visuals::register_extra_color_maps();
		polyscope::state::userCallback = [this] { draw_panel(); };
		mesh_ = results_.begin()->first;
		target_ = results_.begin()->second.begin()->first;
		show_result(0);
		polyscope::view::resetCameraToHomeView();
		polyscope::show();
	}

private:
	std::map<std::string, std::map<double, std::vector<ResultEntry>>> results_;
	std::string mesh_;
	double target_ = 1.0;
	int selected_ = 0;
	bool show_input_ = false;
	bool show_ratios_ = true;
	std::string error_;
	visuals::LossVisuals style_;
	remeshing::Mesh shown_;
	float shown_ell_star_ = 1.0f;

	std::vector<ResultEntry>& current() { return results_.at(mesh_).at(target_); }

	void draw() { visuals::draw_mesh(shown_, shown_ell_star_, style_)->setEdgeWidth(WIREFRAME_WIDTH); }

	void show_result(int index) {
		selected_ = index;
		show_input_ = false;
		const ResultEntry& result = current().at(index);
		load(visuals::remeshed_mesh_in(result.folder), result.summary);
	}

	void show_input() {
		const ResultEntry& result = current().at(selected_);
		load(result.summary.at("input").get<std::string>(), result.summary);
	}

	void load(const fs::path& path, const json& summary) {
		try {
			remeshing::Mesh mesh;
			pmp::read(mesh, path);
			shown_ = std::move(mesh);
			shown_ell_star_ = summary.at("target_length").get<float>();
			error_.clear();
			draw();
		} catch (const std::exception& error) {
			error_ = path.string() + ": " + error.what();
		}
	}

	void draw_panel() {
		ImGui::PushItemWidth(180.0f);
		if (ImGui::BeginCombo("Mesh", mesh_.c_str())) {
			for (const auto& [mesh, by_target] : results_) {
				if (!ImGui::Selectable(mesh.c_str(), mesh == mesh_)) continue;
				mesh_ = mesh;
				target_ = by_target.begin()->first;
				show_result(0);
				polyscope::view::resetCameraToHomeView();
			}
			ImGui::EndCombo();
		}
		char target_label[32];
		snprintf(target_label, sizeof(target_label), "%g", target_);
		if (ImGui::BeginCombo("Target T", target_label)) {
			for (const auto& [target, results] : results_.at(mesh_)) {
				snprintf(target_label, sizeof(target_label), "%g", target);
				if (!ImGui::Selectable(target_label, target == target_)) continue;
				target_ = target;
				show_result(0);
			}
			ImGui::EndCombo();
		}
		ImGui::PopItemWidth();
		if (ImGui::Checkbox("Show the input mesh", &show_input_)) {
			if (show_input_)
				show_input();
			else
				show_result(selected_);
		}
		ImGui::Checkbox("Ratios to the five-cycle baseline", &show_ratios_);
		if (!error_.empty()) ImGui::TextColored(ImVec4(0.9f, 0.3f, 0.3f, 1.0f), "%s", error_.c_str());

		draw_table();

		ImGui::Separator();
		if (visuals::draw_loss_controls(style_)) draw();
		visuals::draw_camera_controls();
	}

	void draw_table() {
		const std::vector<ResultEntry>& results = current();
		const auto baseline = std::find_if(results.begin(), results.end(),
										   [](const ResultEntry& result) { return result.five_cycle_baseline; });
		const bool ratios = show_ratios_ && baseline != results.end();

		const ImGuiTableFlags flags =
			ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_ScrollY;
		if (!ImGui::BeginTable("Results", 1 + IM_ARRAYSIZE(MEASURES), flags, ImVec2(0.0f, 420.0f))) return;
		ImGui::TableSetupScrollFreeze(1, 1);
		ImGui::TableSetupColumn("Configuration");
		for (const Measure& measure : MEASURES) ImGui::TableSetupColumn(measure.name);
		ImGui::TableHeadersRow();

		for (int i = 0; i < static_cast<int>(results.size()); ++i) {
			const ResultEntry& result = results[i];
			ImGui::TableNextRow();
			ImGui::TableNextColumn();
			if (ImGui::Selectable(result.label.c_str(), i == selected_ && !show_input_,
								  ImGuiSelectableFlags_SpanAllColumns))
				show_result(i);
			for (const Measure& measure : MEASURES) {
				ImGui::TableNextColumn();
				if (result.summary.count(measure.key) == 0) continue;
				const double value = result.summary.at(measure.key).get<double>();
				if (ratios && !result.five_cycle_baseline) {
					if (baseline->summary.count(measure.key) > 0)
						ImGui::Text("%.3f", value / baseline->summary.at(measure.key).get<double>());
				} else {
					ImGui::Text(measure.format, value);
				}
			}
		}
		ImGui::EndTable();
	}
};

} // namespace

int main(int argc, char** argv) {
	if (argc != 2) {
		std::cerr << "usage: viewer RESULTS_DIR\n";
		return 2;
	}
	try {
		Viewer viewer(argv[1]);
		if (viewer.empty()) {
			std::cerr << "viewer: no summary.json below " << argv[1] << "\n";
			return 1;
		}
		viewer.run();
	} catch (const std::exception& error) {
		std::cerr << "viewer: " << error.what() << "\n";
		return 1;
	}
	return 0;
}
