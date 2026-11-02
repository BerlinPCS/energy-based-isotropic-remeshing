#include "shared/visuals.h"

#include "remeshing/parameters.h"

#include "imgui.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <polyscope/curve_network.h>
#include <polyscope/render/color_maps.h>
#include <polyscope/render/engine.h>
#include <polyscope/surface_scalar_quantity.h>
#include <polyscope/view.h>
#include <stdexcept>
#include <vector>

namespace visuals {

using namespace remeshing;

namespace {

const char* const LONG_EDGES_NAME = "Edges longer than ell_max";
const char* const SHORT_EDGES_NAME = "Edges shorter than ell_min";
// Hues that magma_r does not use.
const glm::vec3 LONG_EDGES_COLOR{0.10f, 0.75f, 0.20f};
const glm::vec3 SHORT_EDGES_COLOR{0.00f, 0.80f, 0.95f};
const float OUT_OF_BAND_RADIUS = 0.0005f; 

glm::vec3 to_glm(const Point& p) { return {p[0], p[1], p[2]}; }

template <typename Selected>
void draw_edge_band(const Mesh& mesh, double ell_star, const char* name, Selected&& selected, glm::vec3 color) {
	std::vector<glm::vec3> nodes;
	std::vector<std::array<size_t, 2>> edges;
	for (Edge e : mesh.edges()) {
		if (!selected(edge_length(mesh, e) / ell_star)) continue;
		nodes.push_back(to_glm(mesh.position(mesh.vertex(e, 0))));
		nodes.push_back(to_glm(mesh.position(mesh.vertex(e, 1))));
		edges.push_back({nodes.size() - 2, nodes.size() - 1});
	}
	if (edges.empty()) {
		polyscope::removeCurveNetwork(name, false);
		return;
	}
	polyscope::registerCurveNetwork(name, nodes, edges)->setColor(color)->setRadius(OUT_OF_BAND_RADIUS, true);
}

} // namespace

LossVisuals figure_style() {
	LossVisuals style;
	style.color_map = "magma_r";
	style.map_range = {0.0, 0.12};
	style.coloring = LossColoring::Face;
	style.out_of_band_edges = true;
	return style;
}

void register_extra_color_maps() {
	auto& maps = polyscope::render::engine->colorMaps;
	if (std::any_of(maps.begin(), maps.end(), [](const auto& map) { return map->name == "magma_r"; })) return;
	auto reversed = std::make_unique<polyscope::render::ValueColorMap>();
	reversed->name = "magma_r";
	reversed->values = polyscope::render::engine->getColorMap("magma").values;
	std::reverse(reversed->values.begin(), reversed->values.end());
	maps.push_back(std::move(reversed));
}

polyscope::SurfaceMesh* draw_mesh(const Mesh& mesh, float ell_star, const LossVisuals& style) {
	std::vector<glm::vec3> points;
	std::vector<double> vertex_losses;
	std::vector<size_t> point_index(mesh.vertices_size());
	points.reserve(mesh.n_vertices());
	vertex_losses.reserve(mesh.n_vertices());
	for (Vertex v : mesh.vertices()) {
		point_index[v.idx()] = points.size();
		points.push_back(to_glm(mesh.position(v)));
		double sum = 0.0;
		for (Edge e : mesh.edges(v)) sum += edge_loss(edge_length(mesh, e), ell_star);
		vertex_losses.push_back(sum / mesh.valence(v));
	}
	std::vector<std::array<size_t, 3>> faces;
	faces.reserve(mesh.n_faces());
	for (Face f : mesh.faces()) {
		std::array<size_t, 3> corners;
		size_t i = 0;
		for (Vertex v : mesh.vertices(f)) corners[i++] = point_index[v.idx()];
		faces.push_back(corners);
	}

	polyscope::SurfaceMesh* surface = polyscope::registerSurfaceMesh("Mesh", points, faces);
	polyscope::SurfaceScalarQuantity* quantity = nullptr;
	if (style.coloring == LossColoring::Vertex) {
		quantity = surface->addVertexScalarQuantity("Edge Loss", vertex_losses);
	} else {
		std::vector<double> face_losses;
		face_losses.reserve(faces.size());
		for (const auto& face : faces)
			face_losses.push_back((vertex_losses[face[0]] + vertex_losses[face[1]] + vertex_losses[face[2]]) / 3.0);
		quantity = surface->addFaceScalarQuantity("Edge Loss", face_losses);
	}
	quantity->setColorMap(style.color_map)->setMapRange(style.map_range)->setEnabled(true);

	if (style.out_of_band_edges) {
		draw_edge_band(
			mesh, ell_star, LONG_EDGES_NAME, [](double ratio) { return ratio > ELL_MAX_RATIO; }, LONG_EDGES_COLOR);
		draw_edge_band(
			mesh, ell_star, SHORT_EDGES_NAME, [](double ratio) { return ratio < ELL_MIN_RATIO; }, SHORT_EDGES_COLOR);
	} else {
		polyscope::removeCurveNetwork(LONG_EDGES_NAME, false);
		polyscope::removeCurveNetwork(SHORT_EDGES_NAME, false);
	}
	return surface;
}

bool draw_loss_controls(LossVisuals& style) {
	bool changed = false;
	ImGui::SetNextItemWidth(160.0f);
	changed |= polyscope::render::buildColormapSelector(style.color_map, "Loss Colour Map");
	float range[2] = {static_cast<float>(style.map_range.first), static_cast<float>(style.map_range.second)};
	ImGui::SetNextItemWidth(160.0f);
	if (ImGui::InputFloat2("Loss Range", range, "%.3f")) {
		style.map_range = {range[0], std::max(range[1], range[0] + 1e-6f)};
		changed = true;
	}
	const char* colorings[] = {"Per Vertex", "Per Face"};
	int coloring = static_cast<int>(style.coloring);
	ImGui::SetNextItemWidth(160.0f);
	if (ImGui::Combo("Loss Colouring", &coloring, colorings, IM_ARRAYSIZE(colorings))) {
		style.coloring = static_cast<LossColoring>(coloring);
		changed = true;
	}
	changed |= ImGui::Checkbox("Edges beyond ell_min, ell_max", &style.out_of_band_edges);
	return changed;
}

void draw_camera_controls() {
	static std::chrono::steady_clock::time_point copied_at;
	if (ImGui::Button("Copy Camera")) {
		glfwSetClipboardString(nullptr, polyscope::view::getViewAsJson().c_str());
		copied_at = std::chrono::steady_clock::now();
	}
	ImGui::SameLine();
	if (std::chrono::steady_clock::now() - copied_at < std::chrono::seconds(2))
		ImGui::TextDisabled("copied to the clipboard");
	else
		ImGui::TextDisabled("view JSON for mesh-screenshot --camera");
}

std::filesystem::path remeshed_mesh_in(const std::filesystem::path& folder) {
	for (const auto& entry : std::filesystem::directory_iterator(folder))
		if (entry.path().stem() == "remeshed") return entry.path();
	throw std::runtime_error("no remeshed mesh in " + folder.string());
}

} // namespace visuals
