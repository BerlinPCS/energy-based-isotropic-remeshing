#pragma once

// Edge-loss colouring and the camera JSON.

#include "remeshing/edge_loss.h"

#include <filesystem>
#include <polyscope/surface_mesh.h>
#include <string>
#include <utility>

namespace visuals {

enum class LossColoring { Vertex, Face };

struct LossVisuals {
	std::string color_map = "coolwarm";
	std::pair<double, double> map_range = {0.0, 0.2};
	LossColoring coloring = LossColoring::Vertex;
	bool out_of_band_edges = false; // highlight edges longer than ell_max (green) and shorter than ell_min (cyan)
};

LossVisuals figure_style();

void register_extra_color_maps();

polyscope::SurfaceMesh* draw_mesh(const remeshing::Mesh& mesh, float ell_star, const LossVisuals& style);

bool draw_loss_controls(LossVisuals& style);

void draw_camera_controls();

std::filesystem::path remeshed_mesh_in(const std::filesystem::path& folder);

} // namespace visuals
