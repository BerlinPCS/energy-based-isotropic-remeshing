// Renders one mesh with the edge-loss colouring and camera of the viewers to a PNG.

#include "args/args.hxx"
#include "remeshing/edge_loss.h"
#include "shared/visuals.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <pmp/io/io.h>
#include <polyscope/polyscope.h>
#include <polyscope/screenshot.h>
#include <polyscope/view.h>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

std::vector<float> parse_floats(const std::string& text, const char* what) {
	std::vector<float> values;
	std::stringstream stream(text);
	std::string item;
	while (std::getline(stream, item, ',')) values.push_back(std::stof(item));
	if (values.empty()) throw std::invalid_argument(std::string("empty ") + what);
	return values;
}

std::string read_text(const fs::path& path) {
	std::ifstream input(path);
	if (!input) throw std::runtime_error("Could not open " + path.string());
	std::stringstream buffer;
	buffer << input.rdbuf();
	return buffer.str();
}

int run(int argc, char** argv) {
	args::ArgumentParser parser(
		"Render a mesh with the edge-loss colouring to a PNG.",
		"INPUT is a mesh file or a result folder of remesher-cli, whose target length is used. For a mesh file, give "
		"--target or --target-length. --camera takes the JSON of the viewers' Copy Camera button, inline or as a "
		"file.");
	args::HelpFlag help(parser, "help", "Show this help message", {'h', "help"});
	args::Positional<std::string> input_arg(parser, "INPUT", "Mesh file or result folder");
	args::Positional<std::string> output_arg(parser, "OUTPUT", "PNG file to write");
	args::ValueFlag<double> target_arg(parser, "T", "Target as a multiple of the mesh's average edge length",
									   {"target"});
	args::ValueFlag<double> target_length_arg(parser, "VALUE", "Absolute target edge length", {"target-length"});
	args::ValueFlag<std::string> camera_arg(parser, "JSON|FILE", "Polyscope view JSON (Copy Camera)", {"camera"});
	args::ValueFlag<std::string> look_at_arg(parser, "ex,ey,ez,tx,ty,tz[,ux,uy,uz]", "Camera position and target",
											 {"look-at"});
	args::ValueFlag<std::string> size_arg(parser, "WxH", "Window size in points (overrides the camera's)", {"size"});
	args::Flag figure_arg(parser, "preview-style", "magma_r, [0, 0.12], per face, out-of-band edges",
						  {"preview-style"});
	args::ValueFlag<std::string> colormap_arg(parser, "NAME", "Colour map (Polyscope names or magma_r)", {"colormap"});
	args::ValueFlag<std::string> range_arg(parser, "MIN,MAX", "Loss colour range", {"range"});
	args::ValueFlag<std::string> coloring_arg(
		parser, "MODE", "vertex (mean of its edges) or face (mean of its vertices)", {"coloring"});
	args::Flag band_arg(parser, "out-of-band",
						"Highlight edges longer than ell_max (green), shorter than ell_min (cyan)", {"out-of-band"});
	args::Flag no_loss_arg(parser, "no-loss", "Plain surface colour instead of the loss", {"no-loss"});
	args::ValueFlag<float> edge_width_arg(parser, "VALUE", "Wireframe width, 0 hides it (default 1)", {"edge-width"});
	args::ValueFlag<float> exposure_arg(parser, "VALUE", "Tone-mapping exposure (Polyscope default 1.1)", {"exposure"});
	args::Flag no_ground_arg(parser, "no-ground", "Disable the ground shadow", {"no-ground"});
	args::Flag transparent_arg(parser, "transparent", "Transparent background", {"transparent"});
	args::ValueFlag<int> ssaa_arg(parser, "N", "Supersampling factor (default 1)", {"ssaa"});

	try {
		parser.ParseCLI(argc, argv);
	} catch (const args::Help&) {
		std::cout << parser;
		return 0;
	} catch (const args::Error& error) {
		std::cerr << error.what() << "\n" << parser;
		return 2;
	}
	if (!input_arg || !output_arg) {
		std::cerr << "INPUT and OUTPUT are required.\n" << parser;
		return 2;
	}

	const fs::path input = args::get(input_arg);
	const bool is_result_folder = fs::is_directory(input);
	remeshing::Mesh mesh;
	pmp::read(mesh, is_result_folder ? visuals::remeshed_mesh_in(input) : input);

	double ell_star = 0.0;
	if (target_length_arg)
		ell_star = args::get(target_length_arg);
	else if (target_arg)
		ell_star = args::get(target_arg) * remeshing::average_edge_length(mesh);
	else if (is_result_folder)
		ell_star = nlohmann::json::parse(read_text(input / "summary.json")).at("target_length").get<double>();
	else
		throw std::invalid_argument("No target length: pass a result folder, --target or --target-length");

	visuals::LossVisuals style = figure_arg ? visuals::figure_style() : visuals::LossVisuals{};
	if (coloring_arg) {
		const std::string coloring = args::get(coloring_arg);
		if (coloring != "vertex" && coloring != "face") throw std::invalid_argument("--coloring needs vertex or face");
		style.coloring = coloring == "vertex" ? visuals::LossColoring::Vertex : visuals::LossColoring::Face;
	}
	if (colormap_arg) style.color_map = args::get(colormap_arg);
	if (range_arg) {
		const auto range = parse_floats(args::get(range_arg), "range");
		if (range.size() != 2) throw std::invalid_argument("--range needs MIN,MAX");
		style.map_range = {range[0], range[1]};
	}
	if (band_arg) style.out_of_band_edges = true;

	polyscope::options::programName = "mesh-screenshot";
	polyscope::options::usePrefsFile = false;
	polyscope::options::giveFocusOnShow = false;
	polyscope::options::verbosity = 0;
	polyscope::options::groundPlaneMode =
		no_ground_arg ? polyscope::GroundPlaneMode::None : polyscope::GroundPlaneMode::ShadowOnly;
	polyscope::options::shadowBlurIters = 6;
	if (ssaa_arg) polyscope::options::ssaaFactor = args::get(ssaa_arg);
	polyscope::init();
	visuals::register_extra_color_maps();
	if (exposure_arg) polyscope::render::engine->exposure = args::get(exposure_arg);

	auto* surface = visuals::draw_mesh(mesh, static_cast<float>(ell_star), style);
	surface->setEdgeWidth(edge_width_arg ? args::get(edge_width_arg) : 1.0f);
	surface->getQuantity("Edge Loss")->setEnabled(!no_loss_arg);

	if (camera_arg) {
		const std::string camera = args::get(camera_arg);
		polyscope::view::setViewFromJson(camera.rfind('{', 0) == 0 ? camera : read_text(camera), false);
	} else if (look_at_arg) {
		const auto v = parse_floats(args::get(look_at_arg), "look-at");
		if (v.size() != 6 && v.size() != 9) throw std::invalid_argument("--look-at needs 6 or 9 values");
		const glm::vec3 eye(v[0], v[1], v[2]), target(v[3], v[4], v[5]);
		if (v.size() == 9)
			polyscope::view::lookAt(eye, target, glm::vec3(v[6], v[7], v[8]));
		else
			polyscope::view::lookAt(eye, target);
	} else {
		polyscope::view::resetCameraToHomeView();
	}
	if (size_arg) {
		const std::string size = args::get(size_arg);
		const auto x = size.find('x');
		if (x == std::string::npos) throw std::invalid_argument("--size needs WxH");
		polyscope::view::setWindowSize(std::stoi(size.substr(0, x)), std::stoi(size.substr(x + 1)));
	}

	polyscope::ScreenshotOptions options;
	options.transparentBackground = transparent_arg;
	polyscope::screenshot(args::get(output_arg), options);
	std::cout << args::get(output_arg) << " (" << polyscope::view::bufferWidth << "x" << polyscope::view::bufferHeight
			  << " px, target length " << ell_star << ")\n";
	return 0;
}

} // namespace

int main(int argc, char** argv) {
	try {
		return run(argc, argv);
	} catch (const std::exception& error) {
		std::cerr << "mesh-screenshot: " << error.what() << "\n";
		return 1;
	}
}
