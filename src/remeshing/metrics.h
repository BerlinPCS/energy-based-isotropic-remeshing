#pragma once

// Sec. 4.1: the quality measures reported in the thesis. The surface deviation d_RMS is measured by
// scripts/measure_deviation.py with MeshLab.

#include "remeshing/edge_loss.h"

namespace remeshing {

struct Metrics {
	double average_edge_loss = 0.0;	   // lambda-bar
	double minimum_angle = 0.0;		   // smallest triangle angle, in degrees
	double angle_p05 = 0.0;			   // 5th percentile of the per-triangle minimum angles, in degrees
	size_t faces_below_1_degree = 0;   // triangles whose smallest angle is below 1 degree
	size_t faces_below_0_001_degree = 0; // ... below 0.001 degrees
	double valence_error = 0.0;		   // RMS deviation of the vertex valences from 6
};

Metrics measure(const Mesh& mesh, double ell_star);

} // namespace remeshing
