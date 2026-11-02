#include "remeshing/edits.h"

#include "remeshing/parameters.h"

#include <iomanip>

namespace remeshing {

void trace_event(const Run& run, const char* event) {
	if (!run.trace) return;
	*run.trace << event << ',' << run.edits.total() << ',' << std::setprecision(17) << run.average_loss() << ','
			   << run.mesh.n_edges() << ',' << run.mesh.n_vertices() << '\n';
}

namespace {

void count_edit(Run& run, uint64_t& counter) {
	counter++;
	if (run.trace && run.edits.total() % run.trace_interval == 0) trace_event(run, "edits");
	if (run.edits_done) run.edits_done->store(run.edits.total(), std::memory_order_relaxed);
}

} // namespace

Run::Run(Mesh& mesh, Objective objective, double ell_star, std::atomic<uint64_t>* edits_done)
	: mesh(mesh), objective(objective), ell_star(ell_star), ell_max(ell_star * ELL_MAX_RATIO),
	  ell_min(ell_star * ELL_MIN_RATIO), losses(mesh, ell_star), edits_done(edits_done) {}

double Run::average_loss() const { return losses.energy() / static_cast<double>(mesh.n_edges()); }

void Run::collect_garbage() {
	mesh.garbage_collection();
	losses.rebuild();
}

Vertex split_edge(Run& run, Edge e, const Point& position) {
	run.losses.remove(e);
	const Vertex created = run.mesh.to_vertex(run.mesh.split(e, position));
	for (Edge edge : run.mesh.edges(created)) run.losses.add(edge);
	count_edit(run, run.edits.splits);
	return created;
}

Vertex collapse_edge(Run& run, Halfedge h, const Point& position) {
	const Vertex from = run.mesh.from_vertex(h);
	const Vertex kept = run.mesh.to_vertex(h);
	const Edge collapsed = run.mesh.edge(h);
	for (Edge edge : run.mesh.edges(from)) run.losses.remove(edge);
	for (Edge edge : run.mesh.edges(kept))
		if (edge != collapsed) run.losses.remove(edge);
	run.mesh.position(kept) = position;
	run.mesh.collapse(h);
	for (Edge edge : run.mesh.edges(kept)) run.losses.add(edge);
	count_edit(run, run.edits.collapses);
	return kept;
}

void flip_edge(Run& run, Edge e) {
	run.losses.remove(e);
	run.mesh.flip(e);
	run.losses.add(e);
	count_edit(run, run.edits.flips);
}

void relocate_vertex(Run& run, Vertex v, const Point& position) {
	for (Edge edge : run.mesh.edges(v)) run.losses.remove(edge);
	run.mesh.position(v) = position;
	for (Edge edge : run.mesh.edges(v)) run.losses.add(edge);
	count_edit(run, run.edits.relocations);
}

} // namespace remeshing
