// Sec. 3.2.2: the dynamic queue (Algorithm 3.1).

#include "remeshing/evaluate.h"
#include "remeshing/schedules.h"

#include <cmath>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace remeshing {

namespace {

struct Entry {
	double priority; // g(o), or g_sum(o) in the queues of the global average
	int version;
	Operation operation;
	int element; // index of the edge or vertex
	int d_o;
	Point position;
	Halfedge collapse;
};

struct LowerPriority {
	bool operator()(const Entry& lhs, const Entry& rhs) const { return lhs.priority < rhs.priority; }
};

class Heap : public std::priority_queue<Entry, std::vector<Entry>, LowerPriority> {
public:
	void reserve(size_t capacity) { this->c.reserve(capacity); }
};

// One priority queue, or one per value of d_o for the global average.
class DynamicQueue {
public:
	explicit DynamicQueue(Run& run)
		: run_(run), mesh_(run.mesh), global_average_(run.objective == Objective::GlobalAverage),
		  heaps_(global_average_ ? 3 : 1), split_versions_(mesh_.edge_property<int>("e:split_version", 0)),
		  collapse_versions_(mesh_.edge_property<int>("e:collapse_version", 0)),
		  flip_versions_(mesh_.edge_property<int>("e:flip_version", 0)),
		  relocation_versions_(mesh_.vertex_property<int>("v:relocation_version", 0)) {
		if (!global_average_) heaps_[0].reserve(3 * mesh_.n_edges() + mesh_.n_vertices());
	}

	~DynamicQueue() {
		mesh_.remove_edge_property(split_versions_);
		mesh_.remove_edge_property(collapse_versions_);
		mesh_.remove_edge_property(flip_versions_);
		mesh_.remove_vertex_property(relocation_versions_);
	}

	void evaluate_every_candidate() {
		tick_++;
		for (Edge e : mesh_.edges()) {
			evaluate_split_candidate(e);
			evaluate_collapse_candidate(e);
			evaluate_flip_candidate(e);
		}
		for (Vertex v : mesh_.vertices()) evaluate_relocation_candidate(v);
	}

	bool empty() const {
		return std::all_of(heaps_.begin(), heaps_.end(), [](const Heap& heap) { return heap.empty(); });
	}

	// The next entry to remove: the head of the only queue, or for the global average, of the three heads, the one
	// with the largest current gain among those that pass their acceptance test. 
	Heap& next_heap() {
		if (!global_average_) return heaps_[0];
		size_t best = heaps_.size();
		double best_gain = 0.0;
		for (size_t i = 0; i < heaps_.size(); ++i) {
			if (heaps_[i].empty()) continue;
			const Entry& head = heaps_[i].top();
			const double gain = current_gain_if_accepted(head);
			if (best == heaps_.size() || gain > best_gain) {
				best = i;
				best_gain = gain;
			}
		}
		return heaps_[best];
	}

	bool is_global_average() const { return global_average_; }

	bool accepted_now(const Entry& entry) const {
		return current_gain_if_accepted(entry) != -std::numeric_limits<double>::infinity();
	}

	// An entry is current only if its element still exists and still has the version stored in the entry.
	bool is_current(const Entry& entry) const {
		switch (entry.operation) {
		case Operation::Split:
			return !mesh_.is_deleted(Edge(entry.element)) && entry.version == split_versions_[Edge(entry.element)];
		case Operation::Collapse:
			return !mesh_.is_deleted(Edge(entry.element)) && entry.version == collapse_versions_[Edge(entry.element)];
		case Operation::Flip:
			return !mesh_.is_deleted(Edge(entry.element)) && entry.version == flip_versions_[Edge(entry.element)];
		case Operation::Relocation:
			return !mesh_.is_deleted(Vertex(entry.element)) &&
				   entry.version == relocation_versions_[Vertex(entry.element)];
		}
		return false;
	}

	// Applies a current entry with its stored edit and re-evaluates the footprint, unless `stops` says the stage ends.
	template <typename Stops> bool apply(const Entry& entry, Stops&& stops) {
		anchors_.clear();
		reconnected_.clear();
		switch (entry.operation) {
			case Operation::Split: {
				const Edge e(entry.element);
				reconnected_.emplace_back(mesh_.vertex(e, 0), mesh_.vertex(e, 1));
				anchors_.push_back(split_edge(run_, e, entry.position));
				break;
			}
			case Operation::Collapse:
				anchors_.push_back(collapse_edge(run_, entry.collapse, entry.position));
				break;
			case Operation::Flip: {
				const Edge e(entry.element);
				const Vertex v0 = mesh_.vertex(e, 0);
				const Vertex v1 = mesh_.vertex(e, 1);
				const Vertex w0 = mesh_.to_vertex(mesh_.next_halfedge(mesh_.halfedge(e, 0)));
				const Vertex w1 = mesh_.to_vertex(mesh_.next_halfedge(mesh_.halfedge(e, 1)));
				flip_edge(run_, e);
				anchors_ = {v0, v1, w0, w1};
				reconnected_ = {{v0, v1}, {w0, w1}};
				break;
			}
			case Operation::Relocation:
				relocate_vertex(run_, Vertex(entry.element), entry.position);
				anchors_.push_back(Vertex(entry.element));
				break;
		}
		if (stops()) return false;
		evaluate_footprint(entry);
		return true;
	}

private:
	Run& run_;
	Mesh& mesh_;
	bool global_average_;
	std::vector<Heap> heaps_;
	pmp::EdgeProperty<int> split_versions_;
	pmp::EdgeProperty<int> collapse_versions_;
	pmp::EdgeProperty<int> flip_versions_;
	pmp::VertexProperty<int> relocation_versions_;
	int tick_ = 0;				  // the version of all candidates evaluated after the current edit
	std::vector<Vertex> anchors_; // the new, merged or relocated vertex, or the four vertices of a flip
	std::vector<std::pair<Vertex, Vertex>> reconnected_; // The edges the edit connects or disconnects without anchor.

	double current_gain_if_accepted(const Entry& entry) const {
		const int edge_count = static_cast<int>(mesh_.n_edges());
		const double gain = global_average_gain(run_.losses.energy(), edge_count, entry.priority, entry.d_o);
		return global_average_accepts(gain, run_.epsilon, edge_count + entry.d_o)
				   ? gain
				   : -std::numeric_limits<double>::infinity();
	}

	void push(const Evaluation& evaluation, Operation operation, int element) {
		if (!evaluation.passes) return;
		const double priority = global_average_ ? evaluation.g_sum : evaluation.gain;
		const Entry entry{priority, tick_, operation, element, evaluation.d_o, evaluation.position,
						  evaluation.collapse};
		heaps_[global_average_ ? (evaluation.d_o + 3) / 3 : 0].push(entry);
	}

	void evaluate_split_candidate(Edge e) {
		split_versions_[e] = tick_;
		push(evaluate_split(run_, e, global_average_), Operation::Split, e.idx());
	}
	void evaluate_collapse_candidate(Edge e) {
		collapse_versions_[e] = tick_;
		push(evaluate_collapse(run_, e, global_average_), Operation::Collapse, e.idx());
	}
	void evaluate_flip_candidate(Edge e) {
		flip_versions_[e] = tick_;
		push(evaluate_flip(run_, e), Operation::Flip, e.idx());
	}
	void evaluate_relocation_candidate(Vertex v) {
		relocation_versions_[v] = tick_;
		push(evaluate_relocation(run_, v), Operation::Relocation, v.idx());
	}

	// The candidates whose evaluation reads elements the edit changed, each evaluated once.
	void evaluate_footprint(const Entry& executed) {
		tick_++;
		const bool flip = executed.operation == Operation::Flip;

		// Split and flip candidates read the two triangles at their edge: after a flip, its two new triangles;
		// otherwise, the triangles at the new or merged vertex or the relocated vertex.
		const auto for_each_edge_of_changed_faces = [&](auto&& visit) {
			const auto visit_face = [&](Face f) {
				for (Halfedge h : mesh_.halfedges(f)) visit(mesh_.edge(h));
			};
			if (flip) {
				const Edge e(executed.element);
				visit_face(mesh_.face(e, 0));
				visit_face(mesh_.face(e, 1));
				return;
			}
			for (Vertex anchor : anchors_)
				for (Face f : mesh_.faces(anchor)) visit_face(f);
		};
		for_each_edge_of_changed_faces([&](Edge e) {
			if (split_versions_[e] != tick_) evaluate_split_candidate(e);
		});
		for_each_edge_of_changed_faces([&](Edge e) {
			if (flip_versions_[e] != tick_) evaluate_flip_candidate(e);
		});
		// Whether a flip is legal also depends on whether its two opposite vertices are connected.
		for (const auto& [a, b] : reconnected_)
			for (Edge e : flips_connecting(mesh_, a, b))
				if (flip_versions_[e] != tick_) evaluate_flip_candidate(e);

		// A collapse candidate reads the rings of both its ends. A flip changes only the rings of its four vertices.
		const auto evaluate_collapses_at = [&](Vertex v) {
			for (Edge e : mesh_.edges(v))
				if (collapse_versions_[e] != tick_) evaluate_collapse_candidate(e);
		};
		for (Vertex anchor : anchors_) {
			evaluate_collapses_at(anchor);
			if (flip) continue;
			for (Vertex neighbor : mesh_.vertices(anchor)) evaluate_collapses_at(neighbor);
		}

		// A relocation candidate reads its vertex's position, ring and triangles.
		const auto evaluate_relocation_at = [&](Vertex v) {
			if (relocation_versions_[v] != tick_) evaluate_relocation_candidate(v);
		};
		for (Vertex anchor : anchors_) {
			evaluate_relocation_at(anchor);
			if (flip) continue;
			for (Vertex neighbor : mesh_.vertices(anchor)) evaluate_relocation_at(neighbor);
		}
	}
};

// Sec. 3.2.3: the knee is the end of the first weak window of K executed edits. A window is weak if the average loss
// increases, or if its relative decrease is less than rho of the largest relative decrease so far.
class KneeRule {
public:
	explicit KneeRule(const Run& run)
		: window_(
			  std::max<uint64_t>(KNEE_MIN_WINDOW, (run.mesh.n_edges() + run.mesh.n_vertices()) / KNEE_WINDOW_DIVISOR)),
		  window_start_loss_(run.average_loss()), rho_(run.rho) {}

	bool reached(const Run& run, uint64_t executed) {
		if (executed - window_start_ < window_) return false;
		const double loss = run.average_loss();
		const double relative_decrease = (window_start_loss_ - loss) /
										 std::max(std::abs(window_start_loss_), std::numeric_limits<double>::epsilon());
		largest_decrease_ = std::max(largest_decrease_, relative_decrease);
		if (relative_decrease < 0.0 || (largest_decrease_ > 0.0 && relative_decrease < rho_ * largest_decrease_))
			return true;
		window_start_loss_ = loss;
		window_start_ = executed;
		return false;
	}

private:
	uint64_t window_;
	uint64_t window_start_ = 0;
	double window_start_loss_;
	double largest_decrease_ = 0.0;
	float rho_;
};

} // namespace

StopReason run_queue_stage(Run& run, StageEnd end) {
	StopReason reason = StopReason::NoCandidate;
	{
		DynamicQueue queue(run);
		queue.evaluate_every_candidate();

		const uint64_t cap = edit_cap(run.mesh.n_edges() + run.mesh.n_vertices());
		LocalAverageSafeguard safeguard(run);
		KneeRule knee(run);
		uint64_t executed = 0;

		while (!queue.empty()) {
			if (executed >= cap) {
				reason = StopReason::EditCap;
				break;
			}
			Heap& heap = queue.next_heap();
			const Entry entry = heap.top();
			if (queue.is_global_average() && queue.is_current(entry) && !queue.accepted_now(entry)) break;
			heap.pop();
			if (!queue.is_current(entry)) continue;

			executed++;
			const bool keep_going = queue.apply(entry, [&] {
				if (end == StageEnd::AtKnee) {
					if (!knee.reached(run, executed)) return false;
					reason = StopReason::Knee;
					return true;
				}
				if (!safeguard.stops(run, executed)) return false;
				reason = StopReason::Safeguard;
				return true;
			});
			if (!keep_going) break;
		}
	}
	run.collect_garbage();
	trace_event(run, "stage");
	return reason;
}

} // namespace remeshing
