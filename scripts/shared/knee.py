"""The knee rule of Sec. 3.2.3, replayed on the traced queue run of the knee figure."""
import pandas as pd

KNEE_FIGURE = dict(mesh='stanford-bunny', approach='dynamic-queue', objective='local-average', target=1.0)
KNEE_RATE_FRACTION = 0.25


def knee_windows(trace):
    """The windows of the knee rule on a traced queue stage, and the first weak one."""
    first = trace.iloc[0]
    window = max(50, int(first.edges + first.vertices) // 50)
    by_edits = trace.drop_duplicates('edits').set_index('edits').average_edge_loss
    rows, best, start = [], 0.0, 0
    while start + window <= by_edits.index.max():
        end = start + window
        gain = (by_edits[start] - by_edits[end]) / abs(by_edits[start])
        best = max(best, gain)
        weak = gain < 0 or (best > 0 and gain < KNEE_RATE_FRACTION * best)
        rows.append(dict(start=start, end=end, gain=gain, threshold=KNEE_RATE_FRACTION * best, weak=weak))
        start = end
    windows = pd.DataFrame(rows)
    return window, windows, windows[windows.weak].iloc[0]


def knee_figure_trace(data):
    traces = data.traces
    selected = (traces.mesh == KNEE_FIGURE['mesh']) & (traces.approach == KNEE_FIGURE['approach']) & \
        (traces.objective == KNEE_FIGURE['objective']) & (traces.target == KNEE_FIGURE['target'])
    return traces[selected]
