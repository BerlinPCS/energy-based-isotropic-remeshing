#pragma once

// The floating-point constants that are not exactly representable stay in single precision, like PMP's coordinates

namespace remeshing {

// Sec. 3.1.3: edges longer than ell_max are split, edges shorter than ell_min collapse.
inline constexpr double ELL_MAX_RATIO = 4.0 / 3.0; // ell_max = 4/3 ell_*
inline constexpr double ELL_MIN_RATIO = 0.8;	   // ell_min = 4/5 ell_*

// Sec. 3.1.3: normal test, cos(60 degrees).
inline constexpr double NORMAL_TEST_COSINE = 0.5;

// Sec. 3.1.3: relocation step sizes s = 1, 1/2, 1/4, 1/8, 1/16.
inline constexpr int RELOCATION_STEP_SIZES = 5;

// Sec. 3.2.2: base tolerance epsilon = max(1e-5, 3e-5 * average edge loss of the input).
inline constexpr float EPSILON_FLOOR = 1e-5f;
inline constexpr float EPSILON_SCALE = 3e-5f;

// Sec. 3.2.2: safeguard of the local average, W = |E| + |V| executed edits must lower the average loss by delta_q.
inline constexpr float DELTA_Q = 1e-4f;

// Sec. 3.1.4: a cycle or round stagnates if it lowers the average loss by less than delta_c (default).
inline constexpr float DELTA_C = 1e-3f;

// Sec. 3.2.1: the baseline runs k cycles (default).
inline constexpr int DEFAULT_CYCLES = 5;

// Sec. 3.2.3: cycles of the hybrid before its queue stage.
inline constexpr int HYBRID_CYCLES = 2;

// Sec. 3.2.3: knee rule, K = max(50, floor((|E| + |V|) / 50)) edits per window, a window is weak below rho = 1/4.
inline constexpr int KNEE_MIN_WINDOW = 50;
inline constexpr int KNEE_WINDOW_DIVISOR = 50;
inline constexpr float KNEE_RHO = 0.25f;

// Sec. 4.1: practical limits, max(1000, 20 (|E| + |V|)) edits per queue stage and 50 cycles or rounds per schedule.
inline constexpr int EDIT_CAP_MIN = 1000;
inline constexpr int EDIT_CAP_FACTOR = 20;
inline constexpr int MAX_CYCLES = 50;

} // namespace remeshing
