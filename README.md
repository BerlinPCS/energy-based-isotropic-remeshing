# Energy-Based Isotropic Remeshing

Code of the bachelor's thesis *Energy-Based Isotropic Remeshing* (TU Berlin, Computer Graphics). It implements the
approaches of Ch. 3 and some of the metrics of Sec. 4.1, and nothing else. Code was adapted after finishing the thesis
to reflect the names and definitions of the thesis.

## Build

Needs CMake 3.14 or newer and a C++17 compiler. PMP 3.0.0 and Polyscope 2.6.1 are downloaded when building for the first time.

```sh
cmake -B build
cmake --build build -j
```

The programs end up in `build/bin/`.

## Applications

| Apps | Purpose |
|---|---|
| `remesher-cli` | Runs one approach on one mesh and writes `remeshed.<ext>` and `summary.json`. |
| `remesher` | GUI for `remesher-cli`. Does not write results files.|
| `viewer` | Browses the result folders of `remesher-cli` with a GUI. |
| `mesh-screenshot` | Renders a mesh or a result folder with the edge-loss colouring to a PNG. |

```sh
build/bin/remesher-cli meshes/cube.obj results/baseline --approach baseline --target 0.3
build/bin/remesher-cli meshes/cube.obj results/hybrid --approach hybrid --objective sum --target 0.3
build/bin/viewer results
```

The following settings are changeable:

| Option | Values | Applies to |
|---|---|---|
| `--approach` | `baseline`, `convergent`, `ordered`, `dynamic-queue`, `batched`, `hybrid`, `alternating` | all |
| `--objective` | `sum`, `maximum`, `local-average`, `global-average` | queue-based approaches |
| `--target` | T, with ℓ\* = T · ℓ_avg | all |
| `--cycles` | k (default 5) | `baseline`, `ordered` |
| `--no-knee` | ρ = 0: a queue stage ends only at a window that raises the average loss | `alternating` |
| `--trace FILE`, `--trace-interval N` | Writes the average edge loss every N edits (default 1000), and at every cycle and stage | all |

Every other constant is in [`src/remeshing/parameters.h`](src/remeshing/parameters.h).

Input meshes must be closed triangle meshes without isolated vertices (Sec. 3.1.1). `meshes/` holds the ten benchmark
meshes of the thesis and `cube.obj`, a small test mesh.

## Reproducing the thesis

prepare meshes --> runner script --> one dataset --> generate results, figures and appendixes from dataset

```sh
python3 -m pip install -r scripts/requirements.txt          
python3 scripts/run_experiments.py meshes FOLDER                # about 75 minutes and 11 GB. Into data/
python3 scripts/generate_results.py --data FOLDER --runs FOLDER # into results/
python3 scripts/generate_figures.py                             # into results/
python3 scripts/generate_appendix_a.py --data FOLDER            # into results/
```

| Script | Writes |
|---|---|
| `run_experiments.py` |`results.csv`, `traces.csv`, `scans.csv`, `inputs.csv`, `horse-cap.json`, `failures.csv`, `BUILD_INFO`, and the output meshes.  |
| `generate_results.py [--only NAME ...]` | `claims.csv`, the tables, the plots and the mesh images of Ch. 1, 3, 4 and App. B. |
| `generate_figures.py [NAME ...]` | The schematic figures of Ch. 2 and 3 as TikZ files. |
| `generate_appendix_a.py` | The mesh table and gallery of App. A. |

[`data/`](data) holds the CSV files of the run behind the thesis, without its output meshes, so that all numbers, tables
and plots rebuild without running anything (`python3 scripts/generate_results.py`). Only the mesh images need the output
meshes (`--runs`) and a display.
