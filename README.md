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
| `--approach` | `baseline`, `repeated`, `ordered`, `dynamic-queue`, `batched`, `hybrid`, `alternating` | all |
| `--objective` | `sum`, `maximum`, `local-average`, `global-average` | queue-based approaches |
| `--target` | T, with ℓ\* = T · ℓ_avg | all |
| `--cycles` | k (default 5) | `baseline`, `ordered` |
| `--no-knee` | ρ = 0: a queue stage ends only at a window that raises the average loss | `alternating` |
| `--trace FILE`, `--trace-interval N` | Writes the average edge loss every N edits (default 1000), and at every cycle and stage | all |

Every other constant is in [`src/remeshing/parameters.h`](src/remeshing/parameters.h).

Input meshes must be closed triangle meshes without isolated vertices (Sec. 3.1.1). `meshes/` holds the ten benchmark
meshes of the thesis and `cube.obj`, a small test mesh.

## Reproducing the thesis

runner script --> single full dataset --> generate results, figures and appendixes from dataset

```sh
python3 -m pip install -r scripts/requirements.txt          
python3 scripts/run_experiments.py meshes FOLDER                # about 75 minutes and 11 GB. Into data/
python3 scripts/generate_figures.py --data FOLDER --runs FOLDER # into results/
```

| Script | Writes |
|---|---|
| `run_experiments.py` |`results.csv`, `traces.csv`, `scans.csv`, `inputs.csv`, `failures.csv`, `BUILD_INFO`, and the output meshes.  |
| `generate_figures.py [NAME ...]` | Every figure and table of the thesis, from one module per figure or table in [`scripts/figures/`](scripts/figures): the schematic figures of Ch. 2 and 3, the plots and tables of Ch. 3, 4 and App. A and B, and with `--runs` the mesh images of Ch. 1 and 4. Helpers they share are in [`scripts/shared/`](scripts/shared). |

[`data/`](data) holds the CSV files of the run behind the thesis, without its output meshes, so that all tables and plots
rebuild without running anything (`python3 scripts/generate_figures.py`). The mesh images and the gallery of App. A are
rendered with `mesh-screenshot` and need a display; they are only made with `--runs`, which also gives the mesh images
their output meshes, or when named (`generate_figures.py mesh_gallery`).

[`scripts/claims.py`](scripts/claims.py) is not part of the pipeline. It shows how each number in the text follows from
`data/`, going through the thesis section by section, and prints the numbers when run (`python3 scripts/claims.py`).

## Licence and mesh credits

The code and scripts are under the MIT licence ([`LICENSE`](LICENSE)). The benchmark meshes in `meshes/` are not. They
come from the sources below and were prepared as described in App. A of the thesis; the header of each file names its
preparation.

| Mesh | Source |
|---|---|
| `stanford-bunny`, `stanford-bunny-low` (decimated) | [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) |
| `armadillo-r128` | [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) |
| `stanford-dragon-r128` | [Stanford 3D Scanning Repository](https://graphics.stanford.edu/data/3Dscanrep/) |
| `fandisk` | Hoppe et al. 1994, from the sample data of [PMP](https://github.com/pmp-library/pmp-library) 3.0.0 |
| `max-planck` (MPI), `horse` (Cyberware), `rocker-arm` (INRIA), `homer` (origin unknown) | [Common 3D Test Models](https://github.com/alecjacobson/common-3d-test-models), Alec Jacobson |
| `elk` | [CGAL sample meshes](https://github.com/CGAL/cgal/tree/master/Data/data/meshes) |

The Stanford models may be used and redistributed for research with credit to the Stanford Computer Graphics
Laboratory, but not for commercial purposes without its permission. The other sources state no licence for these models.
