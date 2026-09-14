# TSP-FactorGraph

Reproducible EV waste-collection experiments for the assignment/trellis
message-passing solver and the NN, ACO, PSO, and GA baselines.

## Repository layout

```text
solver/       proposed solver, message updates, exact trellis, and baselines
simul/        operational EV energy model and required SUMO/traffic/bin data
scripts/      experiment runner, experiment implementations, and plotters
experiments/  one JSON configuration for each retained paper experiment
results/      retained CSV/JSON/log outputs
figures/      publication figures, separated from raw results
tests/        solver and online-framework regression tests
docs/         manuscript and energy-model documentation
```

The four retained experiments are:

- `k_sweep`: K=6,...,14 at 04:00, 11:00, and 19:00 with 11,700 kg demand.
- `hourly_24h`: K=10 over all 24 hourly traffic snapshots.
- `online`: paired adaptive, live-static, and frozen-static policies over ten
  traffic seeds with gradual disruption and recovery.
- `scalability`: nested N=84,...,500 instances with K selected so that each
  vehicle services approximately 8.4 bins. Each point is a reproducible
  ten-seed Monte Carlo mean; the energy and runtime figures show one sample
  standard deviation as a shaded band and are refreshed after every completed
  seed. Its checked-in configuration uses
  the `global_assignment_trajectory` schedule: the full I/V graph performs an
  unrestricted decode, every feasible decoded partition advances to an exact
  assigned-set trellis even when its instantaneous energy is worse, and the
  best feasible plan is retained separately. All cut assignment edges receive
  physical fixed-order insertion/exchange messages; only a detected loopy
  cycle triggers exact refinement of the moved source edges. No K-best,
  cardinality cap, or Hamming-radius restriction is applied. Assigned sets of
  at most 20 bins use the exact state-aware FullRouteTable. A larger transient
  set remains in the assignment domain but uses the same operational energy
  tensor with a polynomial fixed-path boundary update until it returns to the
  exact range; each such use is recorded as `oversize_path_fallbacks`.

## Setup

### Git LFS runtime input

The 226 MB elevated SUMO network
`simul/seongbuk_buffer_elevation.net.xml` is stored with Git Large File Storage
(Git LFS). Install Git LFS before cloning so that Git replaces the small pointer
in the repository with the actual XML file.

On macOS:

```bash
brew install git-lfs
git lfs install
```

On Windows, install Git LFS from [git-lfs.com](https://git-lfs.com/), open a
new PowerShell or Git Bash session, and run:

```powershell
git lfs install
```

Then clone the repository normally. The final `git lfs pull` is harmless when
the file was already downloaded automatically and ensures that it is present:

```bash
git clone https://github.com/shoreview01/TSP-FactorGraph.git
cd TSP-FactorGraph
git lfs pull
```

For an existing clone, install Git LFS and retrieve the runtime input with:

```bash
git lfs install
git lfs pull
```

Verify that the network is managed by Git LFS:

```bash
git lfs ls-files
```

The output should include `simul/seongbuk_buffer_elevation.net.xml`. Prefer
`git clone` over GitHub's **Download ZIP**, because a source archive may contain
only the LFS pointer instead of the 226 MB XML file.

### Python environment

Use Python 3.11 or newer and install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The runtime road inputs are already under `simul/`; SUMO itself is not required
to rerun these experiments because the operational hourly edge table is checked
in.

## Run an experiment

Run from the repository root. Each configuration fixes the instance, solver,
baseline, traffic, parallelism, and destination paths. Completed shards/seeds
are reused while `"resume": true`. For scalability, proposed rows whose
recorded `route_message_mode` differs from the configured solver are
invalidated on resume while completed baseline rows remain reusable. This
prevents old and new proposed schedules from being mixed in one curve.

```bash
python -m scripts.run_experiment experiments/k_sweep/config.json
python -m scripts.run_experiment experiments/hourly_24h/config.json
python -m scripts.run_experiment experiments/online/config.json
python -m scripts.run_experiment experiments/scalability/config.json
```

On Windows, the ten-seed scalability run can also be launched in the
background from an activated environment. It writes timestamped logs and a PID
under `results/scalability/`:

```powershell
.\scripts\start_scalability_monte_carlo.ps1 -Python (Get-Command python).Source
```

To inspect the commands without starting a long run:

```bash
python -m scripts.run_experiment experiments/k_sweep/config.json --dry-run
```

The scalability location set can be rebuilt deterministically with:

```bash
python -m scripts.generate_scalability_bins --maximum-bins 500 --seed 2026
```

To regenerate only the paper figures from retained CSV files:

```bash
python -m scripts.run_experiment experiments/k_sweep/config.json --figures-only
python -m scripts.run_experiment experiments/hourly_24h/config.json --figures-only
python -m scripts.run_experiment experiments/online/config.json --figures-only
python -m scripts.run_experiment experiments/scalability/config.json --figures-only
```

New tabular outputs stay under `results/<experiment>/`; plots are written only
to `figures/<experiment>/` in PDF, EPS, and PNG formats.

## Validation

```bash
python -m unittest discover -s tests -v
```

The physical energy terms and parameter definitions are documented in
`docs/energy_model.md`. The latest manuscript is `docs/manuscript.pdf`.
