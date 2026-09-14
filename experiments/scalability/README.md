# Scalability experiment

This experiment increases the number of service bins while keeping the average
route cardinality close to 8.4 bins per vehicle. Vehicle counts use
`floor(N / 8.4 + 0.5)`.

| N | K | N/K | Assignment edges N x K |
|---:|---:|---:|---:|
| 84 | 10 | 8.400 | 840 |
| 100 | 12 | 8.333 | 1,200 |
| 120 | 14 | 8.571 | 1,680 |
| 140 | 17 | 8.235 | 2,380 |
| 160 | 19 | 8.421 | 3,040 |
| 180 | 21 | 8.571 | 3,780 |
| 200 | 24 | 8.333 | 4,800 |
| 220 | 26 | 8.462 | 5,720 |
| 240 | 29 | 8.276 | 6,960 |
| 260 | 31 | 8.387 | 8,060 |
| 280 | 33 | 8.485 | 9,240 |
| 300 | 36 | 8.333 | 10,800 |
| 350 | 42 | 8.333 | 14,700 |
| 400 | 48 | 8.333 | 19,200 |
| 450 | 54 | 8.333 | 24,300 |
| 500 | 60 | 8.333 | 30,000 |

The first 84 rows of `simul/scalability_bins_500.csv` are the measured bin
locations used by the other experiments. The remaining 416 locations are
fixed-seed synthetic service points sampled from unique, depot-round-trip-
reachable nodes in the same operational SUMO graph. Every size is a prefix of
this file, so the instances are nested rather than independently resampled.

The primary outputs are solver runtime, oracle preparation time, their sum,
energy per serviced bin, makespan, feasibility, convergence, and the realized
minimum/maximum route cardinality. Runtime figures use a logarithmic y-axis.

Run or resume the complete experiment from the repository root:

```bash
python -m scripts.run_experiment experiments/scalability/config.json
```

The proposed N=84 run alone previously required roughly 23 minutes on the
current machine. The full N=84,...,500 schedule should therefore be treated as
a long, resumable batch. Each completed method/N/seed row is written
immediately. After every N has finished for all configured methods and seeds,
the cumulative runtime and energy figures are regenerated and atomically
overwritten in `figures/scalability/` as EPS, PDF, and PNG files. A plotting
error is reported in the log but does not stop the next N from running.
