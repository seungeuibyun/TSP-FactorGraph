# Scalability experiment

This experiment increases the number of service bins from 84 to 350 while
keeping the average route cardinality close to 8.4 bins per vehicle. Vehicle
counts use `floor(N / 8.4 + 0.5)`.

Ten deterministic spatial seeds independently permute the 500 valid candidate
locations in `simul/scalability_bins_500.csv`. Within each spatial layout,
larger N values extend the same prefix, so growth with N remains paired and
nested. Two demand seeds are evaluated for each layout. The experiment
therefore contains 20 instances per N and 1,300 solver runs in total:
13 sizes x 10 layouts x 2 demand seeds x 5 methods.

The parent process uses two worker processes. Each worker owns one complete
`(N, spatial seed, demand seed)` instance and evaluates all five methods using
the same loaded instance and prepared energy oracle. This avoids duplicating
the oracle within an instance and prevents concurrent writes to the combined
CSV. Every completed method is first stored in an independent shard; the
parent then merges shards into the combined CSV and refreshes the figures.
Interrupted work resumes at method granularity.

Generated layouts, shards, routes, logs, manifests, and combined CSV files are
stored below `results/scalability/spatial_mc/`. Earlier fixed-layout results in
`results/scalability/` are intentionally retained and are not mixed into this
experiment.

Run or resume from the repository root in the `tsp` environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start_scalability_monte_carlo.ps1 `
  -Python 'C:\Users\guild\.conda\envs\tsp\python.exe'
```

The configured worker count is deliberately two because each large proposed
instance can consume substantial memory. Increase `execution.parallel_jobs`
only after checking available RAM; native numerical thread counts are fixed to
one by the launcher to avoid CPU oversubscription.
