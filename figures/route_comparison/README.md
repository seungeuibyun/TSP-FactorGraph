# Route-comparison figures

The three `route_h11_*_ieee` figures compare Proposed, ACO, and GA on the
same 84-bin, 10-vehicle, hour-11, seed-7 instance. Road paths are reconstructed
from each saved route with the same time- and payload-dependent SUMO energy
oracle used by the evaluator.

Regenerate them with:

```powershell
python -m scripts.plotting.route_maps --hour 11 --seed 7
```

The hourly experiment pipeline also regenerates these panels after completing
an hour-11 run.
