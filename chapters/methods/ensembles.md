Combining runs in ensembles
===========================

```python
%matplotlib
import numpy as np
import matplotlib.pyplot as plt
import gc_turbulence as g

run = g.ProcessedRun(cache_path=g.default_processed + 'r13_12_17c.hdf5')

base = g.analysic.Basis(run)

fig = base.mean_velocity_Uf()
plt.show()
```

