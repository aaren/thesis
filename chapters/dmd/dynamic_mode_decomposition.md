## Introduce DMD

### Build Physical Intuition

### Basic Linear Algebra

### Compare with POD / EOF


## Apply DMD to data

```python
import numpy as np
import matplotlib.pyplot as plt

import gc_turbulence as g
import sparse_dmd
from sparse_dmd import DMD


index = 'r14_01_14a'
# waveless is temporary until we integrate into ProcessedRun
r = g.WavelessRun(index)

# 400 is temporary until we clean out the nans
uf = r.Uf[:, :, 400:]

snapshots = sparse_dmd.to_snaps(uf, decomp_axis=1)
dmd = DMD(snapshots)
dmd.compute()
```

Basic dmd plots...

```python
fig, ax = plt.subplots()
ax.set_title('|xdmd| vs frequency')
ax.set_xlabel('frequency')
ax.set_ylabel('amplitude')
ax.plot(dmd.frequencies.imag, np.abs(dmd.amplitudes), 'ko')
```



## Introduce problem 'how to select modes'?

Overview existing methods - the intensive one used by chen?, semeraro, sparse
dmd

semeraro - 'convergence on attractors of fully developed turbulence'

## Define sparse method


## Apply sparse method

```python
from sparse_dmd import SparseDMD

sdmd = SparseDMD(dmd=dmd)
# TODO: define appropriate gammaval
sdmd.compute_sparse(gammaval=np.logspace(-2, 6, 50))
```

look at modes, reconstruct data.


## Compare basic flow characterisation before and after

## Compare pdf before and after


