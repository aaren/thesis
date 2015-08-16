---
layout: chapter
title: Dynamic Mode Decomposition of a Gravity Current
---

## Introduce DMD

### Build Physical Intuition

### Basic Linear Algebra

### Compare with POD / EOF


## Apply DMD to data

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

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

Mode frequency vs. amplitude:

```python
fig, ax = plt.subplots()
ax.set_title('|xdmd| vs frequency')
ax.set_xlabel('frequency')
ax.set_ylabel('amplitude')
ax.plot(dmd.frequencies.imag, np.abs(dmd.amplitudes), 'ko')
```

Mode frequency vs. growth rate:

```python
fig, ax = plt.subplots()
ax.set_title('mode spectrum')
ax.set_xlabel('frequency')
ax.set_ylabel('growth rate')
scat = ax.scatter(dmd.frequencies.imag, dmd.frequencies.real,
           c=np.abs(dmd.amplitudes), linewidth=0, cmap=plt.cm.bone_r)
```

First few modes:

```python
# reshape from modes from snapshot to data
modes = sparse_dmd.to_data(dmd.modes, shape=uf.shape, decomp_axis=1)
# transpose so that we index on first axis
modes = modes.transpose((1, 0, 2))
# multiply by amplitude
modes = modes * dmd.amplitudes[:, None, None]
```

```python
# mean is the first mode
mean_levels = np.linspace(-0.11, 0.04, 100)
# subsequent modes need different scaling as they add on top of the mean
non_mean_levels = np.linspace(-0.02, 0.02, 100)

fig, ax = plt.subplots(nrows=5)
ax[0].contourf(modes[0], levels=mean_levels)
ax[1].contourf(modes[2], levels=non_mean_levels)
ax[2].contourf(modes[4], levels=non_mean_levels)
ax[3].contourf(modes[6], levels=non_mean_levels)
ax[4].contourf(modes[8], levels=non_mean_levels)
```

Whilst these modes capture the qualitative features of the current -
the mean field perturbed by a shear layer at the top - they don't
provide us with an especially useful quantitative means of
understanding the flow.

Compare with the first few POD modes. Together, these capture the
bulk of the statistical variance within the current.

```python
# extract the pod modes (the left singluar vectors from svd of snapshots)
pod_modes = sparse_dmd.to_data(dmd.reduction.U, shape=uf.shape, decomp_axis=1)
pod_modes = pod_modes.transpose((1, 0, 2))

singular_values = np.diag(dmd.reduction.S)
norms = np.sqrt(singular_values)

pod_modes = pod_modes * norms[:, None, None]
```

```python
# mean is the first mode
mean_levels = np.linspace(-0.11, 0.04, 100)
# subsequent modes need different scaling as they add on top of the mean
non_mean_levels = np.linspace(-0.03, 0.03, 100)

fig, ax = plt.subplots(nrows=5)
ax[0].contourf(pod_modes[0], levels=mean_levels)
ax[1].contourf(pod_modes[2], levels=non_mean_levels)
ax[2].contourf(pod_modes[4], levels=non_mean_levels)
ax[3].contourf(pod_modes[6], levels=non_mean_levels)
ax[4].contourf(pod_modes[8], levels=non_mean_levels)
```

Here we see a series of modes ranked by their statistical
contribution to the flow. We could rank our DMD modes by this, their
contribution to the variance, but this doesn't necessarily capture
the dynamical drivers of the flow (##reference).

There is no clear way to select the dominant DMD modes. This is a
weakness of the standard DMD method.


## Introduce problem 'how to select modes'?

We have the problem of selecting the dynamic modes that best
represent the flow. A number of techniques have been proposed to
achieve this.

Overview existing methods - the intensive one used by chen?, semeraro, sparse
dmd

semeraro - 'convergence on attractors of fully developed turbulence'

In the sparse method we select a subset of the dmd modes that best
approximate the data series, given a sparsity criterion that varies
the number of modes in the subset.

## Define sparse method


## Apply sparse method

```python
from sparse_dmd import SparseDMD

sdmd = SparseDMD(dmd=dmd)
# TODO: define appropriate gammaval
sdmd.compute_sparse(gammaval=np.logspace(-2, 6, 50))
```

```python
# index where n=7
sdmd.compute_sparse_reconstruction(Ni=21, data=uf.shape, decomp_axis=1)

sparse_amplitudes = sdmd.sparse.xsp[:, 21].astype(bool)
sparse_modes = modes[sparse_amplitudes]
```

In this case I think the sparse modes are just the same as the
leading modes! At least we have demonstrated that.

Will this vary if we add more data?? Are these modes the same as the
ones that survive using semeraros method?

look at modes, reconstruct data.

```python
from sparse_dmd import SparsePlots

plotter = SparsePlots(sdmd)
```


## Compare basic flow characterisation before and after

## Compare pdf before and after


