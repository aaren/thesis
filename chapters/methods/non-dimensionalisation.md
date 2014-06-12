Non-dimensionalisation
======================

We have performed experiments over a range of physical parameters.
Combining these experiments in an ensemble is possible provided that
the data is non-dimensionalised.

Non-dimensionalising the data consists of two steps:

1. Dividing through by length / time / velocity scales.

2. Re-sampling to a regular grid that is the same for all runs.

```python
%matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

import gc_turbulence as g

r = g.ProcessedRun(cache_path=g.default_processed + 'r13_12_16a.hdf5')
```


### Scales

There are two length scales - horizontal and vertical. The
horizontal length scale is defined as the lock length $L$. The
vertical length scale is defined as the fluid depth $H$.

There velocity scale is $U = \sqrt{g' H}$, where the reduced gravity
$g'$ is

$$
g' = g \frac{(\rho_{lock} - \rho_{ambient})}{\rho_{ambient}}
$$

We can define a time scale $T$ as $H / U$.

```python
p = r.attributes

L = p['L']
H = p['H']
g_ = 9.81 * (p['rho_lock'] - p['rho_ambient']) / p['rho_ambient']
U = (g_ * H) ** .5
T = H / U
```

Be careful. There are experiments with varying $L$. It may appear
that we can compare these runs non-dimensionally but this is not
neccesarily true as they capture different phases in the gravity
current lifetime.

### Re-sampling

Scaling the dimensional quantities also scales their spacing on the
measurement grid, i.e. as well as scaling $X, Z, T$ we also scale
$dx, dz, dt$. This means that runs with different parameters are
sampled differently in non-dimensional space.

To completely compare runs we have to re-sample to a regular grid in
non-dimensional space. 

Our dimensional sampling intervals are: 

- time: 0.01 seconds
- vertical: 0.001158 metres
- horizontal: 0.001439 metres

We need to choose non-dimensional sampling intervals as these need
to be the same across all runs.

It doesn't make sense to sample any run to a higher resolution than
it was measured at.

We also need to choose the limits of our non-dimensional grid, which
can't extend beyond the edges of the dimensional grid.

We'll work here with the idea that we are looking at full depth runs
with $H=0.25$ and $L=0.25$.

Set the sampling intervals by doubling the non dim interval of the
fastest run.

```python
dx_ = 0.01
dz_ = 0.012
dt_ = 0.015
```

Resampling a regular grid to another regular grid amounts to scaling
each of the axes by some factor. Therefore the actual resampling is
done with `scipy.ndimage.zoom`:

```python
zoom_factor = (dz / (H * dz_),
               dx / (L * dx_),
               dt / (T * dt_))

zoom_kwargs = {'zoom':  zoom_factor,
               'order': 1,          # spline interpolation.
               'mode': 'constant',  # points outside the boundaries
               'cval': np.nan,      # are set to np.nan
               }

Z_ = ndi.zoom(r.Z[:] / H, **zoom_kwargs)
```

It is possible to do this differently using `map_coordinates` by
defining the regular non dimensional grid that we want to map our
data to. This approach is inflexible: different runs have different
scalings and occupy different regions of the non dimensional space
(which may not overlap). We would have to define the grid extent,
which means either a single huge grid that contains the entire
parameter space or a custom grid for each run.

The `zoom` approach conserves all of the run data and determines the
necessary grid extents automatically, whilst still
non-dimensionalising the sampling interval.


### Ensembles

One motivation for non-dimensionalising the data is to allow
comparison between runs with different parameters.

At this point the data are scaled and resampled on a regular non
dimensional grid. However, runs with different scaling will cover
different volumes over this grid. This means no direct stacking of
runs with different scalings.

Some runs are unstackable because they come from different parts of
the gravity current lifetime (those with different L). Other runs
have varying U, H and T but can be considered comparable and
stackable.

Ensemble stacking of runs just requires that, for a set of runs, we
extract a (x, z, t) volume from each run that can be found in all of
the runs.

The only trick here is determining what the shared limits of
multiple runs are.

example:

```python
t = r.T_
tf = r.Tf_
ii = np.where((32 < t[0, 0, :]) & (t[0, 0, :] < 40))
iif = np.where((32 < tf[0, 0, :]) & (tf[0, 0, :] < 40))

t[ii] == tf[iif]
```
