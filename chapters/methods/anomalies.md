Filtering anomalies in PIV data
===============================

As well as having missing values, the velocity data contains some
anomalous values. For example:

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import gc_turbulence as g

run = g.SingleLayerRun(cache_path=g.default_cache + 'r13_12_17c.hdf5')
run.load()
pp = g.PreProcessor(run)
pp.extract_valid_region()
pp.filter_zeroes()
pp.interpolate_nan()

# find excessive peaks in the velocity
bad = (np.abs(ndi.filters.uniform_filter(pp.U, 9) - pp.U) > 0.05)
max_bad = bad.sum(axis=(0, 1)).argmax()

plt.contourf(pp.U[:, :, max_bad], 100)
```

We want to remove these values and interpolate over them.  We have
already covered interpolation over arbitrary parts of our data:
therefore we need only set a criteria for finding the anomalous
values, then set them equal to nan and re-interpolate.

```python
filters = [(ndi.uniform_filter,   dict(input=pp.U, size=1)),
           (ndi.uniform_filter,   dict(input=pp.U, size=3)),
           (ndi.uniform_filter,   dict(input=pp.U, size=6)),
           (ndi.uniform_filter,   dict(input=pp.U, size=9)),
           (ndi.gaussian_filter,  dict(input=pp.U, sigma=1)),
           (ndi.gaussian_filter,  dict(input=pp.U, sigma=3)),
           (ndi.gaussian_filter,  dict(input=pp.U, sigma=6)),
           (ndi.gaussian_filter,  dict(input=pp.U, sigma=9))]

fig, ax = plt.subplots()

# compare a load of filters
for i, (filter, kwargs) in enumerate(filters):
    # cumulative histogram of each one
    # all on same plot
    print i
    fout = filter(**kwargs)
    d = np.abs(pp.U - fout)
    values, base = np.histogram(d.flatten(), bins=50)

    ax.plot(base[:-1], values, label=i)

ax.legend()
ax.set_xlim(0, 0.4)
ax.set_yscale('log')

plt.show()
```
