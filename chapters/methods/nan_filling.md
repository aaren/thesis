Interpolation of missing values in gridded data
===============================================

The data computed by Dynamic Studio contains some missing values,
represented by the velocity being identically zero. We would like to
fill in these gaps in the data.

First of all, *do we actually need to do this?* We can probably get
away with not interpolating the regular data, but we have problems
when we make the front relative transform. The front relative
transform is implemented with `scipy.ndimage.map_coordinates` which
doesn't like `nan` values. If you apply `map_coordinates` to data
that hasn't had nans removed, you end up with nans everywhere.

As the front relative transform is a fundamental technique in this
work, allowing us to do analysis on a stationary gravity current, it
is essential that we find a way to deal with this issue.

Let's get an idea of the scale of the problem:

```python
%matplotlib
import numpy as np
import matplotlib.pyplot as plt
import gc_turbulence as g

r = g.ProcessedRun(cache_path=g.default_processed + 'r13_12_17c.hdf5')

velocities = (r.U, r.V, r.W)

zeros = [np.where(d[:] == 0) for d in velocities]
missing = [z[0].size for z in zeros]
total_size = [d[:].size for d in velocities]
proportion = [float(m) / t for m, t in zip(missing, total_size)]

print "Proportion of missing values for each velocity component:"
print proportion
```
Proportion of missing values for each velocity component:
[0.0019078614229048865, 0.0019078614229048865, 0.0019078614229048865]


0.2% - that's pretty small, but bear in mind here that this is
across all of  the data. The invalid values are largely present when
the front is passing through.


### Resources

http://stackoverflow.com/questions/5551286/filling-gaps-in-a-numpy-array

http://stackoverflow.com/questions/20753288/filling-gaps-on-an-image-using-numpy-and-scipy

http://stackoverflow.com/questions/12923593/interpolation-of-sparse-grid-using-python-preferably-scipy

http://stackoverflow.com/questions/14119892/python-4d-linear-interpolation-on-a-rectangular-grid

http://stackoverflow.com/questions/16217995/fast-interpolation-of-regularly-sampled-3d-data-with-different-intervals-in-x-y/16221098#16221098


### Approach

The simplest solution is to replace the invalid data with the
nearest valid neighbour.

http://stackoverflow.com/questions/5551286/filling-gaps-in-a-numpy-array

We have 3 dimensions - two space and one time. How many of them do
we actually need to use in the interpolation?

If we do it at each time step then we can just use 2d interpolation
- just need to find a way to apply this efficiently.

However, the time coordinate is finely resolved and it makes sense
to include it. The speed of the current is

```python
fx, ft = pp.fit_front()
uf = (np.diff(fx) / np.diff(ft)).mean()
print "front speed = ", uf
```

The sampling intervals are

```python
dz = pp.Z[1, 0, 0] - pp.Z[0, 0, 0]
dx = pp.X[0, 1, 0] - pp.X[0, 0, 0]
dt = pp.T[0, 0, 1] - pp.T[0, 0, 0]

print "Sampling intervals:"
print "time: %s seconds" % dt
print "vertical: %s metres" % dz
print "horizontal: %s metres" % dx
```
Sampling intervals:
time: 0.0100002 seconds
vertical: 0.00115751 metres
horizontal: 0.00143858 metres

By using a speed that is representative of stuff moving in the data
we can consider the time interval as an equivalent distance, which
is

```python
print "equivalent time: %s metres" % (uf * dt)
```
equivalent time: 0.000636176 metres

That is, the time sampling is finer than the spatial sampling.


Linear interpolation
--------------------

Let's try find time slice with the most missing values and take a
subsection of it:

```python
sample = np.s_[20:30, :, :]

u = r.U[sample]
v = r.V[sample]
w = r.W[sample]
x = r.X[sample]
z = r.Z[sample]
t = r.T[sample]

# replace the zeros with nans
u[u==0] = np.nan

# find the time index with the most invalid values
import scipy.stats as stats
mode_index, mode_count = stats.mode(np.where(np.isnan(u))[-1])

sample_2d = np.s_[..., mode_index[0]]

us = u[sample_2d]
xs = x[sample_2d]
zs = z[sample_2d]

plt.contourf(xs, zs, us, 50)
```

The white areas are what we want to interpolate over. We can do this
using the `LinearNDInterpolator` from `scipy.interpolate`:

```python
import scipy.interpolate as interp

valid = np.where(~np.isnan(us))
invalid = np.where(np.isnan(us))

valid_points = np.vstack((zs[valid], xs[valid])).T
valid_values = us[valid]

%time imputer = interp.LinearNDInterpolator(valid_points, valid_values)

invalid_points = np.vstack((zs[invalid], xs[invalid])).T
invalid_values = imputer(invalid_points)

infilled = us.copy()
infilled[invalid] = invalid_values

plt.figure()
plt.contourf(xs, zs, infilled, 50)
```

This scales terribly:

```python
valid = np.where(~np.isnan(u))
invalid = np.where(np.isnan(u))

valid_points = np.vstack((z[valid], x[valid], t[valid])).T
valid_values = u[valid]

%time imputer = scipy.interpolate.LinearNDInterpolator(valid_points, valid_values)

invalid_points = np.vstack((z[invalid], x[invalid])).T
invalid_values = imputer(invalid_points)
```

Alternately we could use griddata, which basically wraps the above
method.

Regardless, constructing the linear interpolator on 10% of the data
from a single run takes a very long time. It scales with $N^3$ and
the full N is about 50 million.

We are constructing a Qhull triangulation on a set of points that
come from a regular grid. However we aren't using the fact that we
have a regular grid at all.



Interpolation from valid shell
------------------------------

We can be more efficient by taking advantage of the fact that we
know where our data is invalid - we don't have to construct an
interpolator across the entire data field, just over the regions
that contain invalid data, which are localised.

Each invalid region of the data is surrounded by a shell of valid
data. We can use this shell of valid data as the source for a linear
interpolator and then compute the estimated values of the
interpolated data inside the shell on the regular grid.

We follow this approach:

1. Label the invalid regions of the data
2. Find the valid shell of each region.
3. Construct an interpolator for each valid shell
4. For each label, evaluate the corresponding interpolator over the
   internal coordinates of the label.

Labelling the regions:

```python
import scipy.ndimage as ndi

invalid = np.isnan(u)
valid = ~invalid

# diagonally connected neighbours
connection_structure = np.ones((3, 3, 3))
labels, n = ndi.label(invalid, structure=connection_structure)
```

We find the valid shell by exploiting the fact that our data is on a
rectangular grid and using binary dilation:

```python
def find_valid_shell(label):
    """For an n-dimensional boolean input, return an array of the
    same shape that is true on the exterior surface of the true
    volume in the input."""
    # we use two iterations so that we get the corner pieces as well
    dilation = ndi.binary_dilation(label, structure=np.ones((3, 3, 3)))
    shell = dilation & ~label
    return shell
```

We have now drastically reduced the number of points that our
interpolator uses in its construction.

Construct an interpolator from a valid shell:

```python
def construct_interpolator(valid_shell):
    valid_points = np.vstack((z[valid_shell], x[valid_shell], t[valid_shell])).T
    valid_values = u[valid_shell]
    # this is how to work with three components:
    # valid_values = np.vstack((u[valid_shell], v[valid_shell], w[valid_shell])).T
    interpolator = interp.LinearNDInterpolator(valid_points, valid_values)
    return interpolator
```

Evaluate the points inside the shell:

```python
label = labels == 1
valid_shell = find_valid_shell(label)

interpolator = construct_interpolator(valid_shell)
invalid_points = np.vstack((z[label], x[label], t[label])).T
invalid_values = interpolator(invalid_points)
```

As we are using linear interpolation, we actually need only compute
a single interpolator for the entire field. We can compute the valid
shell around all of the invalid regions and use that as input:

```python
def compare_techniques():
    complete_valid_shell = np.where(find_valid_shell(invalid))
    complete_interpolator = construct_interpolator(complete_valid_shell)

    single_valid_shell = np.where(find_valid_shell(labels==1))
    single_interpolator = construct_interpolator(single_valid_shell)

    # only evaluate inside the labels == 1 shell
    label1 = np.where(labels == 1)
    invalid_points = np.vstack((z[label1], x[label1], t[label1])).T

    complete_invalid_values = complete_interpolator(invalid_points)
    single_invalid_values = single_interpolator(invalid_points)
    return complete_invalid_values, single_invalid_values

print np.allclose(*compare_techniques())
```

Putting it all together:

```python
invalid = np.isnan(u)
complete_valid_shell = np.where(find_valid_shell(invalid))
interpolator = construct_interpolator(complete_valid_shell)

invalid_points = np.vstack((z[invalid], x[invalid], t[invalid])).T
invalid_values = interpolator(invalid_points)

uc = u.copy()
uc[invalid] = invalid_values

ucs = uc[sample_2d]
xs = x[sample_2d]
zs = z[sample_2d]

plt.figure()
plt.contourf(xs, zs, ucs, 50)
```

### Performance

This is slow. Our single interpolator method evaluates a lot of
space on the exterior of the valid shells. Our patch by patch method
calculates a binary dilation for each patch, which is time consuming
over ~5000 patches.

We can avoid repeatedly computing the binary dilation by doing it
once over the entire field. We then isolate disconnected
interpolation volumes, where these contain both the invalid points
and the valid points surrounding them, and construct an interpolator
for each volume.

Isolating features in a nd array can be done using
`scipy.ndimage.label`. Our array is 3d and our features are the
volumes of invalid points with valid shells.

```python
import numpy as np
import gc_turbulence as g
import scipy.ndimage as ndi
import scipy.interpolate as interp

run = g.SingleLayerRun(cache_path=g.default_cache + 'r13_12_17c.hdf5')
run.load()
pp = g.PreProcessor(run)
pp.extract_valid_region()
pp.filter_zeroes()
# pp.interpolate_zeroes()

invalid = np.isnan(pp.U)
coords = pp.Z, pp.X, pp.T
data = pp.U, pp.V, pp.W

# find the complete valid shell
invalid_with_shell = ndi.binary_dilation(invalid, structure=np.ones((3, 3, 3)))
complete_valid_shell = invalid_with_shell & ~invalid
# isolate disconnected volumes. A volume consists of an invalid core
# and a valid shell
volumes, n = ndi.label(invalid_with_shell)

all_coords = np.concatenate([c[None] for c in coords])

def fillin(volume):
    """Construct the interpolator for a single shell and evaluate
    inside."""
    # construct the interpolator for a single shell
    shell = volume & ~invalid
    inside_shell = volume & invalid

    valid_points = all_coords[..., shell].T
    # valid_values = np.vstack(d[shell] for d in data).T
    valid_values = pp.U[shell]

    # coordinates of all of the invalid points
    invalid_points = all_coords[..., inside_shell].T

    interpolator = interp.LinearNDInterpolator(valid_points, valid_values)
    # this is nan outside of the shell
    invalid_values = interpolator(invalid_points)

    pp.U[inside_shell] = invalid_values


def fillall(n=n):
    for i in range(1, n):
        print "# {} \r".format(i),
        sys.stdout.flush()
        fillin(volumes==i)
```

This works and would be fine if our arrays were smaller, but for the
sizes we are working with it isn't very fast. On the inner loop we
are finding the & of multiple 60 million element arrays, and then
indexing the coordinates with the boolean result.

It is much faster to index arrays with slice objects than with
boolean arrays. `scipy.nimage.find_objects` gives a list of slices
that isolate features in a nd array. We can apply it directly to the
output from `scipy.ndimage.label`.

This approach is a lot faster but it doesn't uniquely isolate
volumes: each slice is guaranteed to contain one feature, but it
could contain any amount of any number of others. We can't find the
coordinates of the points specifically inside the volume of interest
without losing the speed of the method.

If a volume that has already been processed  is partially contained
within a subsequent slice then we may not be able to calculate a
value from the interpolator.

In practice this means that we will have some nans leftover in the
data. We can get around this be running the process again, as the
number of nans should be drastically reduced in each iteration.

The method is this: we binary dilate the invalid data so that there
is some valid data around it. We then label this and get the slices
from find objects. For each slice we construct an interpolator from
the points that are in the valid shell and interpolate the points
that are invalid.

```python
import numpy as np
import gc_turbulence as g
import scipy.ndimage as ndi
import scipy.interpolate as interp

run = g.SingleLayerRun(cache_path=g.default_cache + 'r13_12_17c.hdf5')
run.load()
pp = g.PreProcessor(run)
pp.extract_valid_region()
pp.filter_zeroes()

invalid = np.isnan(pp.U)
coords = pp.Z, pp.X, pp.T
data = pp.U, pp.V, pp.W

invalid_with_shell = ndi.binary_dilation(invalid, structure=np.ones((3, 3, 3)))
complete_valid_shell = invalid_with_shell & ~invalid
labels, n = ndi.label(invalid_with_shell)
slices = ndi.find_objects(labels)

def interpolate_region(slice):
    nans = invalid[slice]
    shell = complete_valid_shell[slice]

    valid_points = np.vstack(c[slice][shell] for c in coords).T
    valid_values = pp.U[slice][shell]

    interpolator = interp.LinearNDInterpolator(valid_points, valid_values)

    invalid_points = np.vstack(c[slice][nans] for c in coords).T
    invalid_values = interpolator(invalid_points).astype(valid_values.dtype)

    ## PROBLEM: the slice might contain points that are outside of
    ## the shell. These points will be over written with nan
    ## regardless of their original state. We need to not overwrite
    ## these points if they have already been filled in by
    ## interpolation from another shell.
    good = ~np.isnan(invalid_values)
    # see https://stackoverflow.com/questions/7179532
    pp.U[slice].flat[np.flatnonzero(nans)[good]] = invalid_values[good]


def intall():
    for i, slice in enumerate(slices):
        print "# {} \r".format(i),
        sys.stdout.flush()
        interpolate_region(slice)

%time intall()
```

The `interpolate_region` method is by far the fastest. This took 15
mins for our test run. We could parallelise it, calculating the
invalid values in parallel and storing them in memory, then
assigning to the array when all are calculated.

This method still hangs on non convex regions, but not for as long.
There might be a way to reduce the hang by ignoring points that
dont' contribute much.


### Implementation

[This commit](https://github.com/aaren/lab_turbulence/commit/3b0c316e8bda32939f406a2fcbe4288416b09f80)

Usage:

```python
%matplotlib
import numpy as np
import matplotlib.pyplot as plt
import gc_turbulence as g

run = g.SingleLayerRun(cache_path=g.default_cache + 'r13_12_17c.hdf5')
run.load()
pp = g.PreProcessor(run)
pp.extract_valid_region()
pp.filter_zeroes()

# find the time index with the most invalid values
import scipy.stats as stats
mode_index, mode_count = stats.mode(np.where(np.isnan(pp.U))[-1])
idx = int(mode_index[0])

sample_2d = np.s_[..., idx]
us = pp.U[sample_2d]
xs = pp.X[sample_2d]
zs = pp.Z[sample_2d]

fig, axes = plt.subplots(ncols=3)
# plot the un treated data
axes[0].contourf(xs, zs, us, 50)

sub=np.s_[:, :, idx - 100: idx + 100]

# plot the interpolated data
pp.interpolate_nan(sub_region=sub, scale=1)
axes[1].contourf(xs, zs, us, 50)

# re-extract and interpolate with different time scaled with front
# speed
pp.extract_valid_region()
pp.filter_zeroes()

pp.interpolate_nan(sub_region=sub, scale='auto')
axes[2].contourf(xs, zs, us, 50)
plt.draw()
```


### Validation

Is our method actually working?

We could take some complete data (no nans) and set some of it equal to
nan, then apply the interpolation above and see how close we get to
the actual values.



### Extension

The obvious way to improve this method is to upgrade the
interpolator to something more fancy than a linear method.
For example, we could use a radial basis function:
`scipy.interpolate.rbf`. The linear interpolator works fine though.
