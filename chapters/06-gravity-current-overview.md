---
layout: chapter
title: Turbulence data overview
---

Estimating the pdf for a gravity current
========================================

We want to look at the probability distribution function of the
velocities inside a gravity current over time.

I had a go at doing this and went on a journey from line plots to
kernel density estimation.


### Summary

These plots show the evolution of vertical velocity as a function of
time behind the gravity current front passage, for various different
ways of assessing the probability density function:

```python
## RUN THIS LAST!!
display_all(outputs)
```

### Setup

```python
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

import gc_turbulence as g

plt.rc('figure', figsize=(15, 5))
```

```python
r = g.ProcessedRun(cache_path=g.default_processed + 'r13_12_16e.hdf5', forced_load=True)
```

### Plots

**A single point in height and horizontal, over front relative time:**

```python
def plot_single_run(what, *args, **kwargs):
    fig, ax = plt.subplots()
    data = r.W_[what]
    t = r.T_[what]
    ax.plot(t.T, data.T, *args, **kwargs)
    ax.set_xlabel('time after front passage')
    ax.set_ylabel('vertical velocity')
    ax.set_ylim(-0.03, 0.04)
    return ax
```
```python
iz = 30
ix = 50
single_point = np.s_[iz, ix, :]

single_run_single_point_plot = plot_single_run(single_point, 'k')
```

**All points at a particular height, over front relative time:**

```python
multi_point = np.s_[iz, :, :]

single_run_plot = plot_single_run(multi_point, 'k', linewidth=0.1)
```

**All points from runs with the same parameters at a particular height,
over front relative time:**


```python
params = g.Parameters()

conditions = dict(H=0.25, L=0.25, D=0.25, rho_ambient=1.0047)

which = [params.single_layer[k] == v for k, v in conditions.items()]
where = np.all(which, axis=0)
runs = params.single_layer['run_index'][where]
print runs

h5names = [g.default_processed + r + '.hdf5' for r in runs]
processed_runs = [g.ProcessedRun(cache_path=h5, forced_load=True) for h5 in h5names]
```
```python
fig, ax = plt.subplots()
colors = ('k', 'r', 'b', 'g', 'y')
for i, r in enumerate(processed_runs):
    data = r.W_[30, :, :]
    t = r.T_[30, :, :]
    ax.plot(t.T, data.T, color=colors[i], linewidth=0.1, alpha=0.3)

ax.set_xlabel('time after front passage')
ax.set_ylabel('vertical velocity')
ax.set_ylim(-0.03, 0.04)

multi_run_plot = ax
```

This last plot already looks a bit like a pdf. We want to formalise
this and generate the pdf using the same axes, with colour
indicating the density of points.

Let's work with a single run for now. First look is to scatter plot:

```python
r = g.ProcessedRun(cache_path=g.default_processed + 'r13_12_16e.hdf5', forced_load=True)
scatter_plot = plot_single_run(multi_point, 'k.', alpha=0.02)
```

That doesn't tell us much except that the vertical velocity remains
within fairly sharp bounds.


### Histogram

We'll get a better look with a 2d histogram:

```python
xbins = np.linspace(-5, 20, 1000)
ybins = np.linspace(-0.03, 0.04, 100)

def plot_histogram(data, bins=(xbins, ybins), where=np.s_[:], **kwargs):
    H, edges = np.histogramdd(data, bins=bins, normed=True)

    # hide empty bins
    Hmasked = np.ma.masked_where(H==0, H)
    xedges, yedges = edges[:2]
    if not 'levels' in kwargs:
        kwargs['levels'] = np.linspace(0, 10)

    fig, ax = plt.subplots()
    ax.contourf(xedges[1:], yedges[1:], Hmasked.T[where], **kwargs)
    ax.set_xlabel('time after front passage')
    ax.set_ylabel('vertical velocity')
    return ax
```
```python
data = r.W_[multi_point].flatten()
time = r.T_[multi_point].flatten()

single_run_histogram = plot_histogram((time, data))
```

We can compute the histogram over multiple runs:

```python
data_multi = np.hstack((r.W_[multi_point] for r in processed_runs)).flatten()
time_multi = np.hstack((r.T_[multi_point] for r in processed_runs)).flatten()

multi_run_histogram = plot_histogram((time_multi, data_multi))
```

We can see that the distribution is dominated by individual events.

My previous method for making this plot was to make a 1d histogram
of each time slice and then stack them all together. This was very
slow. We recover my naive approach when the bin width is set to the
time interval that a single time slice covers.

Perhaps we should use a n-dimensional histogram (`np.histogramdd`)
on the data over all heights?  This doesn't actually take that much
longer, the main overhead being in pulling the data from disk. This
also lets us plot the pdf for any number of heights from a single
run whilst only doing the computation once.

We can do this and recreate the plot above:

```python
w = r.W_[:].flatten()
t = r.T_[:].flatten()
z = r.Z_[:].flatten()

zbins = np.linspace(0, 0.1, 50)

multi_dim_histogram = plot_histogram((t, w, z), bins=(xbins, ybins, zbins),
                                     where=20, levels=np.linspace(0, 100))
```

All that histogramming does is put a grid over the data and count
the number of data points inside each grid box. We are just
digitising the continuous field of data. The histogram approximates
a pdf, but it is not neccesarily a good estimator for one.

A problem with making histograms is that we don't always know what
to set the bin width at. For our coordinate variables x, z, t it is
easy - we just make sure that the bins are wider than the sampling
interval.

However for our random variables u, v, w it is more difficult as
they are not uniformly distributed. It might be the case that our
pdf will approach a smooth distribution in the limit of many
ensembles, but for small numbers of ensembles we are dominated by
individual events which we do not know how to bin.


### Kernel Density Estimation

http://stackoverflow.com/questions/21918529/multivariate-kernel-density-estimation-in-python

http://jakevdp.github.io/blog/2013/12/01/kernel-density-estimation/

http://scikit-learn.org/stable/modules/density.html

A solution to the problem of rigid bin widths is *Kernel Density
Estimation* (KDE). A simple KDE is to place a box of width w around
each data point, with the data point at the centre. We then sum all
the boxes to get the pdf.

KDE works well when we don't know what the range of the bins should
be (i.e. most of the time) but it is still dependent on w - the
*bandwidth*, which has to be chosen carefully.

Too large or small a bandwidth will lead to over or under smoothing,
which can lead to radically different indications of the underlying
pdf.

Here, we might choose a bandwidth of 0.001 for w and 0.05 for t. Not
all implementations of KDE allow us to specify a bandwidth for
multiple variables - we can get around this by scaling one of the
axes such that the bandwidth applies to both variables.

The shape of the box (the kernel) that we place at each data point
can be altered as well. A simple choice is a tophat function, but we
could use anything. A natural choice is a Gaussian. This is
complicated in more than one dimension as each dimension might have
a different optimal kernel shape and a different optimal bandwidth!

It is important to consider what it is that we want the estimation
to give us. The KDE can fill in gaps in our observed data. Our data
contains natural gaps due to the occurence of discrete events - the
large scale eddies in the turbulence. We don't expect that these
events are localised in space, so in many realisations of a gravity
current we would expect a smooth distribution.

The main problem with KDE compared to making a histogram is that a
naive implementation can be *really slow* ($N^2$ time).

Here's how we set it up, with a plotting function:

```python
# can't have multi dim bandwidth so we can scale the t here
# and undo at the end. This might be equivalent to changing the
# metric of the kd-tree but I'm not sure.
w_bandwidth = 0.002
t_bandwidth = 0.05
# scaling to use on t to make bandwidth applicable
ts = w_bandwidth / t_bandwidth
bandwidth = w_bandwidth


# extract some data and rescale it
w = r.W_[30, :, :].flatten()
t = r.T_[30, :, :].flatten() * ts
data = np.vstack((w, t))

# construct regular grid of coordinates to evaluate at
gt = xbins * ts
gw = ybins
GW, GT = np.meshgrid(gw, gt)
coords = np.vstack((GW.flatten(), GT.flatten()))

def plot_kde(pdf, levels=np.linspace(0, 150)):
    """Contour plot over given levels."""
    fig, ax = plt.subplots()
    # mask equivalent to where the histogram has empty bins
    pdf = np.ma.masked_where(pdf < 1, pdf)
    ax.contourf(GT / ts, GW, pdf.reshape(GT.shape), levels=levels)
    ax.set_xlabel('time after front passage')
    ax.set_ylabel('vertical velocity')
    return ax
```

Scipy has a built in gaussian KDE, which takes a nice long time
O(N^2) with our data:

```python
from scipy import stats

scipy_kde = stats.gaussian_kde(data, bw_method=bandwidth)
%time scipy_pdf = scipy_kde.evaluate(coords)
scipy_kde_plot = plot_kde(scipy_pdf)
```

I'm not sure why it looks so rubbish - I suspect something to do
with the bandwidth choice. Also, this method somehow fully utilises
10 cores (of a 24 core machine) without me telling it to (look at
the timing output)- this happens when run in a .py script as well.
It doesn't really matter because there is a much more efficient way
to compute the KDE:


#### Tree-based KDE

We can get a significant speed up in the computation by making use
of a KD-Tree of the data points. With some tolerance on the
precision of our estimate we then evaluate the kernel sum only over
points that are geometrically close. This is what the implementation
in Scikit-learn does:

```python
from sklearn.neighbors import KernelDensity

# set rtol to allow 0.1% error
kde_sk = KernelDensity(bandwidth=bandwidth, rtol=1E-3)
%time kde_sk.fit(data.T)
%time log_pdf = kde_sk.score_samples(coords.T)
pdf_sk = np.exp(log_pdf)
sklearn_kde_plot = plot_kde(pdf_sk)
```

We can do this over multiple runs:

```python
## this takes a long time. like half an hour long.
w = np.hstack((r.W_[multi_point] for r in processed_runs)).flatten()
t = np.hstack((r.T_[multi_point] for r in processed_runs)).flatten() * ts
data = np.vstack((w, t))

%time kde_sk.fit(data.T)
%time log_pdf_multi = kde_sk.score_samples(coords.T)
pdf_sk_multi = np.exp(log_pdf_multi)
sklearn_kde_plot_multi = plot_kde(pdf_sk_multi)
```


-------------------------------------------------------------------------------

### Performance

We've been working with a subset of data from a single run. Here's
the number of points that we might be working with:

```python
print "Size of our example data: ", r.W_[30, :, :].size
print "Size of a single component of a full run: ", r.W_[:, :, :].size

components = (r.U_, r.V_, r.W_, r.X_, r.Z_, r.T_)
print "potential number of dimensions: ", len(components)
print "potential number of points from 10 runs: ", r.W_.size * len(components) * 10
```

This might take some time to execute! A way of saving the generated
models might be useful. Luckily, we can [save our model][persistence] 
using `pickle` if we need to.

[persistence]: http://scikit-learn.org/stable/tutorial/basic/tutorial.html#model-persistence

The tree-based KDE computation is sensitive to the kernel bandwidth.
The performance benefit of the tree approach comes from only
considering training points that are close to a query point (the
'training' points are the original data; the 'query' points are
where we evaluate the kde). What 'close' means is going to vary with
varying kernel width.

We've used a kernel bandwidth of 0.05 in time and 0.002 in vertical
velocity. The time sampling interval is 0.01. Vertical velocities
exist in the range [-0.03, 0.04] 


-------------------------------------------------------------------------------

### Bandwidth selection

We can search for the best estimator for our data across a parameter
space of inputs. As our estimator is only dependent on the bandwidth
and we have scaled our axes such that this is the samme in both
dimensions, we need to search over a 1d array of inputs. 

We optimise with the score of the estimator, which for our KDE is
the integrated log probability of the distribution model.

If we fit (*train*) an estimator to all of the data, we can easily
tune the bandwidth until we get the best score. However, this
estimator would not be useful on new data - this is *overfitting*.
Since we have a vast quantity of data that we cannot hope to train
an estimator on in its entireity we need to find a way to overcome
this problem.

Fitting using Cross Validation is a way to overcome this. The
simplest version (leave-one-out) is to split the data into $k$
folds, train the estimator on $k - 1$ of them and validate on the
remainder. We do this over all permutations and take best scoring
estimator.

Scikit-learn implements a cross validation grid search:

```python
from sklearn.grid_search import GridSearchCV

grid = GridSearchCV(KernelDensity(rtol=1E-4),  # allow 0.01% errors
                   {'bandwidth': np.logspace(-4, -2, 20)},
                   cv=5,       # k, number of folds
                   n_jobs=20)  # multi core!

# take some subset of the data and make it (N, d) in shape
data = r.W_[30, :, 1000:1200].flatten()[None].T
# search for the optimum estimator for the data over the grid:
%time grid.fit(data)
print "Optimum bandwidth 1E-4 error: ", grid.best_params_
# see how higher allowable error changes the bandwidth:
grid.rtol = 1E-2
%time grid.fit(data)
print "Optimum bandwidth 1E-2 error: ", grid.best_params_
```

-------------------------------------------------------------------------------

### Bayesian blocks

It is worth mentioning another approach, based on applying a
variable bin width across the data. The Bayesian Blocks algorithm
computes a fitness function that depends only on the width of each
bin and the number of points in it. We then evaluate the fitness
function over all bin combinations and select the best one.

The advantage of this method is that it will select the
quantitatively best set of bin widths for the data.

The problem with this method is that the computation time scales
with the number of points as $2^N$. Scargle showed that this time
can be reduced to $N^2$ whilst still guranteeing the best choice.
This would be competitive with the scipy kde above - however, a
mature multi-dimensional implementation does not exist (but there
are ideas using Voronoi cells), making this unsuitable for our
problem.

http://jakevdp.github.io/blog/2012/09/12/dynamic-programming-in-python/

That said here is an example using an implementation in astroML:

```python
from astroML.plotting import hist

fig, axes = plt.subplots(nrows=2)

# consider a sub sample of the data
data = r.W_[30, :, 1500:1750].flatten()

h0 = axes[0].hist(data, bins=np.linspace(-0.03, 0.04, 100), normed=True)
h1 = hist(data, bins='blocks', ax=axes[1], normed=True)

axes[0].set_xlim(-0.03, 0.04)
axes[0].set_title('regular bins')
axes[1].set_xlim(-0.03, 0.04)
axes[1].set_title('bayesian blocks')

fig.tight_layout()
```

This method might be suitable for considering the distribution of
subsets of our data. For example, we might wish to see whether the
velocity data is distributed in a different way in different
sections of the flow.

-------------------------------------------------------------------------------

How we display the figures at the top:

```python
from IPython.display import display

outputs = [(single_run_single_point_plot, 'one run, single point'),
           (single_run_plot, 'one run, multiple points'),
           (single_run_histogram, 'one run, histogram'),
           (sklearn_kde_plot, 'one run, kernel density estimate'),
           (multi_run_plot, 'multiple runs'),
           (multi_run_histogram, 'multiple runs histogram'),
           (sklearn_kde_plot_multi, 'multiple runs kde')]

def display_all(outputs):
    for i, (plot, title) in enumerate(outputs):
        plot.set_title(title)
        plot.figure.set_size_inches((12, 2))
        plot.figure.tight_layout()
        display(plot.figure)
```
