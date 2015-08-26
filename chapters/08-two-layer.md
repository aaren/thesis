---
layout: chapter
title: The two layer fluid
---

# Gravity currents in a two layer fluid

```python
from __future__ import division
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

import scipy.interpolate as interp
import scipy.ndimage as ndi
```

```python

```

```python
from wavelets import WaveletTransform, Ricker, Morlet, Paul

from labwaves import Run
```

```python
okruns = [
    'r13_01_08a',
    'r13_01_08c',
    'r13_01_08d',
    'r13_01_09a',
    'r13_01_11b',
    'r13_01_11c',
    'r13_01_12a',
]

runs = [Run(index) for index in okruns]
```

## Introduction

We are considering the release of a gravity current into a two layer
fluid in a lock release configuration.

@fig:example-wavefield shows the non-dimensionalised
interfacial wavefield from one of these experiments in a hovmoller
plot.

```python
r = runs[2]

plt.contourf(r.waves.x, r.waves.t, r.waves.z, cmap=plt.cm.bone_r, levels=np.linspace(0.45, 0.65))
plt.xlabel('distance')
plt.ylabel('time')
plt.colorbar()
```

We clearly see the evolution of an amplitude ordered series of waves
following the release of the lock gate at $t=0$.

We have performed a number of experiments. The wavefields for each
run can be seen in @fig:all-wavefields.

*[Aaron]: many more runs to consider but they need a little
straightening up first.*

```python
fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(6, 6))

for r, ax in zip(runs, axes.flatten()):
    ax.contourf(r.waves.x, r.waves.t, r.waves.z, 50, cmap=plt.cm.bone)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 60)

fig.tight_layout()
```

We wish to investigate the properties of these wavefields. The
motivation for this study is based in the field - in the analysis of
the effects of storms on the atmosphere. Therefore the analysis here
will use techniques that can be extended to field data. In
particular we expect that the techniques used will be effective in
the presence of noise and other confounding factors.


## Wave detection

We want to identify the individual waves that we can see in the wave
fields. There are many ways that we could do this but we are looking
for a robust method.

The defining property of our waves is that they are of constant
speed and constant form, i.e. they form straight lines that cover
most of the field.

Detection of such features is efficiently performed by the ridgelet
transform.

### The Ridgelet Transform

The *Ridgelet Transform* is conceptually similar to the
one-dimensional wavelet transform. Instead of a one-dimensional
wavelet we use a two-dimensional ridge that forms a wavelet in cross
section. Instead of convolving over a single axis, we convolve this
ridge over our data in multiple directions.

The ridgelet transform is effective at finding straight, ridge-like
structures in data, with the caveat that they must be comparable to
the data field in length.

The easiest way to perform a ridgelet transform is to perform a 1D
wavelet transform over the position axis of the *Radon Transform*.

The radon transform is computed by calculating a line integral
through our data, whilst varying the position of the line and
rotating the data field.

The radon transform has two axes: position and angle. The position
is that of a straight line that transects our data and the angle is
what the data has been rotated to.


It is necessary to perform the transform over a circle in the data,
created by extracting a square and zeroing all the points outside
the circle that fits inside.

If this zeroing is not performed there is a background artefact in
the radon transform (a cross), formed because the line integral over
a square of constant value varies with the position of the line.
This artefact forms a cross with its centre and peak at 45 degrees
rotation and half way along the position axis, as shown in
@fig:radon-circle.

```python
from skimage.transform import radon

time = np.linspace(0, 60, 300)
space = np.linspace(4, 12, 300)

# square the wave data
data = runs[1].waves(space, time)
unzeroed_data = data.copy()

# zero the data outside of a circle
i, j = np.indices(data.shape)
i_ = i - i.max() / 2
j_ = j - j.max() / 2

outside_circle = (i_ ** 2 + j_ ** 2) ** .5 + 2 > i.max() / 2
data[outside_circle] = 0

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(5, 5))

sinogram = radon(data, theta=np.linspace(0, 90, 90), circle=True)
unzeroed_sinogram = radon(unzeroed_data, theta=np.linspace(0, 90, 90), circle=False)

axes[0, 0].contourf(data, 50)
axes[0, 1].contourf(unzeroed_data, 50)
axes[1, 0].contourf(sinogram, 50)
axes[1, 1].contourf(unzeroed_sinogram[50:350], 50)

fig.tight_layout()
```

In @fig:radon-circle we can see the individual waves represented as
peaks in the radon space $(r, \theta)$. Identifying individual waves
is as simple as distinguishing these peaks. The full ridgelet
transform makes this even easier.

The ridgelet transform has three axes: position, angle and scale.
The first two are in common with the radon transform $(r, \theta)$
and the last is as in the wavelet transform. By selecting a subset
of scales we effectively band-pass filter our search, selecting only
features within a particular scale range.

We have some a priori knowledge about our wave fields that help us
constrain the search for waves: (i) the waves exist within a certain
range of speeds; (ii) the wavelength of the waves is roughly
bounded. This knowledge limits the angles (speed) and scales
(wavelength) that we need to search over.

```python
def ridgelet_transform(run, theta=None, scales=None, wavelet=Morlet(),
                       space=np.linspace(4, 12, 300), time=np.linspace(0, 60, 300)):
    """Produce the ridgelet transform of a given run, using
    a given wavelet as the shape of the ridge.
    """
    if theta is None:
        theta = np.linspace(0, 180, 101, endpoint=True)

    if scales is None:
        scales = np.logspace(0, 1, 100)

    # square the wave data
    data = run.waves(space, time)

    # zero the data outside of a circle
    i, j = np.indices(data.shape)
    i_ = i - i.max() / 2
    j_ = j - j.max() / 2

    outside_circle = (i_ ** 2 + j_ ** 2) ** .5 + 1 > i.max() / 2
    data[outside_circle] = 0

    # perform the radon transform
    sinogram = radon(data, theta=theta, circle=True)

    # wavelet transform the sinogram over the position axis
    # to compute the ridgelet transform
    wavelet_transform = WaveletTransform(sinogram, axis=0, wavelet=wavelet, unbias=True)
    wavelet_transform.dj = 0.1
    #wavelet_transform.scales = scales
    ridgelet = wavelet_transform.wavelet_transform
    return ridgelet
```

For example, consider the ridgelet transform of the run in
@fig:example-wavefield.

```python
theta = np.linspace(20, 90, 150, endpoint=True)

rt = ridgelet_transform(r, theta=theta, wavelet=Ricker())
```

The ridgelet transform, summed over a subset of the scales, is shown
in @fig:ridgelet-example.

```python
position = np.arange(rt.shape[1])
position_slice = slice(20, 250)
#position_slice = slice(0, None)

Theta, Position = np.meshgrid(theta, position[position_slice])
plt.contourf(Theta, Position, (rt ** 2).real[10:30, position_slice].sum(axis=0), 50)
plt.xlabel('Angle, degrees')
plt.ylabel('Position, pixels')
```

The wavelet filtering has vastly improved the signal of the peaks
and gives us a signature wave pattern for this particular run.

The peaks of the transform indicate an angle and position of a ridge
in the data. The angle is relative to the frame of the data but the
position is along the radon axis. These coordinates can be converted
into the gradient and intercept of a straight line through the data.

```python
def radon_to_square(r, theta, width):
    """Convert a point in the radon transform into a
    line in the pixel space of the input to the transform.

    r - radon position
    theta - radon angle, degrees
    width - width of input array

    returns:

    [m, c] - to construct y = m * x + c

    Assumes that the transform has been performed around the
    central point of a square 2d array, which transforms to
    the centre of the radon position.
    """
    # use radians
    theta = np.deg2rad(theta)

    xc = width / 2.
    yc = width / 2.

    # centre of the radon axis
    rc = (2 * width ** 2) ** .5 / 2
    rc = width / 2

    # determine point in x, y that is in the centre of the
    # line integral
    x = xc + (r - rc) * np.cos(theta)
    y = yc - (r - rc) * np.sin(theta)
    
    # determine the gradient
    m = 1 / np.tan(theta)

    # determine the y intercept
    c = y - x * m

    return m, c


def square_to_real(m, c, x, y):
    """Convert gradient and y-intercept from square pixel coords into
    real data coords, given the axes used to interpolate the square
    array.

    m - gradient (pixels / pixel)
    c - y-intercept (pixels)
    x - x array
    y - y array

    where

        square_array = r.waves(x, y)
    """
    dx = np.diff(x).mean()
    dy = np.diff(y).mean()

    mr = m * dy / dx
    cr = y[c] - mr * x[0]

    return mr, cr
```

#### Ridgelet reconstruction

As an aside, the ridgelet transform can be used to filter and reconstruct the
original input. As with the wavelet reconstruction, we can reconstruct given a
subset of the scales:

```python
from skimage.transform import iradon

recon = iradon(rt[20:50].sum(axis=0), theta=theta, circle=False, output_size=300)

interior = (slice(50, 250), slice(50, 250))
fig, axes = plt.subplots(nrows=2)
axes[0].contourf(recon[interior], 100)
axes[1].contourf(r.waves(space, time)[interior], 100)
```

In this way we reconstruct a synthetic wave-field with no noise. Whilst this may
look convincing we must remember that it is the wave-field as seen by the
wavelet function: if the shape of our wavelet function does not exactly match
the shape of our waves then we can expect our reconstruction to be off. Here we
have excluded some of the scales used to compute the transform and the
reconstruction is not absolute.

We could equally well perform the reconstruction using a subset of the
projection angles, seeking structures within a range of speeds.



#### Wavelet selection

```python
all_transforms = [[ridgelet_transform(r, theta=theta, wavelet=W()) for W in Ricker, Morlet, Paul] for r in runs]
```

```python
fig, axes = plt.subplots(nrows=len(runs), ncols=4, figsize=(12, 8))

for i, transforms in enumerate(all_transforms):
    for j, transform in enumerate(transforms + [0]):
        if j != 3:
            axes[i, j].contourf(Theta, Position, (transform ** 2).real[20:50, position_slice].sum(axis=0), 50)
        else:
            axes[i, j].imshow(runs[i].waves(space, time), origin='lowerleft')
        if j != 0:
                axes[i, j].set_yticks([])
        if i != 6:
            axes[i, j].set_xticks([])

fig.tight_layout()
```

*Fixme: problem with the above is that the scales aren't directly comparable
between the different wavelets. Need to use equivalent fourier
period*

The comparison of ridgelet transforms in @fig:ridgelet-wavelet-comparison
confirms that the Ricker wavelet is the one to use when searching for peaks.

The Ricker-based ridgelet transform of each of the runs is show in
@fig:ricker-ridgelets.

```python
ricker_ridgelets = [ridgelet_transform(r, theta=theta, wavelet=Ricker()) for r in runs]
```

```python
fig, axes = plt.subplots(ncols=len(runs))

for ax, ridge in zip(axes, ricker_ridgelets):
    ax.contourf(ndi.filters.gaussian_filter(ridge[10:30, position_slice, 60:140].real.sum(axis=0) ** 2, sigma=3), 50)
    ax.set_yticks([])
    ax.set_xticks([])
```

### Peak finding

We can search for the maxima using a local maximum filter. The difficulty with
doing this in two dimensions is that there are many small peaks that would be
picked up. We can set a peak size criterion but there isn't a robust way of
doing it - we will have difficulty detecting the smaller peaks with anything
sure to filter out noisy peaks.

```python
#ridge = ricker_ridgelets[3]
ridge = ridgelet_transform(runs[3], theta=theta, wavelet=Ricker())

descaled = ridge.real[10:30, 20:-20, 60:140].sum(axis=0)
smoothed = ndi.gaussian_filter(descaled, sigma=2, mode='constant')

maxima_2d = ndi.maximum_filter(smoothed, 10) == smoothed
filtered_maxima_2d = (smoothed > 3 * smoothed[smoothed > 0].mean()) & maxima_2d
position_max, angle_max = np.where(filtered_maxima_2d)

plt.contourf(smoothed.T, 50)
plt.plot(position_max, angle_max, 'o')
```

We can simplify our peak finding problem by making an assumption about the form
of the data. We are looking for amplitude ordered solitary waves. These travel
with constant speed that decreases with amplitude. Barring reflections in the
tank we do not expect these waves to collide and they should take the form of
separated straight lines in $(x, t)$. Inspection of the wave-fields suggest that
this is the case.

Lines separated in $(x, t)$ will be ordered along $r$ in the radon space $(r,
\theta)$. This is what we see in our ridgelet transforms. Therefore, instead of
finding the maxima in the 2d space, we need only find the maxima along $r$ as
this is enough to uniquely identify a wave.

```python
fig, axes = plt.subplots(nrows=2)
#axes[0].plot(descaled.mean(axis=1))
axes[0].plot(smoothed.mean(axis=1))
axes[0].set_xlim(0, smoothed.shape[0])

axes[1].contourf(smoothed.T, 50)
```

```python
signal = smoothed.mean(axis=1)
x = np.arange(signal.shape[0])
maxima = signal == ndi.maximum_filter1d(signal, 10)
minima = signal == ndi.minimum_filter1d(signal, 10)
plt.plot(x, signal)
#plt.plot(x[np.where(maxima)], signal[np.where(maxima)], 'o')
#plt.plot(x[np.where(minima)], signal[np.where(minima)], 'o')

plt.axhline(signal.mean())

# use the biggest peak to identify the first and remove
# all the extrema before it
extrema = signal ** 2 == ndi.maximum_filter1d(signal ** 2, 10)
first = np.abs(signal * extrema).argmax()
maxima[first + 1:] = False
minima[first + 1:] = False
plt.plot(x[np.where(maxima)], signal[np.where(maxima)], 'o')
plt.plot(x[np.where(minima)], signal[np.where(minima)], 'o')
```

We have reduced our wave detection problem to that of peak detection of a strong
signal in a single dimension.

Again, we must be careful in our interpretation of the peaks: what we are seeing
is the wavelet function convolved with the sinogram. The wavelet function
necessarily has zero mean - it cannot be singly peaked. In particular at the
leading edge of the wave packet we can expect to see a small spurious minima
before the proper maxima (or vice-versa if the first wave is a depression). We
remove these spurious peaks by ignoring all points ahead of the absolute
extremal value, making the assumption that the largest amplitude
wave is the first wave in the series.

With the positions ($r$) of the extrema found, we find the
corresponding angles ($\theta$) by looking for the maximum at the
given $r$.

```python
angle_max = smoothed[maxima].argmax(axis=1)
position_max = np.where(maxima)[0]
plt.contourf(smoothed.T, 50)
plt.plot(position_max, angle_max, 'o')
```

These locations are approximate as we have assumed that the peaks in the mean of
the array across angles coincide with the actual peaks in the 2d array. This is
a reasonable approximation given that we are assuming the peaks are well
separated in position, but isn't exactly right.

To find the best guess for the location of the extrema we find the nearest-
neighbours in the 2d extrema to our 1d signal detected extrema. A particularly
efficient way to do this is to use a KD-tree:

```python
# zero in on the maxima location
# all local maxima
max_2d = ndi.maximum_filter(smoothed, size=10) == smoothed

all_max = np.array(np.where(max_2d)).T
found_max = np.array((position_max, angle_max)).T

# A kd-tree finds the nearest points to a given set
from scipy.spatial import KDTree
tree = KDTree(all_max)
abs_max_distance, abs_max_indices = tree.query(found_max)
abs_max = all_max[abs_max_indices]

plt.figure()
plt.contourf(smoothed.T, 50)
plt.plot(*abs_max.T, marker='o')
```

### Summary

```python
from skimage.transform import radon

from wavelets import Ricker

def detect_waves(run, index=False, tlim=(0, 60), xlim=(0, 12), res=300, velocity_range=None, ransac_filter=False):
    """Detect waves in a given run, returning a list of
    the gradients and y-intercepts of the waves in either
    index or real space (toggled by index).
    """
    # square the data
    time = np.linspace(*tlim, num=res)
    space = np.linspace(*xlim, num=res)

    if velocity_range is None:
        theta = np.linspace(20, 90, 150, endpoint=True)
    else:
        # set the angular range based on the speed
        vmin, vmax = velocity_range
        dx = np.diff(space)[0]
        dt = np.diff(time)[0]
        theta_min = np.degrees(np.arctan(vmin / (dx / dt)))
        theta_max = np.degrees(np.arctan(vmax / (dx / dt)))
        theta = np.linspace(theta_min, theta_max, 150)


    # calculate the ridgelet transform
    transform = ridgelet_transform(run, wavelet=Ricker(), theta=theta, time=time, space=space)

    # sum over a subset of scales and ignore extremes of position
    position_offset = 20
    filtered_sinogram = transform[10:30, position_offset:-position_offset].real.sum(axis=0)
    smoothed_sinogram = ndi.gaussian_filter(filtered_sinogram, sigma=5, mode='constant')

    # approx maxima detection
    signal = smoothed_sinogram.mean(axis=1)
    x = np.arange(signal.shape[0])
    maxima = signal == ndi.maximum_filter1d(signal, 10)
    minima = signal == ndi.minimum_filter1d(signal, 10)

    # use the largest peak to determine the first one
    # and ignore any peaks that come before it
    extrema = signal ** 2 == ndi.maximum_filter1d(signal ** 2, 10)
    first = np.abs(signal * extrema).argmax()
    maxima[first + 1:] = False
    minima[first + 1:] = False

    # now find the angles
    angle_max = smoothed_sinogram[maxima].argmax(axis=1)
    position_max = np.where(maxima)[0]

    # kdtree refinement
    max2d = ndi.maximum_filter(smoothed_sinogram, size=10) == smoothed_sinogram

    all_max = np.array(np.where(max2d)).T
    found_max = np.array((position_max, angle_max)).T

    # overkill! A kd-tree finds the nearest points to a given set
    from scipy.spatial import KDTree
    tree = KDTree(all_max)
    abs_max_distance, abs_max_indices = tree.query(found_max)
    refined_maxima = all_max[abs_max_indices]

    if ransac_filter:
        # now use RANSAC to get rid of any outliers
        from sklearn.linear_model import RANSACRegressor, LinearRegression
        ransac = RANSACRegressor(LinearRegression())
        x, y = refined_maxima.T
        ransac.fit(x[:, None], y)
        inliers = ransac.inlier_mask_
        filtered_maxima = refined_maxima[inliers]
    else:
        filtered_maxima = refined_maxima

    # convert index based maxima into (r, theta) coordinates
    angles = theta[filtered_maxima.T[1]]
    positions = filtered_maxima.T[0] + position_offset

    # convert into gradients / intercepts in index space
    detected_waves = [radon_to_square(pos, angle, width=time.size)
                            for pos, angle in zip(positions, angles)][::-1]


    #plt.contourf(smoothed_sinogram.T, 100)
    #plt.plot(signal)
    #plt.plot(*refined_maxima.T, marker='o')
    #plt.plot(*filtered_maxima.T, marker='o')
    #plt.plot(np.arange(signal.size)[refined_maxima.T[0]], signal[refined_maxima.T[0]], 'o')

    if index:
        return detected_waves

    elif not index:
        # convert into gradients / intercepts in real space
        detected_waves = [square_to_real(m, c, x=space, y=time) for m, c in detected_waves]
        detected_waves.sort(key=lambda x: x[0] * xlim[1] + x[1])
        return detected_waves
```

The results of wave detection using the above method are shown in
@fig:all-detection.

```python
def plot_waves(r, ax, xlim, tlim, res):

    x = np.linspace(*xlim, num=res)
    t = np.linspace(*tlim, num=res)

    waves_real = detect_waves(r, index=False, xlim=xlim, tlim=tlim, res=res, velocity_range=(0.2, 1))

    if not ax:
        fig, ax = plt.subplots()

    cax = ax.contourf(x, t, r.waves(x, t), levels=np.linspace(0.1, 0.3), cmap=plt.cm.bone_r)

    for i, (m, c) in enumerate(waves_real):
        wt = x * m + c
        ax.plot(x, wt, 'k', alpha=0.6, linewidth=2)
        mx = xlim[0] + np.diff(xlim)[0] / 2
        my = mx * m + c
        ax.text(mx, my, i)

    ax.set_ylim(*tlim)
    ax.set_xlim(*xlim)

    ax.set_yticks([])
    ax.set_xticks([])

    return ax

fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(6, 6))

xlim = (6, 12)
tlim = (0, 60)
res = 300

for r, ax in zip(runs, axes.flatten()):
    plot_waves(r, ax, xlim=xlim, tlim=tlim, res=res)

fig.tight_layout()
```

```python
for r in runs:
    r.wave_lines = detect_waves(r, xlim=xlim, tlim=tlim, res=res, velocity_range=(0.2, 1))
```

```python
x = np.linspace(10, 12, 100)

def wave_trace(wave_line):
    m, c = wave_line
    return x, x * m + c

for r in runs:
    r.wave_traces = [wave_trace(wave_line) for wave_line in r.wave_lines]

# look at interval [6, 12], [0, 60]
for r in runs:
    r.amplitudes = [r.waves(x[t< 80], t[t < 80]).diagonal().mean() - r.waves.z[:10, :].mean() for x, t in r.wave_traces]
    r.speeds = [1 / wave_line[0] for wave_line in r.wave_lines]
```

## Results

@fig:speed-vs-amplitude shows wave speed against wave
amplitude for each wave detected in a run and for each run.

```python
for i, r in enumerate(runs):
    plt.plot(r.speeds, r.amplitudes, '-o')
    plt.text(r.speeds[0], r.amplitudes[0], i)
    plt.axvline(0.5, color='k')
    plt.axvline((r.h1 * (1 - r.h1)) ** .5)

plt.ylim(-0.1, 0.2)

plt.xlabel('Speed')
plt.ylabel('Amplitude')
```

```python
background = (-r.parameters['h_1'] + r.waves.z[:10, :]).mean(axis=0)

plt.contourf(r.waves.z - background, 100)

plt.plot(background * 1000)
plt.colorbar()
```

```python
plt.figure(figsize=(12, 3))
plt.contourf(im.wave_fluid, 100)
x, y = im.wave_interface
plt.plot(x, y, 'r.')
plt.ylim(200, 150)

#plt.colorbar()
```

```python
def nonlinear_coeffs(n0, h1):
    h2 = 1 - h1
    c = (h1 * h2) ** .5
    alpha = (3 * c * (h2 - h1)) / (2 * h1 * h2)
    beta = c * h1 * h2 / 6

    V = c + alpha * n0 / 3
    D = (12 * beta / (alpha * n0))
    return c, V, D
```

```python
# amplitudes = (0.02, 0.036, 0.009, 0.08, 0.012, 0.028, 0.03)
amplitudes = [r.amplitudes[0] for r in runs]
```

```python
fig, axes = plt.subplots(nrows=len(runs), figsize=(6, 18))

for r, ax, amp in zip(runs, axes, amplitudes):
    ax.contourf(r.waves.x, r.waves.t, r.waves.z, 50, cmap=plt.cm.bone)

    t = r.waves.t
    
    # conjugate state
    Ucs = 0.5 * t
    ax.plot(Ucs, t, linewidth=2, label=r'$U_{cs}$')

    # linear two layer speed
    c0 = (r.h1 * (1 - r.h1)) ** .5
    xc = c0 * t

    #ax.plot(xc, t, linewidth=2)

    # linear single
    cl = (r.h1) ** .5
    xl = cl * t
    #ax.plot(xl, t)

    # non-linear wave speed
    c, V, D = nonlinear_coeffs(amp, r.h1)
    #print("c = {}, V = {}, D = {}".format(c, V, D))
    ax.plot(c * t, t, label=r'$c_0$')
    ax.plot(V * t, t, label=r'non-linear')

    ax.set_xlim(0, 12)
    ax.set_ylim(0, 60)
    
    ax.legend(frameon=False, loc='upper left')

fig.tight_layout()
```

### Bimodal wave distribution:

```python
plt.plot(np.abs(smoothed).sum(axis=0))
```

The waves can travel at two distinct speeds, before and after
crossing the current head. This means two distinct branches of
maxima in the ridgelet transform, separated in theta.


## Wavelets

Now we will apply the wavelet transform to each run. We can equally
well do this over a space or time series from any transect through
the signal. For now we will apply the transform to the time series
found by considering the interface height at $x=2$m.

We will consider the transform with both the Morlet and Ricker (aka
Mexican Hat) wavelets. The Morlet wavelet is oscillatory and complex
valued, whereas the Ricker is real valued and single peaked.

```python
transforms_morlet = [WaveletTransform(data=r.waves(x=8, t=r.waves.t).squeeze(), time=r.waves.t, dt=0.04, wavelet=Morlet(), unbias=True) for r in runs]
transforms_ricker = [WaveletTransform(data=r.waves(x=8, t=r.waves.t).squeeze(), time=r.waves.t, dt=0.04, wavelet=Ricker(), unbias=True) for r in runs]
```

```python
fig, axes = plt.subplots(nrows=len(runs), figsize=(6, 18))

for t, ax in zip(transforms_morlet, axes):
    ax.plot(t.time, t.data)

fig.tight_layout()
```

```python
fig, axes = plt.subplots(nrows=len(runs), ncols=2, figsize=(12, 18))

for i, (tm, tr) in enumerate(zip(transforms_morlet, transforms_ricker)):
    tm.plot_power(axes[i, 0])
    tr.plot_power(axes[i, 1])

fig.tight_layout()
```

```python
fig, axes = plt.subplots(nrows=2)
axes[0].contourf(tm.time, tm.fourier_periods, tm.wavelet_power, 50)
axes[1].contourf(tr.time, tr.fourier_periods, tr.wavelet_power, 50)

axes[0].set_ylim(0.5, 10)
axes[1].set_ylim(0.5, 10)
axes[0].set_yscale('log')
axes[1].set_yscale('log')
```

```python
tmi = interp.interp2d(tm.time, tm.fourier_periods, tm.wavelet_power)
tri = interp.interp2d(tr.time, tr.fourier_periods, tr.wavelet_transform.real)

fp = np.logspace(np.log10(0.5), np.log10(10), 100)
time = np.linspace(0, 60, 100)

morlet_power = tmi(time, fp)
ricker_power = tri(time, fp)

plt.contourf(time, fp, morlet_power * ricker_power, 50)
plt.yscale('log')
plt.colorbar()
```

We can apply the wavelet transform to a 1d signal to find out the
position and wavelength of waves in that signal. In this case the
signal is the interface height from the experiment. This is
perturbed by the gravity current release and we get a series of
amplitude ordered solitary waves.


Questions:

- Can we determine the wave amplitude from the wavelet transform?
- If we pick the maxima of the wavelet power spectrum, can we track
  the waves?

```python
def plot_pattern(run):
    fig, ax = plt.subplots()
    contour = ax.contourf(run.waves.X, run.waves.T, run.waves.z, 100)
    ax.set_title(run.index)
    fig.savefig('patterns/' + run.index + '_pattern.png')

for run in runs:
    print "plotting pattern ", run.index
    # plot_pattern(run)

def ricker_transform(run):
    z_1m = run.waves(1, run.waves.t).squeeze()
    return WaveletTransform(z_1m,
                            time=run.waves.t,
                            dt=0.04,
                            unbias=True,
                            wavelet=Ricker())

wavelet_transforms = [ricker_transform(run) for run in runs]

for wa, run in zip(wavelet_transforms, runs):
    print "plotting wavelet ", run.index
    fig, ax = plt.subplots()
    wa.plot_power(ax)
    ax.set_title(run.index)
    fig.savefig('wavelets/' + run.index + '_wavelet.png')
```

```python
tr = transforms_morlet[2]
sig = (tr.wavelet_transform.real / tr.scales[:, None])[20:60, :].sum(axis=0)

recon = tr.reconstruction(scales=tr.scales[20:45])

plt.plot((recon - recon.mean()).real * 22)
plt.plot(sig)
plt.plot(tr.anomaly_data* 22)
```

## Misc crap

```python
dog0 = wavelets.DOG(m=0)
dog1 = wavelets.DOG(m=1)
x = np.linspace(-6, 6)
plt.plot(x, dog0(x))
plt.plot(x, dog1(x))
```

```python
dog1_ridgelets = [ridgelet_transform(r, theta=theta, wavelet=wavelets.DOG(m=1)) for r in runs]
```

```python
fig, axes = plt.subplots(ncols=len(runs))

for ax, ridge in zip(axes, dog1_ridgelets):
    ax.contourf(ridge[10:40, position_slice, 60:140].real.sum(axis=0) ** 2, 50)
    ax.set_yticks([])
    ax.set_xticks([])
```

```python
square_sino = sinogram[50:250:5, 50:90]
```

```python
plt.contourf(square_sino, 100)
```

```python
i, j = np.indices(square_sino.shape)
rad = square_sino.shape[0] / 2
outside_circle = ((i - rad) ** 2 + (j - rad) ** 2 + 1) > (rad) ** 2
circle_sino = square_sino.copy()
circle_sino[outside_circle] = 0
rad_sino = radon(circle_sino, circle=True)
plt.contourf(rad_sino, 50)
```

Note the trick of limiting the area that we look at.
