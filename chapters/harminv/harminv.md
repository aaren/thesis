---
layout: chapter
title: Removing background waves with Harmonic Inversion
---

# Removing background waves with Harmonic Inversion

A first examination of the turbulence data reveals the presence of
background waves in the tank. These waves are remnants of the
filling process. Ideally the tank would have been allowed to settle
completely before an experiment was performed but logistical
contraints prevented this from occuring: the tank forms an efficient
wave guide and the time needed for wave decay would have been on the
order of one hour.

As it is, the waves form a strong and detectable signal in the data.
The challenge is to isolate and remove this signal. With the
assumption that the waves combine linearly with the gravity current
this can be done easily by subtracting the wave signal.


## Isolation of the signal

### Frame comparison

A first approximation to removing the background wave field is to
use the differnce between the current relative and lab relative
frames to attempt to isolate the wave field.

**Show work from last May with the summation and averaging of the
waves**

This is nearly satisfactory, but doesn't quite work.


### Harmonic Inversion

At first glance, this would appear to be exactly the problem that
the fourier transform solves. However, the fourier transform lacks
resolution for closely spaced frequencies:

```python
# Plot a figure showing the fourier transform results
```

A more robust means of removing the background signal would be to
use our assumptions about the character of the signal in a more
formal way. *Given that* our background signal consists of a sum of
sinusoidal modes, we wish to determine the frequencies and
amplitudes of those mdoes - this is the problem of *harmonic
inversion*.

"Harmonic inversion of time signals," J.  Chem.  Phys.  107, 6756 (1997)

```python
import gc_turbulence as g

index = 'r14_01_14a'
cache_path = g.default_processed + index + '.hdf5'
r = g.ProcessedRun(cache_path=cache_path)

u_levels = np.linspace(*np.percentile(r.Uf[...], (1, 99)), num=100)

tf = r.Tf[:, 0, :]
zf = r.Zf[:, 0, :]
```

As we might expect, there are no waves in the mean front relative data:

```python
mean = np.mean(r.Uf[...], axis=1)
plt.contourf(tf, zf, mean, levels=u_levels)
```

If we subtract the mean from the data and look at a single vertical
slice we can see waves:

```python
mean_subtracted = r.Uf[...] - mean[..., None, ...]
c = plt.contourf(tf, zf, mean_subtracted[:, 30, :], 100)
```

```python
import harminv
signal = mean_subtracted[:50, 30, :].sum(axis=0)

fft = np.fft.fft(signal)
freqs = np.fft.fft_freqs(tf[0], d=0.01)
plt.semilogx(freqs, fft)

mean_u = np.mean(r.U[-10:, :, :], axis=0)
Te = r.T[0, :, :]
plt.contourf(Te, x, mean_u, levels=wave_levels)
inversion = harminv.invert(signal, fmin=0.1, fmax=2, dt=0.01)
```

A better way to get the signal:

```python
signal = np.mean(r.U[-10:, :, :], axis=0).mean(axis=0)[:4000]
t = r.T[0, 0, :4000]

plt.plot(t, signal)
```

```python
fft = np.fft.fft(signal)
freqs = np.fft.fftfreq(signal.size, d=0.01)
plt.semilogx(freqs, fft)
```

```python
inversion = harminv.Harminv(signal, fmin=0.01, fmax=4, dt=0.01)

for i, mode in enumerate(inversion.modes[inversion.Q > 100]):
    plt.plot(t, mode + inversion.frequency[i])
```

```python
# compute the reconstructed signal
rsignal = inversion.modes[inversion.decay > -0.1].sum(axis=0)
plt.plot(t, signal)
plt.plot(t, rsignal)
```

```python
# compute the extended signal
te = r.T[0, 0, :]
emodes = inversion.compute_modes(te)
esignal = emodes[inversion.decay > -0].sum(axis=0)

plt.plot(t, signal)
plt.plot(te, esignal)
```

```python
# now subtract from the data
mean_u = np.mean(r.U[-10:, :, :], axis=0)
Te = r.T[0, :, :]

plt.contourf(mean_u - esignal.real, 100)
```
