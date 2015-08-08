import os
import glob
import logging

import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import matplotlib.pyplot as plt


import wavelets

import gc_turbulence as g

logging.info('hello')


def plot(r):
    logging.info(r)
    transformer = g.turbulence.FrontTransformer(r)
    logging.info('transformer constructed')

    uf = transformer.to_front(r.W, order=0)

    # compute mean from uf
    mean_uf = np.mean(uf, axis=1, keepdims=True)
    # expand mean over all x
    full_mean_uf = np.repeat(mean_uf, uf.shape[1], axis=1)
    # transform to lab frame
    trans_mean_uf = transformer.to_lab(full_mean_uf, order=0)
    # subtract mean current from lab frame
    mean_sub_u = r.W[...] - trans_mean_uf

    logging.info('mean calculated')

    signal = np.nanmean(np.nanmean(mean_sub_u, axis=0), axis=0)
    time = r.T[0, 0]

    signal[np.isnan(signal)] = 0

    wt = wavelets.WaveletTransform(signal, dt=r.dt, time=time)

    levels = np.linspace(*np.nanpercentile(mean_sub_u, (1, 99)), num=100)

    logging.info('plotting...')
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))

    wt.plot_power(axes[0, 0])

    axes[0, 1].semilogy(wt.global_wavelet_spectrum, wt.scales)
    axes[1, 0].plot(time, signal)
    axes[1, 1].contourf(mean_sub_u.mean(axis=0), levels=levels)

    fig.savefig('plots_w/{run.index}.png'.format(run=r))


paths = glob.glob(os.path.join(g.default_processed, '*'))
runs = [g.ProcessedRun(cache_path=path) for path in paths]

for run in runs:
    plot(run)
