#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plotting utilities.

"""

from __future__ import (print_function, division)
import six
from six.moves import range

import sys
import os
import warnings
import math
import numpy as np
import warnings
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

__all__ = ["input_vs_pdf", "input_vs_dpdf", "cdf_vs_epdf", "cdf_vs_ecdf",
           "plot2d_network", "plot_node"]


def input_vs_pdf(vals, errs, vdict, pdfs, pgrid, weights=None,
                 pdf_wt_thresh=1e-3, pdf_cdf_thresh=2e-4, wt_thresh=1e-3,
                 cdf_thresh=2e-4, plot_thresh=0., cmap='viridis', smooth=0,
                 plot_kwargs=None, verbose=False, *args, **kwargs):
    """
    Plot input values vs corresponding PDFs.

    Parameters
    ----------
    vals : `~numpy.ndarray` with shape (Nobj,)
        Input x-axis values.

    errs : `~numpy.ndarray` with shape (Nobj,)
        Errors on the input x-axis values (or smoothing scales).

    vdict : :class:`PDFDict` instance
        Dictionary used to quick map the input `vals` and `errs` and determine
        the relevant size of the stacked 2-D array.

    pdfs : `~numpy.ndarray` with shape (Nobj, Ngrid_y)
        PDFs for each object corresponding to the truth values.

    pgrid : `~numpy.ndarray` with shape (Ngrid_y)
        Corresponding grid the PDFs are evaluated on.

    weights : `~numpy.ndarray` with shape (Nobj,), optional
        An array used to re-weight the corresponding PDFs.
        Default is `None`.

    pdf_wt_thresh : float, optional
        The threshold used to clip values when stacking each PDF.
        Default is `1e-3`.

    pdf_cdf_thresh : float, optional
        The `1 - cdf_thresh` threshold of the (sorted) CDF used to clip values
        when stacking each PDF. This option is only
        used when `pdf_wt_thresh=None`. Default is `2e-4`.

    wt_thresh : float, optional
        The threshold used to ignore PDFs when stacking.
        Default is `1e-3`.

    cdf_thresh : float, optional
        The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
        PDFs when stacking.. This option is only used when `wt_thresh=None`.
        Default is `2e-4`.

    plot_thresh : float, optional
        The threshold used to threshold the colormap when plotting.
        Default is `0.`.

    cmap : colormap, optional
        The colormap used when plotting results. Default is `'viridis'`.

    smooth : float or pair of floats, optional
        The smoothing scale(s) used to apply 2-D Gaussian smoothing to the
        results in the x and y directions. Default is `0` (no smoothing).

    plot_kwargs : kwargs, optional
        Keyword arguments to be passed to `~matplotlib.pyplot.imshow`.

    verbose : bool, optional
        Whether to print progress. Default is `False`.

    Returns
    -------
    temp_stack : `~numpy.ndarray` with shape (Ngrid_x, Ngrid_y)
        2-D PDF stack.

    """

    # Initialize values
    Ngrid_x, Ngrid_y, Nobj = vdict.Ngrid, len(pgrid), len(vals)
    stack = np.zeros((Ngrid_x, Ngrid_y))  # 2-D grid
    if pdf_wt_thresh is None and pdf_cdf_thresh is None:
        pdf_wt_thresh = -np.inf
    if plot_kwargs is None:
        plot_kwargs = dict()

    # Apply weight thresholding.
    if weights is None:
        weights = np.ones(Nobj, dtype='float32')
    if wt_thresh is None and cdf_thresh is None:
        wt_thresh = -np.inf  # default to no clipping/thresholding
    if wt_thresh is not None:
        # Use relative amplitude to threshold.
        sel_arr = weights > (wt_thresh * np.max(weights))
        objids = np.arange(Nobj)
    else:
        # Use CDF to threshold.
        idx_sort = np.argsort(weights)  # sort
        w_cdf = np.cumsum(weights[idx_sort])  # compute CDF
        w_cdf /= w_cdf[-1]  # normalize
        sel_arr = w_cdf <= (1. - cdf_thresh)
        objids = idx_sort

    # Compute 2-D stacked PDF.
    vidxs, eidxs = vdict.fit(vals, errs)  # discretize vals, errs
    for i, objid, sel in zip(np.arange(Nobj), objids, sel_arr):
        # Pring progress.
        if verbose:
            sys.stderr.write('\rStacking {0}/{1}'.format(i+1, Nobj))
            sys.stderr.flush()
        # Stack object if it's above the threshold.
        if sel:
            tpdf = np.array(pdfs[objid])  # pdf
            if pdf_wt_thresh is not None:
                tsel = tpdf > max(tpdf) * pdf_wt_thresh  # pdf threshold cut
            else:
                psort = np.argsort(tpdf)
                pcdf = np.cumsum(tpdf[psort])
                tsel = psort[pcdf <= (1. - pdf_cdf_thresh)]  # cdf thresh cut
            tpdf[tsel] /= np.sum(tpdf[tsel])  # re-normalize

            # Compute kernel.
            x_idx, x_cent = eidxs[objid], vidxs[objid]  # index/position
            x_bound = vdict.sigma_width[x_idx]  # kernel width
            pkern = np.array(vdict.sigma_dict[x_idx])  # kernel
            xlow = max(x_cent - x_bound, 0)  # lower bound
            xhigh = min(x_cent + x_bound + 1, Ngrid_x)  # upper bound
            lpad = xlow - (x_cent - x_bound)  # low pad
            hpad = 2 * x_bound + xhigh - (x_cent + x_bound)  # high pad

            # Create 2-D PDF.
            tstack = (pkern[:, None] * tpdf[tsel])[lpad:hpad]
            tstack /= np.sum(tstack)

            # Stack results.
            stack[xlow:xhigh, tsel] += tstack * weights[i]
    if verbose:
        sys.stderr.write('\n')
        sys.stderr.flush()

    # Smooth results.
    if smooth > 0:
        stack = gaussian_filter(stack, smooth)

    # plot results
    stack[stack < plot_thresh] = np.nan
    plt.imshow(stack.T, origin='lower', aspect='auto',
               extent=(vdict.grid[0], vdict.grid[-1],
                       pgrid[0], pgrid[-1]), cmap=cmap,
               **plot_kwargs)
    plt.colorbar(label='Number Density')
    plt.xlim([vdict.grid[0], vdict.grid[-1]])
    plt.ylim([pgrid[0], pgrid[-1]])
    plt.xlabel('Input')
    plt.ylabel('Predicted')
    plt.tight_layout()

    return stack


def input_vs_dpdf(vals, errs, vdict, pdfs, pgrid, pdf_cent, dgrid,
                  weights=None, disp_func=None, disp_args=None,
                  disp_kwargs=None, pdf_wt_thresh=1e-3, pdf_cdf_thresh=2e-4,
                  wt_thresh=1e-3, cdf_thresh=2e-4, plot_thresh=0.,
                  cmap='viridis', smooth=0, plot_kwargs=None, verbose=False,
                  *args, **kwargs):
    """
    Plot truth values vs corresponding centered PDF dispersions.

    Parameters
    ----------
    vals : `~numpy.ndarray` with shape (Nobj,)
        Input x-axis values.

    errs : `~numpy.ndarray` with shape (Nobj,)
        Errors on the input x-axis values (or smoothing scales).

    vdict : :class:`PDFDict` instance
        Dictionary used to quick map the input `vals` and `errs` and determine
        the relevant size of the stacked 2-D array.

    pdfs : `~numpy.ndarray` with shape (Nobj, Ngrid_y)
        PDFs for each object corresponding to the truth values.

    pgrid : `~numpy.ndarray` with shape (Ngrid_y)
        Corresponding grid the PDFs are evaluated on.

    pdf_cent : `~numpy.ndarray` with shape (Nobj)
        Values used to center the corresponding PDFs.

    dgrid : `~numpy.ndarray` with shape (Ngrid_d)
        Corresponding grid in dispersion that the centered PDFs will be
        resampled onto.

    weights : `~numpy.ndarray` with shape (Nobj,), optional
        An array used to re-weight the corresponding PDFs.
        Default is `None`.

    disp_func : func, optional
        Function used to compute the dispersion around the provided `pdf_cent`
        values that takes inputs of the form `(pgrid, cent)` for `cent`
        an element of `pdf_cent`. If no function is provided, the default is
        just `dx = pgrid - cent`.

    disp_args : args, optional
        Arguments to be passed to `disp_func`.

    disp_kwargs : kwargs, optional
        Keyword arguments to be passed to `disp_func`.

    pdf_wt_thresh : float, optional
        The threshold used to clip values when stacking each PDF.
        Default is `1e-3`.

    pdf_cdf_thresh : float, optional
        The `1 - cdf_thresh` threshold of the (sorted) CDF used to clip values
        when stacking each PDF. This option is only
        used when `pdf_wt_thresh=None`. Default is `2e-4`.

    wt_thresh : float, optional
        The threshold used to ignore PDFs when stacking.
        Default is `1e-3`.

    cdf_thresh : float, optional
        The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
        PDFs when stacking.. This option is only used when `wt_thresh=None`.
        Default is `2e-4`.

    plot_thresh : float, optional
        The threshold used to threshold the colormap when plotting.
        Default is `0.`.

    cmap : colormap, optional
        The colormap used when plotting results. Default is `'viridis'`.

    smooth : float or pair of floats, optional
        The smoothing scale(s) used to apply 2-D Gaussian smoothing to the
        results in the x and y directions. Default is `0` (no smoothing).

    plot_kwargs : kwargs, optional
        Keyword arguments to be passed to `~matplotlib.pyplot.imshow`.

    verbose : bool, optional
        Whether to print progress. Default is `False`.

    Returns
    -------
    temp_stack : `~numpy.ndarray` with shape (Ngrid_x, Ngrid_y)
        2-D PDF stack.

    """

    # Initialize values
    Ngrid_x, Ngrid_y, Nobj = vdict.Ngrid, len(pgrid), len(vals)
    Ngrid_d = len(dgrid)
    stack = np.zeros((Ngrid_x, Ngrid_d))  # 2-D grid
    if pdf_wt_thresh is None and pdf_cdf_thresh is None:
        pdf_wt_thresh = -np.inf
    if plot_kwargs is None:
        plot_kwargs = dict()
    if disp_func is None:
        def disp_func(pgrid, cent):
            return pgrid - cent
    if disp_args is None:
        disp_args = []
    if disp_kwargs is None:
        disp_kwargs = dict()

    # Apply weight thresholding.
    if weights is None:
        weights = np.ones(Nobj, dtype='float32')
    if wt_thresh is None and cdf_thresh is None:
        wt_thresh = -np.inf  # default to no clipping/thresholding
    if wt_thresh is not None:
        # Use relative amplitude to threshold.
        sel_arr = weights > (wt_thresh * np.max(weights))
        objids = np.arange(Nobj)
    else:
        # Use CDF to threshold.
        idx_sort = np.argsort(weights)  # sort
        w_cdf = np.cumsum(weights[idx_sort])  # compute CDF
        w_cdf /= w_cdf[-1]  # normalize
        sel_arr = w_cdf <= (1. - cdf_thresh)
        objids = idx_sort

    # Compute 2-D stacked PDF.
    vidxs, eidxs = vdict.fit(vals, errs)  # discretize vals, errs
    for i, objid, sel in zip(np.arange(Nobj), objids, sel_arr):
        # Pring progress.
        if verbose:
            sys.stderr.write('\rStacking {0}/{1}'.format(i+1, Nobj))
            sys.stderr.flush()
        # Stack object if it's above the threshold.
        if sel:
            tpdf0 = np.array(pdfs[objid])  # pdf
            dx = disp_func(pgrid, pdf_cent[objid], *disp_args, **disp_kwargs)
            tpdf = np.interp(dgrid, dx, tpdf0)  # centered and resampled pdf

            if pdf_wt_thresh is not None:
                tsel = tpdf > max(tpdf) * pdf_wt_thresh  # pdf threshold cut
            else:
                psort = np.argsort(tpdf)
                pcdf = np.cumsum(tpdf[psort])
                tsel = psort[pcdf <= (1. - pdf_cdf_thresh)]  # cdf thresh cut
            tpdf[tsel] /= np.sum(tpdf[tsel])  # re-normalize

            # Compute kernel.
            x_idx, x_cent = eidxs[objid], vidxs[objid]  # index/position
            x_bound = vdict.sigma_width[x_idx]  # kernel width
            pkern = np.array(vdict.sigma_dict[x_idx])  # kernel
            xlow = max(x_cent - x_bound, 0)  # lower bound
            xhigh = min(x_cent + x_bound + 1, Ngrid_x)  # upper bound
            lpad = xlow - (x_cent - x_bound)  # low pad
            hpad = 2 * x_bound + xhigh - (x_cent + x_bound)  # high pad

            # Create 2-D PDF.
            tstack = (pkern[:, None] * tpdf[tsel])[lpad:hpad]
            tstack /= np.sum(tstack)

            # Stack results.
            stack[xlow:xhigh, tsel] += tstack * weights[i]
    if verbose:
        sys.stderr.write('\n')
        sys.stderr.flush()

    # Smooth results.
    if smooth > 0:
        stack = gaussian_filter(stack, smooth)

    # plot results
    stack[stack < plot_thresh] = np.nan
    plt.imshow(stack.T, origin='lower', aspect='auto',
               extent=(vdict.grid[0], vdict.grid[-1],
                       dgrid[0], dgrid[-1]), cmap=cmap,
               **plot_kwargs)
    plt.colorbar(label='Number Density')
    plt.xlim([vdict.grid[0], vdict.grid[-1]])
    plt.ylim([dgrid[0], dgrid[-1]])
    plt.xlabel('Input')
    plt.ylabel('Predicted')
    plt.tight_layout()

    return stack


def cdf_vs_epdf(vals, errs, pdfs, pdf_grid, Nmc=100, weights=None, Nbins=50,
                plot_kwargs=None, rstate=None, *args, **kwargs):
    """
    Plot CDF draws vs the empirical PDF (i.e. normalized counts).

    Parameters
    ----------
    vals : `~numpy.ndarray` with shape (Nobj,)
        Input x-axis values.

    errs : `~numpy.ndarray` with shape (Nobj,)
        Errors on the input x-axis values (or smoothing scales).

    pdfs : `~numpy.ndarray` with shape (Nobj, Ngrid)
        PDFs for each object corresponding to the truth values.

    pdf_grid : `~numpy.ndarray` with shape (Ngrid)
        Grid used to compute the PDFs.

    Nmc : int, optional
        The number of Monte Carlo realizations of the true value(s) if the
        provided error(s) are non-zero. Default is `100`.

    weights : `~numpy.ndarray` with shape (Nobj,), optional
        An array used to re-weight the corresponding PDFs.
        Default is `None`.

    Nbins : int, optional
        The number of bins used for plotting. Default is `50`.

    plot_kwargs : kwargs, optional
        Keyword arguments to be passed to `~matplotlib.pyplot.hist`.

    rstate : `~numpy.random.RandomState` instance, optional
        Random state instance. If not passed, the default `~numpy.random`
        instance will be used.

    Returns
    -------
    counts : `~numpy.ndarray` with shape (Nbins)
        Effective number of counts in each bin.

    """

    # Initialize values
    Ngrid, Nobj = len(pdf_grid), len(vals)
    if plot_kwargs is None:
        plot_kwargs = dict()
        plot_kwargs['color'] = 'blue'
        plot_kwargs['alpha'] = 0.6
    if rstate is None:
        rstate = np.random
    if weights is None:
        weights = np.ones(Nobj, dtype='float32')
    weights = np.array([np.tile(float(w), Nmc) for w in weights]).flatten()

    # Compute CDF values
    cdf_draws = np.zeros((Nobj, Nmc))
    for i, (val, err, pdf) in enumerate(zip(vals, errs, pdfs)):
        cdf = pdf.cumsum()
        cdf /= cdf[-1]
        mcvals = rstate.normal(val, err, size=Nmc)  # Monte Carlo errors
        cdf_draws[i] = np.interp(mcvals, pdf_grid, cdf)
    cdf_draws = cdf_draws.flatten()

    # Plot result.
    n, _, _ = plt.hist(cdf_draws, bins=np.linspace(0., 1., Nbins + 1),
                       weights=weights, density=True, **plot_kwargs)
    plt.xlabel('CDF Draws')
    plt.ylabel('Normalized Counts')

    return n


def cdf_vs_ecdf(vals, errs, pdfs, pdf_grid, Nmc=100, weights=None,
                plot_kwargs=None, rstate=None, *args, **kwargs):
    """
    Plot CDF draws vs the empirical CDF (i.e. cumulative normalized counts).

    Parameters
    ----------
    vals : `~numpy.ndarray` with shape (Nobj,)
        Truth values.

    errs : `~numpy.ndarray` with shape (Nobj,)
        Errors on the truth values (or smoothing scales).

    pdfs : `~numpy.ndarray` with shape (Nobj, Ngrid)
        PDFs for each object corresponding to the truth values.

    pdf_grid : `~numpy.ndarray` with shape (Ngrid)
        Grid used to compute the PDFs.

    Nmc : int, optional
        The number of Monte Carlo realizations of the true value(s) if the
        provided error(s) are non-zero. Default is `100`.

    weights : `~numpy.ndarray` with shape (Nobj,), optional
        An array used to re-weight the corresponding PDFs.
        Default is `None`.

    plot_kwargs : kwargs, optional
        Keyword arguments to be passed to `~matplotlib.pyplot.hist`.

    rstate : `~numpy.random.RandomState` instance, optional
        Random state instance. If not passed, the default `~numpy.random`
        instance will be used.

    Returns
    -------
    sorted_cdf_draws : `~numpy.ndarray` with shape (Nobj * Nmc)
        Sorted set of (weighted) CDF draws.

    ecdf : `~numpy.ndarray` with shape (Nobj * Nmc)
        The empirical sorted (weighted) CDF.

    """

    # Initialize values
    Ngrid, Nobj = len(pdf_grid), len(vals)
    if plot_kwargs is None:
        plot_kwargs = dict()
        plot_kwargs['color'] = 'blue'
        plot_kwargs['alpha'] = 0.6
    if rstate is None:
        rstate = np.random
    if weights is None:
        weights = np.ones(Nobj, dtype='float32')
    weights = np.array([np.tile(float(w), Nmc) for w in weights]).flatten()

    # Compute CDF values
    cdf_draws = np.zeros((Nobj, Nmc))
    for i, (val, err, pdf) in enumerate(zip(vals, errs, pdfs)):
        cdf = pdf.cumsum()
        cdf /= cdf[-1]
        mcvals = rstate.normal(val, err, size=Nmc)
        cdf_draws[i] = np.interp(mcvals, pdf_grid, cdf)
    cdf_draws = cdf_draws.flatten()

    # Compute weighted x and y grids.
    sort_idx = np.argsort(cdf_draws)
    cdf_sorted, weights_sorted = cdf_draws[sort_idx], weights[sort_idx]
    cdf_diff = np.append(cdf_sorted[0], cdf_sorted[1:] - cdf_sorted[:-1])
    x, y = weights_sorted, weights_sorted * cdf_diff
    x = x.cumsum() / x.sum()
    y = y.cumsum() / y.sum()

    # Plot result.
    plt.plot(x, y, **plot_kwargs)
    plt.xlabel('Sorted CDF Draws')
    plt.ylabel('Empirical CDF')

    return x, y


def plot2d_network(network, counts='weighted', label_name=None,
                   labels=None, labels_err=None, vals=None, dims=(0, 1),
                   cmap='viridis',  Nmc=5, point_est='median',
                   plot_kwargs=None, rstate=None, verbose=True,
                   *args, **kwargs):
    """
    Plot a 2-D projection of the network colored by the chosen variable.

    Parameters
    ----------
    network : `~frankenz.networks._Network`-derived object
        The trained and populated network object.

    counts : {'absolute', 'weighted'}, optional
        The number density of objects mapped onto the network. If
        `'absolute'`, the raw number of objects associated with each node
        will be plotted. If `'weighted'`, the weighted number of objects
        will be shown. Default is `'weighted'`.

    labels : `~numpy.ndarray` with shape (Nobj), optional
        The labels we want to project over the network. Will override
        `counts` if provided.

    label_name : str, optional
        The name of the label.

    labels_err : `~numpy.ndarray` with shape (Nobj), optional
        Errors on the labels.

    vals : `~numpy.ndarray` with shape (Nnodes), optional
        The values to be plotted directly on the network. Overrides
        `labels`.

    dims : 2-tuple, optional
        The `(x, y)` dimensions the network should be plotted over. Default is
        `(0, 1)`.

    cmap : colormap, optional
        The colormap used when plotting results. Default is `'viridis'`.

    Nmc : int, optional
        The number of Monte Carlo realizations of the label value(s) if the
        error(s) are provided. Default is `5`.

    point_est : str or func, optional
        The point estimator to be plotted. Pre-defined options include
        `'mean'`, `'median'`, `'std'`, and `'mad'`. If a function is passed,
        it will be used to compute the weighted point estimate using input
        of the form `(labels, wts)`. Default is `'median'`.

    plot_kwargs : kwargs, optional
        Keyword arguments to be passed to `~matplotlib.pyplot.scatter`.

    rstate : `~numpy.random.RandomState` instance, optional
        Random state instance. If not passed, the default `~numpy.random`
        instance will be used.

    verbose : bool, optional
        Whether to print progress. Default is `True`.

    Returns
    -------
    vals : `~numpy.ndarray` with shape (Nnodes)
        Corresponding point estimates for the input labels.

    """

    # Initialize values.
    if plot_kwargs is None:
        plot_kwargs = dict()
    if rstate is None:
        rstate = np.random
    if label_name is None and (labels is not None or vals is not None):
        label_name = 'Node Value'
    Nnodes = network.NNODE
    xpos = network.nodes_pos[:, dims[0]]
    ypos = network.nodes_pos[:, dims[1]]

    # Compute counts.
    if counts == 'absolute' and labels is None and vals is None:
        vals = network.nodes_Nmatch
        if label_name is None:
            label_name = 'Counts'
    elif counts == 'weighted' and labels is None and vals is None:
        vals = np.array([np.exp(logsumexp(logwts))
                         for logwts in network.nodes_logwts])
        if label_name is None:
            label_name = 'Weighted Counts'

    # Compute point estimates.
    if vals is None and labels is not None:
        vals = np.zeros(Nnodes)
        for i in range(Nnodes):
            # Print progress.
            if verbose:
                sys.stderr.write('\rComputing {0} estimate {1}/{2}'
                                 .format(label_name, i+1, Nnodes))
                sys.stderr.flush()
            # Grab relevant objects.
            idxs, logwts = network.nodes_idxs[i], network.nodes_logwts[i]
            wts = np.exp(logwts - logsumexp(logwts))  # normalized weights
            ys = labels[idxs]  # labels
            Ny = len(ys)
            # Account for label errors (if provided) using Monte Carlo methods.
            if labels_err is not None:
                yes = labels_err[idxs]  # errors
                ys = rstate.normal(ys, yes, size=(Nmc, Ny)).flatten()
                wts = np.tile(wts, Nmc) / Nmc
            if point_est == 'mean':
                # Compute weighted mean.
                val = np.dot(wts, ys)
            elif point_est == 'median':
                # Compute weighted median.
                sort_idx = np.argsort(ys)
                sort_cdf = wts[sort_idx].cumsum()
                val = np.interp(0.5, sort_cdf, ys[sort_idx])
            elif point_est == 'std':
                # Compute weighted std.
                ymean = np.dot(wts, ys)  # mean
                val = np.dot(wts, np.square(ys - ymean))
            elif point_est == 'mad':
                # Compute weighted MAD.
                sort_idx = np.argsort(ys)
                sort_cdf = wts[sort_idx].cumsum()
                ymed = np.interp(0.5, sort_cdf, ys[sort_idx])  # median
                dev = np.abs(ys - ymed)  # absolute deviation
                sort_idx = np.argsort(dev)
                sort_cdf = wts[sort_idx].cumsum()
                val = np.interp(0.5, sort_cdf, dev[sort_idx])
            else:
                try:
                    val = point_est(ys, wts)
                except:
                    raise RuntimeError("`point_est` function failed!")
            vals[i] = val
        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

    # Plot results.
    plt.scatter(xpos, ypos, c=vals, cmap=cmap, **plot_kwargs)
    plt.xlabel(r'$x_{0}$'.format(str(dims[0])))
    plt.ylabel(r'$x_{0}$'.format(str(dims[1])))
    plt.colorbar(label=label_name)

    return vals


def plot_node(network, models, models_err, pos=None, idx=None, models_x=None,
              Nrsamp=1, Nmc=5, node_kwargs=None, violin_kwargs=None,
              rstate=None, *args, **kwargs):
    """
    Plot a 2-D projection of the network colored by the chosen variable.

    Parameters
    ----------
    network : `~frankenz.networks._Network`-derived object
        The trained and populated network object.

    models : `~numpy.ndarray` with shape (Nobj, Ndim)
        The models mapped onto the network.

    models_err : `~numpy.ndarray` with shape (Nobj, Ndim)
        Errors on the models.

    pos : tuple of shape (Nproj), optional
        The `Nproj`-dimensional position of the node. Mutually exclusive with
        `idx`.

    idx : int, optional
        Index of the node. Mutually exclusive with `pos`.

    models_x : `~numpy.ndarray` with shape (Ndim), optional
        The `x` values corresponding to the `Ndim` model values.

    Nrsamp : int, optional
        Number of times to resample the weighted collection of models
        associated with the given node. Default is `1`.

    Nmc : int, optional
        The number of Monte Carlo realizations of the model values if the
        errors are provided. Default is `5`.

    node_kwargs : kwargs, optional
        Keyword arguments to be passed to `~matplotlib.pyplot.plot` when
        plotting the node model.

    violin_kwargs : kwargs, optional
        Keyword arguments to be passed to `~matplotlib.pyplot.violinplot`
        when plotting the distribution of model values.

    rstate : `~numpy.random.RandomState` instance, optional
        Random state instance. If not passed, the default `~numpy.random`
        instance will be used.

    """

    # Initialize values.
    if node_kwargs is None:
        node_kwargs = dict()
    if violin_kwargs is None:
        violin_kwargs = dict()
    if rstate is None:
        rstate = np.random
    if idx is None and pos is None:
        raise ValueError("Either `idx` or `pos` must be specified.")
    elif idx is not None and pos is not None:
        raise ValueError("Both `idx` and `pos` cannot be specified.")
    if models_x is None:
        models_x = np.arange(models.shape[-1] + 1)
    node_kwargs['color'] = node_kwargs.get('color', 'black')
    node_kwargs['marker'] = node_kwargs.get('marker', '*')
    node_kwargs['markersize'] = node_kwargs.get('markersize', '10')
    node_kwargs['alpha'] = node_kwargs.get('alpha', 0.6)
    violin_kwargs['widths'] = violin_kwargs.get('widths', 600)
    violin_kwargs['showextrema'] = violin_kwargs.get('showextrema', False)

    # Get node.
    (idx, node_model, pos,
     idxs, logwts, scales, scales_err) = network.get_node(pos=pos, idx=idx)
    tmodels, tmodels_err = models[idxs], models_err[idxs]  # grab models
    wts = np.exp(logwts - logsumexp(logwts))  # compute weights

    # Resample models.
    Nmatch = len(idxs)
    idx_rsamp = rstate.choice(Nmatch, p=wts, size=Nmatch*Nrsamp)

    # Perturb model values.
    tmodels_mc = rstate.normal(tmodels[idx_rsamp], tmodels_err[idx_rsamp])

    # Rescale results.
    snorm = np.mean(np.array(scales)[idx_rsamp])
    tmodels_mc /= (np.array(scales)[idx_rsamp, None] / snorm)

    # Rescale baseline model (correction should be small in most cases).
    mean_model = np.mean(tmodels_mc, axis=0)
    std_model = np.std(tmodels_mc, axis=0)
    num = np.dot(mean_model / std_model, node_model / std_model)
    den = np.dot(node_model / std_model, node_model / std_model)
    node_scale = num / den
    if abs(node_scale - 1.) < 0.05:
        node_scale = 1.

    # Plot results.
    plt.plot(models_x, node_model * node_scale, **node_kwargs)
    for i in range(models.shape[-1]):
        vals = tmodels_mc[:, i]
        plt.violinplot(vals, [models_x[i]], **violin_kwargs)
    plt.ylim([min(mean_model - 3 * std_model),
              max(mean_model + 3 * std_model)])
