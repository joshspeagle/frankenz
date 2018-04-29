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

__all__ = ["input_vs_pdf", "input_vs_dpdf", "cdf_vs_epdf", "cdf_vs_ecdf"]


def input_vs_pdf(vals, errs, vdict, pdfs, pgrid, weights=None,
                 pdf_wt_thresh=1e-3, pdf_cdf_thresh=2e-4, wt_thresh=1e-3,
                 cdf_thresh=2e-4, plot_thresh=1., cmap='viridis', smooth=0,
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
        Default is `1.`.

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
                  wt_thresh=1e-3, cdf_thresh=2e-4, plot_thresh=1.,
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
        Default is `1.`.

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
        if err > 0:
            mcvals = rstate.normal(val, err, size=Nmc)  # Monte Carlo errors
        else:
            mcvals = np.tile(val, Nmc)  # tile our inputs
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
        if err > 0:
            mcvals = rstate.normal(val, err, size=Nmc)
        else:
            mcvals = np.tile(val, Nmc)
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
