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

__all__ = ["truth_vs_pdf"]


def truth_vs_pdf(vals, errs, pdfs, pdict, weights=None, pdf_wt_thresh=1e-3,
                 pdf_cdf_thresh=2e-4, wt_thresh=1e-3, cdf_thresh=2e-4,
                 plot_thresh=1., cmap='viridis', smooth=0, plot_kwargs=None,
                 verbose=False, *args, **kwargs):
    """
    Plot truth values vs their corresponding PDFs.

    Parameters
    ----------
    vals : `~numpy.ndarray` with shape (Nobj,)
        Truth values.

    errs : `~numpy.ndarray` with shape (Nobj,)
        Errors on the truth values (or smoothing scales).

    pdfs : `~numpy.ndarray` with shape (Nobj, Ngrid)
        PDFs for each object corresponding to the truth values.

    pdict : :class:`PDFDict` instance
        Dictionary used to compute the PDFs.

    weights : `~numpy.ndarray` with shape (Nobj,), optional
        An array used to re-weighted the corresponding PDFs.
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

    smooth : float, optional
        The smoothing scale used to apply 2-D Gaussian smoothing to the
        results. Default is `0` (no smoothing).

    plot_kwargs : kwargs, optional
        Keyword arguments to be passed to `~matplotlib.pyplot.imshow`.

    verbose : bool, optional
        Whether to print progress. Default is `False`.

    Returns
    -------
    temp_stack : `~numpy.ndarray` with shape (Ngrid, Ngrid)
        2-D PDF stack.

    """

    # Initialize values
    Ngrid, Nobj = pdict.Ngrid, len(vals)
    stack = np.zeros((Ngrid, Ngrid))  # 2-D grid
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
    vidxs, eidxs = pdict.fit(vals, errs)  # discretize vals, errs
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
            x_bound = pdict.sigma_width[x_idx]  # kernel width
            pkern = np.array(pdict.sigma_dict[x_idx])  # kernel
            xlow = max(x_cent - x_bound, 0)  # lower bound
            xhigh = min(x_cent + x_bound + 1, Ngrid)  # upper bound
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
               extent=(pdict.grid[0], pdict.grid[-1],
                       pdict.grid[0], pdict.grid[-1]), cmap=cmap,
               **plot_kwargs)
    plt.colorbar(label='PDF')
    plt.plot([0, 100], [0, 100], 'k--', lw=3)  # 1:1 relation
    plt.xlim([pdict.grid[0], pdict.grid[-1]])
    plt.ylim([pdict.grid[0], pdict.grid[-1]])
    plt.xlabel('Truth')
    plt.ylabel('Predicted')
    plt.tight_layout()

    return stack
