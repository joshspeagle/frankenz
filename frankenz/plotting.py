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


def truth_vs_pdf(vals, errs, pdfs, pdict, weights=None, pdf_thresh=1e-3,
                 wt_thresh=1e-3, plot_thresh=1., cmap='viridis',
                 smooth=0, verbose=False):
    """
    Plot truth values vs their corresponding PDFs.

    Parameters
    ----------
    vals : `~numpy.ndarray` with shape (Nobj,)
        Truth values.

    vals : `~numpy.ndarray` with shape (Nobj,)
        Errors on the truth values (or smoothing scales).

    pdfs : `~numpy.ndarray` with shape (Nobj, Ngrid)
        PDFs for each object corresponding to the truth values.

    pdict : :class:`PDFDict` instance
        Dictionary used to compute the PDFs.

    weights : `~numpy.ndarray` with shape (Nobj,), optional
        An array used to re-weighted the corresponding PDFs.
        Default is `None`.

    pdf_thresh : float, optional
        The threshold used to clip values when stacking each PDF.
        Default is `1e-3`.

    wt_thresh : float, optional
        The threshold used to ignore PDFs when stacking.
        Default is `1e-3`.

    plot_thresh : float, optional
        The threshold used to threshold the colormap when plotting.
        Default is `1.`.

    cmap : colormap, optional
        The colormap used when plotting results. Default is `'viridis'`.

    smooth : float, optional
        The smoothing scale used to apply 2-D Gaussian smoothing to the
        results. Default is `0` (no smoothing).

    verbose : bool, optional
        Whether to print progress. Default is `False`.

    Returns
    -------
    temp_stack : `~numpy.ndarray` with shape (Ngrid, Ngrid)
        2-D PDF stack.

    """

    # Initialize values
    Ngrid, Nobj = pdict.Ngrid, len(vals)
    vidxs, eidxs = pdict.fit(vals, errs)
    stack = np.zeros((Ngrid, Ngrid))  # 2-D grid

    if weights is None:
        weights = np.ones(Nobj, dtype='float32')
    wtmax = max(weights)

    # Compute 2-D stacked PDF.
    for i in range(Nobj):
        if verbose and i%5000 == 0: 
            sys.stderr.write(str(i)+' ')
        if weights[i] > wt_thresh * wtmax:  # weight threshold cut
            tpdf = pdfs[i]  # pdf
            tsel = tpdf > max(tpdf) * pdf_thresh  # pdf threshold cut
            x_idx, x_cent = eidxs[i], vidxs[i]  # kernels and positions
            x_bound = pdict.sigma_width[x_idx]  # kernel width

            # Stack results.
            tstack = pdict.sigma_dict[x_idx][:,None] * tpdf[tsel]  # 2-D pdf
            xlow = max(x_cent - x_bound, 0)
            xhigh = min(x_cent + x_bound + 1, Ngrid)
            lpad = xlow - (x_cent - x_bound)
            hpad = 2 * x_bound + xhigh - (x_cent + x_bound)
            stack[xlow:xhigh, tsel] += tstack[lpad:hpad] * weights[i]

    # Smooth results.
    if smooth > 0:
        stack = gaussian_filter(stack, smooth)

    # plot results
    stack[stack < plot_thresh] = np.nan
    plt.imshow(stack.T, origin='lower', aspect='auto', 
               extent=(pdict.grid[0], pdict.grid[-1], 
                       pdict.grid[0], pdict.grid[-1]), cmap=cmap)
    plt.colorbar(label='PDF')
    plt.plot([0, 100], [0, 100], 'k--', lw=3)  # 1:1 relation
    #plot(array([0,100]),array([0,100])*1.15+0.15,'k-.',lw=2)  # +15% bound
    #plot(array([0,100]),array([0,100])*0.85-0.15,'k-.',lw=2)  # -15% bound 
    plt.xlim([pdict.grid[0], pdict.grid[-1]])
    plt.ylim([pdict.grid[0], pdict.grid[-1]])
    plt.xlabel('Truth')
    plt.ylabel('Predicted')
    plt.tight_layout()

    return stack
