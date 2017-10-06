#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Functions for manipulating PDFs.

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

__all__ = ["_loglike", "_loglike_s", "loglike",
           "gaussian", "gauss_kde", "gauss_kde_dict"]


def _loglike(data, data_err, data_mask, models, models_err, models_mask,
             ignore_model_err=False, dim_prior=True):
    """
    Internal function for computing the log-likelihood between noisy
    data and noisy models.

    Parameters
    ----------
    data : `~numpy.ndarray` of shape (Nfilt)
        Observed data values.

    data_err : `~numpy.ndarray` of shape (Nfilt)
        Associated (Normal) errors on the observed values.

    data_mask : `~numpy.ndarray` of shape (Nfilt)
        Binary mask (0/1) indicating whether the data was observed.

    models : `~numpy.ndarray` of shape (Nmodel, Nfilt)
        Model values.

    models_err : `~numpy.ndarray` of shape (Nmodel, Nfilt)
        Associated errors on the model values.

    models_mask : `~numpy.ndarray` of shape (Nmodel, Nfilt)
        Binary mask (0/1) indicating whether the model value was observed.

    ignore_model_err : bool, optional
        Whether to ignore model errors during calculation.
        Default is `False`.

    dim_prior : bool, optional
        Whether to apply a dimensional-based correction (prior) to the
        log-likelihood to "normalize" results to be (roughly) dimensionally
        invariant. Default is `True`.

    Returns
    -------
    lnlike : `~numpy.ndarray` of shape (Nmodel)
        Log-likelihood values.

    Ndim : `~numpy.ndarray` of shape (Nmodel)
        Number of observations used in the fit (dimensionality).

    """

    # Initialize errors.
    if ignore_model_err:
        tot_var = np.square(data_err) + np.zeros_like(models_err)
    else:
        tot_var = np.square(data_err) + np.square(models_err)

    # Initialize mask.
    tot_mask = data_mask * models_mask  # combined binary mask
    Ndim = np.sum(tot_mask, axis=1)  # number of dimensions

    # Compute normalization.
    lnl_norm = -0.5 * (Ndim * np.log(2. * np.pi) +
                       np.sum(np.log(tot_var), axis=1))

    # Compute chi2.
    resid = data - models  # residuals
    chi2 = np.sum(tot_mask * np.square(resid) / tot_var, axis=1)  # chi2
    lnl = -0.5 * chi2  # contribution of chi2 to ln(like)

    # Apply dimensionality prior.
    if dim_prior:
        # Normalize by P(n) = exp(-n/2) * (1 / (sqrt(e) - 1)).
        lnl += (-0.5 * Ndim) - np.log(np.sqrt(np.e) - 1.)

    return lnl + lnl_norm, Ndim


def _loglike_s(data, data_err, data_mask, models, models_err, models_mask,
               ignore_model_err=False, dim_prior=True, ltol=1e-4,
               return_scale=False):
    """
    Internal function for computing the log-likelihood between noisy
    data and noisy models while allowing the model to be rescaled.

    Parameters
    ----------
    data : `~numpy.ndarray` of shape (Nfilt)
        Observed data values.

    data_err : `~numpy.ndarray` of shape (Nfilt)
        Associated (Normal) errors on the observed values.

    data_mask : `~numpy.ndarray` of shape (Nfilt)
        Binary mask (0/1) indicating whether the data was observed.

    models : `~numpy.ndarray` of shape (Nmodel, Nfilt)
        Model values.

    models_err : `~numpy.ndarray` of shape (Nmodel, Nfilt)
        Associated errors on the model values.

    models_mask : `~numpy.ndarray` of shape (Nmodel, Nfilt)
        Binary mask (0/1) indicating whether the model value was observed.

    ignore_model_err : bool, optional
        Whether to ignore the model errors during calculation.
        Default is `False`.

    dim_prior : bool, optional
        Whether to apply a dimensional-based correction (prior) to the
        log-likelihood to "normalize" results to be (roughly) dimensionally
        invariant. Default is `True`.

    ltol : float, optional
        The fractional tolerance in the log-likelihood function used to
        determine convergence when including errors when the scale factor is
        left free (i.e. `free_scale = True` and `ignore_model_err = False`).
        Default is `1e-4`.

    return_scale : bool, optional
        Whether to return the scale factor.
        Default is `False`.

    Returns
    -------
    lnlike : `~numpy.ndarray` of shape (Nmodel)
        Log-likelihood values.

    Ndim : `~numpy.ndarray` of shape (Nmodel)
        Number of observations used in the fit (dimensionality).

    scale : `~numpy.ndarray` of shape (Nmodel), optional
        The factor used to scale the model observations to the observed data.
        Returned if `return_scale = True`.

    """

    # Initialize errors.
    if ignore_model_err:
        tot_var = np.square(data_err) + np.zeros_like(models_err)
    else:
        tot_var = np.square(data_err) + np.square(models_err)

    # Initialize mask.
    tot_mask = data_mask * models_mask  # combined binary mask
    Ndim = np.sum(tot_mask, axis=1)  # number of dimensions

    # Derive scalefactors between data and models.
    inter_num = tot_mask * models * data[None, :]
    inter_vals = np.sum(inter_num / tot_var[None, :], axis=1)  # "interaction"
    shape_num = tot_mask * np.square(models)
    shape_vals = np.sum(shape_num / tot_var[None, :], axis=1)  # "shape" term
    scale = inter_vals / shape_vals  # scalefactor

    # Compute chi2.
    resid = data - scale[:, None] * models  # scaled residuals
    chi2 = np.sum(tot_mask * np.square(resid) / tot_var, axis=1)  # chi2
    lnl = -0.5 * chi2  # contribution of chi2 to ln(likelihood)

    # Compute normalization.
    lnl_norm = -0.5 * (Ndim * np.log(2. * np.pi) +
                       np.sum(np.log(tot_var), axis=1))

    # Iterate until convergence if we don't ignore model errors.
    if ignore_model_err is not True:
        lerr = np.inf  # initialize ln(like) error
        while lerr > ltol:
            # Compute new variance using our previous scale.
            tot_var = np.square(data_err) + np.square(scale[:, None] *
                                                      models_err)

            # Compute new scale.
            inter_vals = np.sum(inter_num / tot_var[None, :], axis=1)
            shape_vals = np.sum(shape_num / tot_var[None, :], axis=1)
            scale = inter_vals / shape_vals

            # Compute new chi2.
            resid = data - scale[:, None] * models
            chi2 = np.sum(tot_mask * np.square(resid) / tot_var, axis=1)
            lnl_new = -0.5 * chi2

            # Compute normalization.
            lnl_norm_new = -0.5 * (Ndim * np.log(2. * np.pi) +
                                   np.sum(np.log(tot_var), axis=1))

            # Check tolerance.
            loglike_err = ((lnl_new + lnl_norm_new) - (lnl + lnl_norm) /
                           (lnl + lnl_norm))
            lerr = max(abs(loglike_err))

            # Assign new values.
            lnl, lnl_norm = lnl_new, lnl_norm_new

    # Apply dimensionality prior.
    if dim_prior:
        # Normalize by P(n) = exp(-n/2) * (sqrt(e) / (sqrt(e) - 1)),
        # where n = dof = Ndim-1.
        lnl += (-0.5 * (Ndim - 1)) + (0.5 - np.log(np.sqrt(np.e) - 1.))

    if return_scale:
        return lnl + lnl_norm, Ndim, scale
    else:
        return lnl + lnl_norm, Ndim


def loglike(data, data_err, data_mask, models, models_err, models_mask,
            free_scale=False, ignore_model_err=False, dim_prior=True,
            ltol=1e-4, return_scale=False):
    """
    Compute the ln(likelihood) between an input set of data vectors and an
    input set of (scale-free and/or error-free) model vectors.

    Parameters
    ----------
    data : `~numpy.ndarray` of shape (Nobj, Nfilt)
        Observed data values.

    data_err : `~numpy.ndarray` of shape (Nobj, Nfilt)
        Associated (Normal) errors on the observed values.

    data_mask : `~numpy.ndarray` of shape (Nobj, Nfilt)
        Binary mask (0/1) indicating whether the data was observed.

    models : `~numpy.ndarray` of shape (Nmodel, Nfilt)
        Model values.

    models_err : `~numpy.ndarray` of shape (Nmodel, Nfilt)
        Associated errors on the model values.

    models_mask : `~numpy.ndarray` of shape (Nmodel, Nfilt)
        Binary mask (0/1) indicating whether the model value was observed.

    free_scale : bool, optional
        Whether to include a free scale factor (scaling the model to the data)
        in the fit. Default is `False`.

    ignore_model_err : bool, optional
        Whether to ignore the model errors during calculation.
        Default is `False`.

    dim_prior : bool, optional
        Whether to apply a dimensional-based correction (prior) to the
        log-likelihood to "normalize" results to be (roughly) dimensionally
        invariant. Default is `True`.

    ltol : float, optional
        The fractional tolerance in the log-likelihood function used to
        determine convergence when including errors when the scale factor is
        left free (i.e. `free_scale = True` and `ignore_model_err = False`).
        Default is `1e-4`.

    return_scale : bool, optional
        Whether to return the scale factor derived when `free_scale = True`.
        Default is `False`.

    Returns
    -------
    lnlike : `~numpy.ndarray` of shape (Nobj, Nmodel)
        Log-likelihood values.

    Ndim : `~numpy.ndarray` of shape (Nobj, Nmodel)
        Number of observations used in the fit (dimensionality).

    scale : `~numpy.ndarray` of shape (Nobj, Nmodel), optional
        The factor used to scale the model observations to the observed data.
        Returned if `return_scale = True`.

    """

    if free_scale:
        results = _loglike_s(data, data_err, data_mask, models, models_err,
                             models_mask, ignore_model_err=ignore_model_err,
                             dim_prior=dim_prior, ltol=ltol,
                             return_scale=return_scale)
    else:
        results = _loglike(data, data_err, data_mask, models, models_err,
                           models_mask, ignore_model_err=ignore_model_err,
                           dim_prior=dim_prior)
    
    return results


def gaussian(mu, std, x):
    """
    Gaussian kernal with mean `mu` and standard deviation `std` evaluated
    over grid `x`. Returns the PDF `N(x | mu, std)`.

    """

    dif = x - mu  # difference
    norm = np.sqrt(2. * np.pi) * std  # normalization
    pdf = np.exp(-0.5 * np.square(dif / std)) / norm

    return pdf


def gauss_kde(y, y_std, x, dx=None, y_wt=None, sig_thresh=5., wt_thresh=1e-3):
    """
    Compute smoothed PDF using kernel density estimation.

    Parameters
    ----------
    y : `~numpy.ndarray` with shape (Ny,)
        Array of observed values.

    y_std : `~numpy.ndarray` with shape (Ny,)
        Array of (Gaussian) errors associated with the observed values.

    x : `~numpy.ndarray` with shape (Nx,)
        Grid over which the PDF will be evaluated. Note that this grid should
        be evenly spaced to ensure appropriate sigma clipping behavior.

    dx : float, optional
        The spacing of the input `x` grid. If not provided, `dx` will be
        computed from `x[1] - x[0]`.

    y_wt : `~numpy.ndarray` with shape (Ny,), optional
        An associated set of weights for each of the elements in `y`. If not
        provided, objects will be weighted uniformly.

    sig_thresh : float, optional
        The number of standard deviations from the mean to evaluate
        from before clipping the Gaussian. Default is `5.`.

    wt_thresh : float, optional
        The threshold `wt_thresh * max(y_wt)` used to ignore objects
        with (relatively) negligible weights. Default is `1e-3`.

    Returns
    -------
    pdf : `~numpy.ndarray` with shape (Nx,)
        Probability distribution function (PDF) evaluated over `x`.

    """

    # Initialize values.
    Nx, Ny = len(x), len(y)
    if dx is None:
        dx = x[1] - x[0]
    if y_wt is None:
        y_wt = np.ones(Ny)

    # Clipping kernels.
    centers = np.array((y - x[0]) / dx, dtype='int')  # discretized centers
    offsets = np.array(sig_thresh * y_std / dx, dtype='int')  # offsets
    uppers, lowers = centers + offsets, centers - offsets  # upper/lower bounds
    uppers[uppers>Nx], lowers[lowers<0] = Nx, 0  # limiting to grid edges

    # Initialize PDF.
    pdf = np.zeros(Nx)

    # Apply weight thresholding.
    sel_arr = np.arange(Ny)[y_wt > (wt_thresh * np.max(y_wt))]

    # Compute PDF.
    for i in sel_arr:
        # Stack weighted Gaussian kernel over array slice.
        pdf[lowers[i]:uppers[i]] += y_wt[i] * gaussian(y[i], y_std[i],
                                                       x[lowers[i]:uppers[i]])

    return pdf


def pdf_kde_dict(ydict, ywidth, y_pos, y_idx, x, dx=None, y_wt=None,
                 wt_thresh=1e-3):
    """
    Compute smoothed PDF from point estimates using KDE utilizing a
    PRE-COMPUTED DICTIONARY.

    Keyword arguments:
    ydict -- dictionary of kernels
    ywidth -- associated widths of kernels
    y_pos -- discretized position of observed data
    y_idx -- corresponding index of kernel from dictionary
    y_wt -- associated weight
    x -- PDF grid
    dx -- PDF spacing
    Ny -- number of objects
    Nx -- number of grid elements
    wt_thresh -- wt/wt_max threshold for clipping observations (default=1e-3)

    Outputs:
    pdf -- probability distribution function (PDF) evaluated over x
    """

    # initialize PDF
    pdf = zeros(Nx) 

    # limit analysis to observations with statistically relevant weight
    sel_arr = y_wt > (wt_thresh*y_wt.max())

    # compute PDF
    for i in arange(Ny)[sel_arr]:  # within selected observations
        idx = y_idx[i]  # dictionary element
        yp = y_pos[i]  # kernel center
        yw = ywidth[idx]  # kernel width
        pdf[yp-yw:yp+yw+1] += y_wt[i] * ydict[idx]
        # stack weighted Gaussian kernel over array slice
    
    return pdf
