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
from scipy.special import erf, xlogy, gammaln

__all__ = ["_loglike", "_loglike_s", "loglike", "logprob",
           "gaussian", "gaussian_bin", "gauss_kde", "gauss_kde_dict",
           "magnitude", "inv_magnitude", "luptitude", "inv_luptitude",
           "PDFDict", "pdfs_resample", "pdfs_summarize"]


def _loglike(data, data_err, data_mask, models, models_err, models_mask,
             ignore_model_err=False, dim_prior=True, *args, **kwargs):
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
        log-likelihood. Transforms the likelihood to a chi2 distribution
        with `Nfilt` degrees of freedom. Default is `True`.

    Returns
    -------
    lnlike : `~numpy.ndarray` of shape (Nmodel)
        Log-likelihood values.

    Ndim : `~numpy.ndarray` of shape (Nmodel)
        Number of observations used in the fit (dimensionality).

    chi2 : `~numpy.ndarray` of shape (Nmodel)
        Chi-square values used to compute the log-likelihood.

    """

    # Initialize errors.
    if ignore_model_err:
        tot_var = np.square(data_err) + np.zeros_like(models_err)
    else:
        tot_var = np.square(data_err) + np.square(models_err)

    # Initialize mask.
    tot_mask = data_mask * models_mask  # combined binary mask
    Ndim = np.sum(tot_mask, axis=1)  # number of dimensions

    # Compute chi2.
    resid = data - models  # residuals
    chi2 = np.sum(tot_mask * np.square(resid) / tot_var, axis=1)  # chi2

    # Apply dimensionality prior.
    if dim_prior:
        # Compute logpdf of chi2 distribution.
        a = 0.5 * Ndim  # dof
        lnl = xlogy(a - 1., chi2) - (chi2 / 2.) - gammaln(a) - (np.log(2.) * a)
    else:
        # Compute logpdf of multivariate normal.
        lnl = -0.5 * chi2
        lnl += -0.5 * (Ndim * np.log(2. * np.pi) +
                       np.sum(np.log(tot_var), axis=1))

    return lnl, Ndim, chi2


def _loglike_s(data, data_err, data_mask, models, models_err, models_mask,
               ignore_model_err=False, dim_prior=True, ltol=1e-3,
               return_scale=False, *args, **kwargs):
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
        log-likelihood. Transforms the likelihood to a chi2 distribution
        with `Nfilt - 1` degrees of freedom. Default is `True`.

    ltol : float, optional
        The tolerance in the log-likelihood function used to
        determine convergence if including errors when the scale factor is
        left free (i.e. `free_scale = True` and `ignore_model_err = False`).
        Default is `1e-3`.

    return_scale : bool, optional
        Whether to return the scale factor.
        Default is `False`.

    Returns
    -------
    lnlike : `~numpy.ndarray` of shape (Nmodel)
        Log-likelihood values.

    Ndim : `~numpy.ndarray` of shape (Nmodel)
        Number of observations used in the fit (dimensionality).

    chi2 : `~numpy.ndarray` of shape (Nmodel)
        Chi-square values used to compute the log-likelihood.

    scale : `~numpy.ndarray` of shape (Nmodel), optional
        The factor used to scale the model observations to the observed data.
        Returned if `return_scale = True`.

    scale_err : `~numpy.ndarray` of shape (Nmodel), optional
        The error on the factor used to scale the model observations.
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
    inter_vals = np.sum(inter_num / tot_var, axis=1)  # "interaction"
    shape_num = tot_mask * np.square(models)
    shape_vals = np.sum(shape_num / tot_var, axis=1)  # "shape" term
    scale = inter_vals / shape_vals  # scalefactor

    # Compute chi2.
    resid = data - scale[:, None] * models  # scaled residuals
    chi2 = np.sum(tot_mask * np.square(resid) / tot_var, axis=1)  # chi2

    # Compute multivariate normal logpdf.
    lnl = -0.5 * chi2
    lnl += -0.5 * (Ndim * np.log(2. * np.pi) +
                   np.sum(np.log(tot_var), axis=1))

    # Iterate until convergence if we don't ignore model errors.
    if ignore_model_err is not True:
        lerr = np.inf  # initialize ln(like) error
        while lerr > ltol:
            # Compute new variance using our previous scale.
            tot_var = np.square(data_err) + np.square(scale[:, None] *
                                                      models_err)

            # Compute new scale.
            inter_vals = np.sum(inter_num / tot_var, axis=1)
            shape_vals = np.sum(shape_num / tot_var, axis=1)
            scale_new = inter_vals / shape_vals

            # Compute new chi2.
            resid = data - scale_new[:, None] * models
            chi2 = np.sum(tot_mask * np.square(resid) / tot_var, axis=1)

            # Compute new logpdf.
            lnl_new = -0.5 * chi2
            lnl_new += -0.5 * (Ndim * np.log(2. * np.pi) +
                               np.sum(np.log(tot_var), axis=1))

            # Check tolerance.
            loglike_err = lnl_new - lnl
            lerr = max(abs(loglike_err))

            # Assign new values.
            lnl, scale = lnl_new, scale_new

    # Apply dimensionality prior.
    if dim_prior:
        # Compute logpdf of chi2 distribution.
        a = 0.5 * (Ndim - 1)  # dof
        lnl = xlogy(a - 1., chi2) - (chi2 / 2.) - gammaln(a) - (np.log(2.) * a)

    if return_scale:
        scale_err = np.sqrt(1. / shape_vals)
        return lnl, Ndim, chi2, scale, scale_err
    else:
        return lnl, Ndim, chi2


def loglike(data, data_err, data_mask, models, models_err, models_mask,
            free_scale=False, ignore_model_err=False, dim_prior=True,
            ltol=1e-4, return_scale=False, *args, **kwargs):
    """
    Compute the ln(likelihood) between an input set of data vectors and an
    input set of (scale-free and/or error-free) model vectors.

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

    free_scale : bool, optional
        Whether to include a free scale factor (scaling the model to the data)
        in the fit. Default is `False`.

    ignore_model_err : bool, optional
        Whether to ignore the model errors during calculation.
        Default is `False`.

    dim_prior : bool, optional
        Whether to apply a dimensional-based correction (prior) to the
        log-likelihood. Transforms the likelihood to a chi2 distribution
        with `dof` degrees of freedom. Default is `True`.

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
    lnlike : `~numpy.ndarray` of shape (Nmodel)
        Log-likelihood values.

    Ndim : `~numpy.ndarray` of shape (Nmodel)
        Number of observations used in the fit (dimensionality).

    chi2 : `~numpy.ndarray` of shape (Nmodel)
        Chi-square values used to compute the log-likelihood.

    scale : `~numpy.ndarray` of shape (Nmodel), optional
        The factor used to scale the model observations to the observed data.
        Returned if `return_scale = True`.

    scale_err : `~numpy.ndarray` of shape (Nmodel), optional
        The error on the factor used to scale the model observations.
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


def logprob(data, data_err, data_mask, models, models_err, models_mask,
            free_scale=False, ignore_model_err=False, dim_prior=True,
            ltol=1e-4, return_scale=False, *args, **kwargs):
    """
    A wrapper for the :meth:`~frankenz.pdf.loglike` function with output
    formats needed by objects in `~frankenz.fitting`.

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

    free_scale : bool, optional
        Whether to include a free scale factor (scaling the model to the data)
        in the fit. Default is `False`.

    ignore_model_err : bool, optional
        Whether to ignore the model errors during calculation.
        Default is `False`.

    dim_prior : bool, optional
        Whether to apply a dimensional-based correction (prior) to the
        log-likelihood. Transforms the likelihood to a chi2 distribution
        with `dof` degrees of freedom. Default is `True`.

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
    lnlike : `~numpy.ndarray` of shape (Nmodel)
        Log-likelihood values.

    Ndim : `~numpy.ndarray` of shape (Nmodel)
        Number of observations used in the fit (dimensionality).

    chi2 : `~numpy.ndarray` of shape (Nmodel)
        Chi-square values used to compute the log-likelihood.

    scale : `~numpy.ndarray` of shape (Nmodel), optional
        The factor used to scale the model observations to the observed data.
        Returned if `return_scale = True`.

    scale_err : `~numpy.ndarray` of shape (Nmodel), optional
        The error on the factor used to scale the model observations.
        Returned if `return_scale = True`.

    """

    # Call `loglike`.
    results = loglike(data, data_err, data_mask, models, models_err,
                      models_mask, free_scale=free_scale,
                      ignore_model_err=ignore_model_err,
                      dim_prior=dim_prior, ltol=ltol,
                      return_scale=return_scale, *args, **kwargs)

    if not return_scale:
        lnlike, ndim, chi2 = results
        lnprior, lnprob = np.zeros_like(lnlike), lnlike[:]
        return lnprior, lnlike, lnprob, ndim, chi2
    else:
        lnlike, ndim, chi2, scale, scale_err = results
        lnprior, lnprob = np.zeros_like(lnlike), lnlike[:]
        return lnprior, lnlike, lnprob, ndim, chi2, scale, scale_err


def gaussian(mu, std, x):
    """
    Gaussian kernal with mean `mu` and standard deviation `std` evaluated
    over grid `x`. Returns the PDF `N(x | mu, std)`.

    """

    dif = x - mu  # difference
    norm = np.sqrt(2. * np.pi) * std  # normalization
    pdf = np.exp(-0.5 * np.square(dif / std)) / norm

    return pdf


def gaussian_bin(mu, std, bins):
    """
    Gaussian kernal with mean `mu` and standard deviation `std` evaluated
    over a set of bins with edges specified by `bins`.
    Returns the PDF integrated over the bins (i.e. an `N - 1`-length vector).

    """

    dif = bins - mu  # difference
    y = dif / (np.sqrt(2) * std)  # divide by relative width
    cdf = 0.5 * (1. + erf(y))  # CDF evaluated at bin edges
    pdf = cdf[1:] - cdf[:-1]  # amplitude integrated over the bins

    return pdf


def gauss_kde(y, y_std, x, dx=None, y_wt=None, sig_thresh=5., wt_thresh=1e-3,
              cdf_thresh=2e-4, *args, **kwargs):
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

    cdf_thresh : float, optional
        The `1. - cdf_thresh` threshold of the (sorted) CDF used to ignore
        objects with (relatively) negligible weights. This option is only
        used when `wt_thresh=None`. Default is `2e-4`.

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
    if wt_thresh is None and cdf_thresh is None:
        wt_thresh = -np.inf  # default to no clipping/thresholding

    # Clipping kernels.
    centers = np.array((y - x[0]) / dx, dtype='int')  # discretized centers
    offsets = np.array(sig_thresh * y_std / dx, dtype='int')  # offsets
    uppers, lowers = centers + offsets, centers - offsets  # upper/lower bounds
    uppers[uppers > Nx], lowers[lowers < 0] = Nx, 0  # limiting to grid edges

    # Initialize PDF.
    pdf = np.zeros(Nx)

    # Apply thresholding.
    if wt_thresh is not None:
        # Use relative amplitude to threshold.
        sel_arr = np.arange(Ny)[y_wt > (wt_thresh * np.max(y_wt))]
    else:
        # Use CDF to threshold.
        idx_sort = np.argsort(y_wt)  # sort
        y_cdf = np.cumsum(y_wt[idx_sort])  # compute CDF
        y_cdf /= y_cdf[-1]  # normalize
        sel_arr = idx_sort[y_cdf <= (1. - cdf_thresh)]

    # Compute PDF.
    for i in sel_arr:
        # Stack weighted Gaussian kernel over array slice.
        gkde = gaussian(y[i], y_std[i], x[lowers[i]:uppers[i]])
        norm = sum(gkde)
        if norm != 0.:
            pdf[lowers[i]:uppers[i]] += y_wt[i] / norm * gkde

    return pdf


def gauss_kde_dict(pdfdict, y=None, y_std=None, y_idx=None, y_std_idx=None,
                   y_wt=None, wt_thresh=1e-3, cdf_thresh=2e-4,
                   *args, **kwargs):
    """
    Compute smoothed PDF using kernel density estimation based on a
    pre-computed dictionary and pre-defined grid.

    Parameters
    ----------
    pdfdict : :class:`PDFDict` instance
        `PDFDict` instance containing the grid and kernels.

    y, y_std : `~numpy.ndarray` with shape (Ny,), optional
        Array of observed values and associated (Gaussian) errors. Mutually
        exclusive with `y_idx` and `y_std_idx`.

    y_idx, y_std_idx : `~numpy.ndarray` with shape (Ny,), optional
        Array of dictionary indices corresponding to the observed values and
        associated errors. Mutually exclusive with `y` and `y_std`. Preference
        will be given to `y_idx` and `y_std_idx` if provided.

    y_wt : `~numpy.ndarray` with shape (Ny,), optional
        An associated set of weights for each of the elements in `y`. If not
        provided, objects will be weighted uniformly.

    wt_thresh : float, optional
        The threshold `wt_thresh * max(y_wt)` used to ignore objects
        with (relatively) negligible weights. Default is `1e-3`.

    cdf_thresh : float, optional
        The `1. - cdf_thresh` threshold of the (sorted) CDF used to ignore
        objects with (relatively) negligible weights. This option is only
        used when `wt_thresh=None`. Default is `2e-4`.

    Returns
    -------
    pdf : `~numpy.ndarray` with shape (Nx,)
        Probability distribution function (PDF) evaluated over `pdfdict.grid`.

    """

    # Check for valid inputs.
    if y_idx is not None and y_std_idx is not None:
        pass
    elif y is not None and y_std is not None:
        y_idx, y_std_idx = pdfdict.fit(y, y_std)
    else:
        raise ValueError("At least one pair of (`y`, `y_std`) or "
                         "(`y_idx`, `y_idx_std`) must be specified.")
    if wt_thresh is None and cdf_thresh is None:
        wt_thresh = -np.inf  # default to no clipping/thresholding

    # Initialize PDF.
    Nx = pdfdict.Ngrid
    pdf = np.zeros(Nx)

    # Apply weight thresholding.
    Ny = len(y_idx)
    if y_wt is None:
        y_wt = np.ones(Ny)
    if wt_thresh is not None:
        # Use relative amplitude to threshold.
        sel_arr = np.arange(Ny)[y_wt > (wt_thresh * np.max(y_wt))]
    else:
        # Use CDF to threshold.
        idx_sort = np.argsort(y_wt)  # sort
        y_cdf = np.cumsum(y_wt[idx_sort])  # compute CDF
        y_cdf /= y_cdf[-1]  # normalize
        sel_arr = idx_sort[y_cdf <= (1. - cdf_thresh)]

    # Compute PDF.
    sigma_dict = pdfdict.sigma_dict  # Gaussian kernel dictionary
    sigma_width = pdfdict.sigma_width  # number of elements in each kernel
    sigma_dict_cdf = pdfdict.sigma_dict_cdf  # CDF of Gaussian kernel
    for i in sel_arr:
        # Select position and kernel.
        idx = y_std_idx[i]  # dictionary element
        pos = y_idx[i]  # kernel center
        kernel = sigma_dict[idx]  # Gaussian kernel
        width = sigma_width[idx]  # kernel width
        kcdf = sigma_dict_cdf[idx]  # kernel CDF

        # Deal with edge effects.
        low, high = max(pos - width, 0), min(pos + width + 1, Nx)
        lpad, hpad = low - (pos - width), high - (pos + width + 1)
        if lpad == 0:
            norm = kcdf[hpad-1]
        else:
            norm = kcdf[hpad-1] - kcdf[lpad-1]

        # Stack weighted Gaussian kernel over array slice.
        pdf[low:high] += (y_wt[i] / norm) * kernel[lpad:2*width+1+hpad]

    return pdf


def magnitude(phot, err, zeropoints=1., *args, **kwargs):
    """
    Convert photometry to AB magnitudes.

    Parameters
    ----------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux densities.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux density errors.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    mag : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitudes corresponding to input `phot`.

    mag_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitudes errors corresponding to input `err`.

    """

    # Compute magnitudes.
    mag = -2.5 * np.log10(phot / zeropoints)

    # Compute errors.
    mag_err = 2.5 / np.log(10.) * err / phot

    return mag, mag_err


def inv_magnitude(mag, err, zeropoints=1., *args, **kwargs):
    """
    Convert AB magnitudes to photometry.

    Parameters
    ----------
    mag : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitudes.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Magnitude errors.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric flux densities corresponding to input `mag`.

    phot_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric errors corresponding to input `err`.

    """

    # Compute magnitudes.
    phot = 10**(-0.4 * mag) * zeropoints

    # Compute errors.
    phot_err = err * 0.4 * np.log(10.) * phot

    return phot, phot_err


def luptitude(phot, err, skynoise=1., zeropoints=1., *args, **kwargs):
    """
    Convert photometry to asinh magnitudes (i.e. "Luptitudes"). See Lupton et
    al. (1999) for more details.

    Parameters
    ----------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux densities.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Observed photometric flux density errors.

    skynoise : float or `~numpy.ndarray` with shape (Nfilt,)
        Background sky noise. Used as a "softening parameter".
        Default is `1.`.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    mag : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Asinh magnitudes corresponding to input `phot`.

    mag_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Asinh magnitudes errors corresponding to input `err`.

    """

    # Compute asinh magnitudes.
    mag = -2.5 / np.log(10.) * (np.arcsinh(phot / (2. * skynoise)) +
                                np.log(skynoise / zeropoints))

    # Compute errors.
    mag_err = np.sqrt(np.square(2.5 * np.log10(np.e) * err) /
                      (np.square(2. * skynoise) + np.square(phot)))

    return mag, mag_err


def inv_luptitude(mag, err, skynoise=1., zeropoints=1., *args, **kwargs):
    """
    Convert asinh magnitudes ("Luptitudes") to photometry.

    Parameters
    ----------
    mag : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Asinh magnitudes.

    err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Asinh magnitude errors.

    skynoise : float or `~numpy.ndarray` with shape (Nfilt,)
        Background sky noise. Used as a "softening parameter".
        Default is `1.`.

    zeropoints : float or `~numpy.ndarray` with shape (Nfilt,)
        Flux density zero-points. Used as a "location parameter".
        Default is `1.`.

    Returns
    -------
    phot : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric flux densities corresponding to input `mag`.

    phot_err : `~numpy.ndarray` with shape (Nobs, Nfilt)
        Photometric errors corresponding to input `err`.

    """

    # Compute photometry.
    phot = (2. * skynoise) * np.sinh(np.log(10.) / -2.5 * mag -
                                     np.log(skynoise / zeropoints))

    # Compute errors.
    phot_err = np.sqrt((np.square(2. * skynoise) + np.square(phot)) *
                       np.square(err)) / (2.5 * np.log10(np.e))

    return phot, phot_err


class PDFDict():
    """
    Class used to establish a set of underlying grids and Gaussian kernels
    used to quickly compute PDFs. PDFs are computed by sliding, truncating, and
    stacking our kernels along the underlying grid.

    Parameters
    ----------
    pdf_grid : `~numpy.ndarray` of shape (Ngrid,)
        The underlying discretized grid used to evaluate PDFs. **This grid
        must be evenly spaced.**

    sigma_grid : `~numpy.ndarray` of shape (Ndict,)
        The standard deviations used to compute the set of discretized
        Gaussian kernels.

    sigma_trunc : float, optional
        The number of sigma used before truncating our Gaussian kernels.
        Default is `5.`.

    """

    def __init__(self, pdf_grid, sigma_grid, sigma_trunc=5.):

        # Initialize quantities.
        self.Ngrid = len(pdf_grid)
        self.min, self.max = min(pdf_grid), max(pdf_grid)
        self.delta = pdf_grid[1] - pdf_grid[0]
        self.grid = np.array(pdf_grid)

        # Create dictionary.
        self.Ndict = len(sigma_grid)
        self.sigma_grid = np.array(sigma_grid)
        self.dsigma = sigma_grid[1] - sigma_grid[0]
        self.sigma_width = np.array(np.ceil(sigma_grid * sigma_trunc /
                                    self.delta), dtype='int')
        mid = int(self.Ngrid / 2)
        self.sigma_dict = [gaussian(mu=self.grid[mid], std=s,
                                    x=self.grid[mid-w:mid+w+1])
                           for i, (s, w) in enumerate(zip(self.sigma_grid,
                                                          self.sigma_width))]
        self.sigma_dict_cdf = [np.cumsum(p) for p in self.sigma_dict]

    def fit(self, X, Xe):
        """
        Map Gaussian PDFs onto the dictionary.

        Parameters
        ----------
        X : `~numpy.ndarray` of shape (Nobs,)
            Observed values.

        Xe : `~numpy.ndarray` of shape (Nobs,)
            Observed errors.

        Returns
        -------
        X_idx : `~numpy.ndarray` of shape (Nobs,)
            Corresponding indices on the dicretized mean grid.

        Xe_idx : `~numpy.ndarray` of shape (Nobs,)
            Corresponding indices on the discretized sigma grid.

        """

        # Mean indices.
        X_idx = ((X - self.grid[0]) / self.delta).round().astype('int')

        # Sigma (dictionary) indices.
        Xe_idx = np.array(np.round((Xe - self.sigma_grid[0]) / self.dsigma),
                          dtype='int')
        Xe_idx[Xe_idx >= self.Ndict] = self.Ndict - 1  # impose error ceiling
        Xe_idx[Xe_idx < 0] = 0  # impose error floor

        return X_idx, Xe_idx


def pdfs_resample(pdfs, old_grid, new_grid, renormalize=True,
                  left=0., right=0.):
    """
    Resample input PDFs from a given grid onto a new grid. Wraps
    `~numpy.interp`.

    Parameters
    ----------
    pdfs : `~numpy.ndarray` with shape (Npdf, Ngrid)
        Original collection of PDFs.

    old_grid : `~numpy.ndarray` with shape (Ngrid)
        Old grid the PDFs are evaluated over.

    new_grid : `~numpy.ndarray` with shape (Ngrid_new)
        New grid to evaluate the PDFs over.

    renormalize : bool, optional
        Whether to renormalize the PDFs after resampling. Default is `True`.

    left : float, optional
        Value to return beyond the left edge of the grid. Default is `0.`.

    right : float, optional
        Value to return beyond the right edge of the grid. Default is `0.`.

    Returns
    -------
    new_pdfs : `~numpy.ndarray` with shape (Npdf, Ngrid_new)
        Resampled PDFs.

    """

    # Resample PDFs.
    new_pdfs = np.array([np.interp(new_grid, old_grid, pdf,
                                   left=left, right=right) for pdf in pdfs])

    # Renormalize PDFs.
    if renormalize:
        new_pdfs /= new_pdfs.sum(axis=1)[:, None]

    return new_pdfs


def pdfs_summarize(pdfs, pgrid, renormalize=True, rstate=None,
                   pkern='lorentz', pkern_grid=None, wconf_func=None):
    """
    Compute PDF summary statistics. Point estimators include:

    * mean: optimal estimator under L2 loss
    * median: optimal estimator under L1 loss
    * mode: optimal estimator under L0 (pseudo-)loss
    * best: optimal estimator under loss from `pkern` and `pkern_grid`

    Estimators also come with multiple quality metrics attached:

    * std: standard deviation computed around the estimator
    * conf: fraction of the PDF contained within a window around the estimator
    * risk: associated risk computed under the loss from `pkern`

    68% and 95% lower/upper credible intervals are also reported.

    For statistical purposes, a Monte Carlo realization of the posterior
    is also generated.

    Based on code from Sogo Mineo and Atsushi Nishizawa and used in the
    HSC-SSP DR1 photo-z release.

    Parameters
    ----------
    pdfs : `~numpy.ndarray` with shape (Npdf, Ngrid)
        Original collection of PDFs.

    pgrid : `~numpy.ndarray` with shape (Ngrid)
        Grid the PDFs are evaluated over.

    renormalize : bool, optional
        Whether to renormalize the PDFs before computation. Default is `True`.

    rstate : `~numpy.random.RandomState` instance, optional
        Random state instance. If not passed, the default `~numpy.random`
        instance will be used.

    pkern : str or func, optional
        The kernel used to compute the effective loss over the grid when
        computing the `best` estimator. Default is `'lorentz'`.

    pkern_grid : `~numpy.ndarray` with shape (Ngrid, Ngrid), optional
        The 2-D array of positions that `pkern` is evaluated over.
        If not provided, a `1. / ((1. + x) * sig)` weighting over `pgrid`
        will be used, where `sig = 0.15`. **Note that this is designed for
        photo-z estimation and will not be suitable for most problems.**

    wconf_func : func, optional
        A function that takes an input point and generates an associated
        +/- width value. Used to construct `conf` estimates.

    Returns
    -------
    (pmean, pmean_std, pmean_conf, pmean_risk) : 4-tuple with `~numpy.ndarray`
    elements of shape (Nobj)
        Mean estimator and associated uncertainty/quality assessments.

    (pmed, pmed_std, pmed_conf, pmed_risk) : 4-tuple with `~numpy.ndarray`
    elements of shape (Nobj)
        Median estimator and associated uncertainty/quality assessments.

    (pmode, pmode_std, pmode_conf, pmode_risk) : 4-tuple with `~numpy.ndarray`
    elements of shape (Nobj)
        Mode estimator and associated uncertainty/quality assessments.

    (pbest, pbest_std, pbest_conf, pbest_risk) : 4-tuple with `~numpy.ndarray`
    elements of shape (Nobj)
        "Best" estimator and associated uncertainty/quality assessments.

    (plow95, plow68, phigh68, phigh95) : 4-tuple with `~numpy.ndarray` elements
    of shape (Nobj)
        Lower 95%, lower 68%, upper 68%, and upper 95% quantiles.

    pmc : `~numpy.ndarray` of shape (Nobj)
        Monte Carlo realization of the posterior.

    """

    if rstate is None:
        rstate = np.random

    Nobj, Ngrid = len(pdfs), len(pgrid)
    if renormalize:
        pdfs /= pdfs.sum(axis=1)[:, None]  # sum to 1

    # Compute mean.
    pmean = np.dot(pdfs, pgrid)

    # Compute mode.
    pmode = pgrid[np.argmax(pdfs, axis=1)]

    # Compute CDF-based quantities.
    cdfs = pdfs.cumsum(axis=1)
    plow2, phigh2 = np.zeros(Nobj), np.zeros(Nobj)  # +/- 95%
    plow1, phigh1 = np.zeros(Nobj), np.zeros(Nobj)  # +/- 68%
    pmed = np.zeros(Nobj)  # median
    pmc = np.zeros(Nobj)  # Monte Carlo realization
    for i, cdf in enumerate(cdfs):
        qs = [0.025, 0.16, 0.5, 0.84, 0.975, rstate.rand()]  # quantiles
        qvals = np.interp(qs, cdf, pgrid)
        plow2[i], plow1[i], pmed[i], phigh1[i], phigh2[i], pmc[i] = qvals

    # Compute kernel-based quantities.
    if pkern_grid is None:
        # Structure grid of "truth" values and "guess" values.
        # **Designed for photo-z estimation -- likely not applicable in most
        # other applications.**
        ptrue = pgrid.reshape(Ngrid, 1)
        pguess = pgrid.reshape(1, Ngrid)
        psig = 0.15  # kernel dispersion
        pkern_grid = (ptrue - pguess) / ((1. + ptrue) * 0.15)
    if pkern == 'tophat':
        # Use top-hat kernel
        kernel = (np.square(pkern_grid) < 1.)
    elif pkern == 'gaussian':
        kernel = np.exp(-0.5 * np.square(pkern_grid))
    elif pkern == 'lorentz':
        kernel = 1. / (1. + np.square(pkern_grid))
    else:
        try:
            kernel = pkern(pkern_grid)
        except:
            raise RuntimeError("The input kernel does not appear to be valid.")
    prisk = np.dot(pdfs, 1.0 - kernel)  # "risk" estimator
    pbest = pgrid[np.argmin(prisk, axis=1)]  # "best" estimator

    # Compute second moment uncertainty estimate (i.e. std-dev).
    grid = pgrid.reshape(1, Ngrid)
    sqdev = np.square(grid - pmean.reshape(Nobj, 1))  # mean
    pmean_std = np.sqrt(np.sum(sqdev * pdfs, axis=1))
    sqdev = np.square(grid - pmed.reshape(Nobj, 1))  # med
    pmed_std = np.sqrt(np.sum(sqdev * pdfs, axis=1))
    sqdev = np.square(grid - pmode.reshape(Nobj, 1))  # mode
    pmode_std = np.sqrt(np.sum(sqdev * pdfs, axis=1))
    sqdev = np.square(grid - pbest.reshape(Nobj, 1))  # best
    pbest_std = np.sqrt(np.sum(sqdev * pdfs, axis=1))

    # Construct "confidence" estimates around our primary point estimators
    # (i.e. how much of the PDF is contained within +/= some fixed interval).
    if wconf_func is None:
        def wconf_func(point):
            return (1. + point) * 0.03
    pmean_conf, pmed_conf, pmode_conf, pbest_conf = np.zeros((4, Nobj))
    for i, cdf in enumerate(cdfs):
        # Mean
        width = wconf_func(pmean[i])
        pmean_low, pmean_high = pmean[i] - width, pmean[i] + width
        # Median
        width = wconf_func(pmed[i])
        pmed_low, pmed_high = pmed[i] - width, pmed[i] + width
        # Mode
        width = wconf_func(pmode[i])
        pmode_low, pmode_high = pmode[i] - width, pmode[i] + width
        # "Best"
        width = wconf_func(pbest[i])
        pbest_low, pbest_high = pbest[i] - width, pbest[i] + width
        # Interpolate CDFs.
        qs = np.array([pmean_low, pmean_high, pmed_low, pmed_high, pmode_low,
                       pmode_high, pbest_low, pbest_high])
        qvs = np.interp(qs, pgrid, cdf)
        (pmean_conf[i], pmed_conf[i],
         pmode_conf[i], pbest_conf[i]) = qvs[[1, 3, 5, 7]] - qvs[[0, 2, 4, 6]]

    # Construct "risk" estimates around our primary point estimators.
    pmean_risk, pmed_risk, pmode_risk, pbest_risk = np.zeros((4, Nobj))
    for i, pr in enumerate(prisk):
        vals = np.interp([pmean[i], pmed[i], pmode[i], pbest[i]], pgrid, pr)
        pmean_risk[i], pmed_risk[i], pmode_risk[i], pbest_risk[i] = vals

    return ((pmean, pmean_std, pmean_conf, pmean_risk),
            (pmed, pmed_std, pmed_conf, pmed_risk),
            (pmode, pmode_std, pmode_conf, pmode_risk),
            (pbest, pbest_std, pbest_conf, pbest_risk),
            (plow2, plow1, phigh1, phigh2), pmc)
