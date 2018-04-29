#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Object used to fit data and compute PDFs using brute-force methods.

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

from .pdf import *

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

__all__ = ["BruteForce"]


class BruteForce():
    """
    Fits data and generates predictions using a simple brute-force approach.

    """

    def __init__(self, models, models_err, models_mask):
        """
        Load the model data into memory.

        Parameters
        ----------
        models : `~numpy.ndarray` of shape (Nmodel, Nfilt)
            Model values.

        models_err : `~numpy.ndarray` of shape (Nmodel, Nfilt)
            Associated errors on the model values.

        models_mask : `~numpy.ndarray` of shape (Nmodel, Nfilt)
            Binary mask (0/1) indicating whether the model value was observed.

        """

        # Initialize values.
        self.models = models
        self.models_err = models_err
        self.models_mask = models_mask
        self.NMODEL, self.NDIM = models.shape
        self.fit_lnprior = None
        self.fit_lnlike = None
        self.fit_lnprob = None
        self.fit_Ndim = None
        self.fit_chi2 = None
        self.fit_scale = None
        self.fit_scale_err = None

        self.NMODEL, self.NDIM = models.shape

    def fit(self, data, data_err, data_mask, lprob_func=None,
            lprob_args=None, lprob_kwargs=None, track_scale=False,
            verbose=True):
        """
        Fit all input models to the input data to compute the associated
        log-posteriors.

        Parameters
        ----------
        data : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Model values.

        data_err : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Associated errors on the data values.

        data_mask : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Binary mask (0/1) indicating whether the data value was observed.

        lprob_func : str or func, optional
            Log-posterior function to be used. Must return ln(prior), ln(like),
            ln(post), Ndim, chi2, and (optionally) scale and std(scale).
            If not provided, `~frankenz.pdf.logprob` will be used.

        lprob_args : args, optional
            Arguments to be passed to `lprob_func`.

        lprob_kwargs : kwargs, optional
            Keyword arguments to be passed to `lprob_func`.

        track_scale : bool, optional
            Whether `lprob_func` also returns the scale-factor. Default is
            `False`.

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

        """

        # Initialize values.
        if lprob_func is None:
            lprob_func = logprob
        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()
        Ndata = len(data)

        # Fit data.
        for i, results in enumerate(self._fit(data, data_err, data_mask,
                                              lprob_func=lprob_func,
                                              lprob_args=lprob_args,
                                              lprob_kwargs=lprob_kwargs,
                                              track_scale=track_scale,
                                              save_fits=True)):
            if verbose:
                sys.stderr.write('\rFitting object {0}/{1}'.format(i+1, Ndata))
                sys.stderr.flush()
        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

    def _fit(self, data, data_err, data_mask, lprob_func=None,
             lprob_args=None, lprob_kwargs=None, track_scale=False,
             save_fits=True):
        """
        Internal generator used to compute fits.

        Parameters
        ----------
        data : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Model values.

        data_err : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Associated errors on the data values.

        data_mask : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Binary mask (0/1) indicating whether the data value was observed.

        lprob_func : str or func, optional
            Log-posterior function to be used. Must return ln(prior), ln(like),
            ln(post), Ndim, chi2, and (optionally) scale and std(scale).
            If not provided, `~frankenz.pdf.logprob` will be used.

        lprob_args : args, optional
            Arguments to be passed to `lprob_func`.

        lprob_kwargs : kwargs, optional
            Keyword arguments to be passed to `lprob_func`.

        track_scale : bool, optional
            Whether `lprob_func` also returns the scale-factor. Default is
            `False`.

        save_fits : bool, optional
            Whether to save fits internally while computing predictions.
            Default is `True`.

        Returns
        -------
        results : tuple
            Output of `lprob_func` yielded from the generator.

        """

        # Initialize values.
        if lprob_func is None:
            lprob_func = logprob
        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()

        Ndata = len(data)
        Nmodels = self.NMODEL
        self.NDATA = Ndata

        if save_fits:
            self.fit_lnprior = np.zeros((Ndata, Nmodels), dtype='float')
            self.fit_lnlike = np.zeros((Ndata, Nmodels), dtype='float')
            self.fit_lnprob = np.zeros((Ndata, Nmodels), dtype='float')
            self.fit_Ndim = np.zeros((Ndata, Nmodels), dtype='int')
            self.fit_chi2 = np.zeros((Ndata, Nmodels), dtype='float')
            self.fit_scale = np.ones((Ndata, Nmodels), dtype='float')
            self.fit_scale_err = np.zeros((Ndata, Nmodels), dtype='float')

        # Fit data.
        for i, (x, xe, xm) in enumerate(zip(data, data_err, data_mask)):
            results = lprob_func(x, xe, xm, self.models, self.models_err,
                                 self.models_mask, *lprob_args, **lprob_kwargs)
            if save_fits:
                self.fit_lnprior[i] = results[0]  # ln(prior)
                self.fit_lnlike[i] = results[1]  # ln(like)
                self.fit_lnprob[i] = results[2]  # ln(prob)
                self.fit_Ndim[i] = results[3]  # dimensionality of fit
                self.fit_chi2[i] = results[4]  # chi2
                if track_scale:
                    self.fit_scale[i] = results[5]  # scale-factor
                    self.fit_scale_err[i] = results[6]  # std(s)

            yield results

    def predict(self, model_labels, model_label_errs, label_dict=None,
                label_grid=None, logwt=None, kde_args=None, kde_kwargs=None,
                return_gof=False, verbose=True):
        """
        Compute photometric 1-D predictions to the target distribution.

        Parameters
        ----------
        model_labels : `~numpy.ndarray` of shape (Nmodel)
            Model values.

        model_label_errs : `~numpy.ndarray` of shape (Nmodel)
            Associated errors on the data values.

        label_dict : `~frankenz.pdf.PDFDict` object, optional
            Dictionary of pre-computed stationary kernels. If provided,
            :meth:`~frankenz.pdf.gauss_kde_dict` will be used for KDE.

        label_grid : `~numpy.ndarray` of shape (Ngrid), optional
            Grid points to evaluate the 1-D PDFs over. Only used when
            `label_dict` is not provided, at which point
            :meth:`~frankenz.pdf.gauss_kde` will be used for KDE.

        logwt : `~numpy.ndarray` of shape (Ndata, Nmodel), optional
            A new set of log-weights used to compute the marginalized 1-D
            PDFs in place of the log-probability.

        kde_args : args, optional
            Arguments to be passed to the KDE function.

        kde_kwargs : kwargs, optional
            Keyword arguments to be passed to the KDE function.

        return_gof : bool, optional
            Whether to return a tuple containing the ln(MAP) and
            ln(evidence) values for the predictions
            along with the pdfs. Default is `False`.

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

        Returns
        -------
        pdfs : `~numpy.ndarray` of shape (Nobj, Ngrid)
            Collection of 1-D PDFs for each object.

        (lmap, levid) : 2-tuple of `~numpy.ndarray` with shape (Nobj), optional
            Set of ln(MAP) and ln(evidence) values for each object.

        """

        # Initialize values.
        if kde_args is None:
            kde_args = []
        if kde_kwargs is None:
            kde_kwargs = dict()
        if logwt is None:
            logwt = self.fit_lnprob
        if label_dict is None and label_grid is None:
            raise ValueError("`label_dict` or `label_grid` must be specified.")
        if self.fit_lnprob is None and logwt is None:
            raise ValueError("Fits have not been computed and weights have "
                             "not been provided.")
        if label_dict is not None:
            Nx = label_dict.Ngrid
        else:
            Nx = len(label_grid)
        Ndata = self.NDATA
        pdfs = np.zeros((Ndata, Nx))
        if return_gof:
            lmap = np.zeros(Ndata)
            levid = np.zeros(Ndata)

        # Compute PDFs.
        for i, res in enumerate(self._predict(model_labels, model_label_errs,
                                              label_dict=label_dict,
                                              label_grid=label_grid,
                                              logwt=logwt, kde_args=kde_args,
                                              kde_kwargs=kde_kwargs)):
            pdf, gof = res
            pdfs[i] = pdf
            if return_gof:
                lmap[i], levid[i] = gof
            if verbose:
                sys.stderr.write('\rGenerating PDF {0}/{1}'
                                 .format(i+1, Ndata))
                sys.stderr.flush()
        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

        if return_gof:
            return pdfs, (lmap, levid)
        else:
            return pdfs

    def _predict(self, model_labels, model_label_errs, label_dict=None,
                 label_grid=None, logwt=None, kde_args=None, kde_kwargs=None):
        """
        Internal generator used to compute photometric 1-D predictions.

        Parameters
        ----------
        model_labels : `~numpy.ndarray` of shape (Nmodel)
            Model values.

        model_label_errs : `~numpy.ndarray` of shape (Nmodel)
            Associated errors on the data values.

        label_dict : `~frankenz.pdf.PDFDict` object, optional
            Dictionary of pre-computed stationary kernels. If provided,
            :meth:`~frankenz.pdf.gauss_kde_dict` will be used for KDE.

        label_grid : `~numpy.ndarray` of shape (Ngrid), optional
            Grid points to evaluate the 1-D PDFs over. Only used when
            `label_dict` is not provided, at which point
            :meth:`~frankenz.pdf.gauss_kde` will be used for KDE.

        logwt : `~numpy.ndarray` of shape (Ndata, Nmodel), optional
            A new set of log-weights used to compute the marginalized 1-D
            PDFs in place of the log-posterior.

        kde_args : args, optional
            Arguments to be passed to the KDE function.

        kde_kwargs : kwargs, optional
            Keyword arguments to be passed to the KDE function.

        Returns
        -------
        pdf : `~numpy.ndarray` of shape (Ngrid)
            1-D PDF yielded by the generator.

        (lmap, levid) : 2-tuple of floats
            ln(MAP) and ln(evidence) values yielded by the generator.

        """

        # Initialize values.
        if kde_args is None:
            kde_args = []
        if kde_kwargs is None:
            kde_kwargs = dict()
        if logwt is None:
            logwt = self.fit_lnprob
        if label_dict is None and label_grid is None:
            raise ValueError("`label_dict` or `label_grid` must be specified.")
        if label_dict is not None:
            y_idx, y_std_idx = label_dict.fit(model_labels, model_label_errs)

        # Generate PDFs.
        for i, lwt in enumerate(logwt):
            lmap, levid = max(lwt), logsumexp(lwt)
            wt = np.exp(lwt - levid)
            if label_dict is not None:
                # Use dictionary if available.
                pdf = gauss_kde_dict(label_dict, y_idx=y_idx,
                                     y_std_idx=y_std_idx, y_wt=wt,
                                     *kde_args, **kde_kwargs)
            else:
                # Otherwise just use KDE.
                pdf = gauss_kde(model_labels, model_label_errs, label_grid,
                                y_wt=wt, *kde_args, **kde_kwargs)

            yield pdf, (lmap, levid)

    def fit_predict(self, data, data_err, data_mask, model_labels,
                    model_label_errs, lprob_func=None, label_dict=None,
                    label_grid=None, kde_args=None, kde_kwargs=None,
                    lprob_args=None, lprob_kwargs=None, return_gof=False,
                    track_scale=False, verbose=True, save_fits=True):
        """
        Fit all input models to the input data to compute the associated
        log-posteriors and 1-D predictions.

        Parameters
        ----------
        data : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Model values.

        data_err : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Associated errors on the data values.

        data_mask : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Binary mask (0/1) indicating whether the data value was observed.

        model_labels : `~numpy.ndarray` of shape (Nmodel)
            Model values.

        model_label_errs : `~numpy.ndarray` of shape (Nmodel)
            Associated errors on the data values.

        lprob_func : str or func, optional
            Log-posterior function to be used. Must return ln(prior), ln(like),
            ln(post), Ndim, chi2, and (optionally) scale and std(scale).
            If not provided, `~frankenz.pdf.logprob` will be used.

        label_dict : `~frankenz.pdf.PDFDict` object, optional
            Dictionary of pre-computed stationary kernels. If provided,
            :meth:`~frankenz.pdf.gauss_kde_dict` will be used for KDE.

        label_grid : `~numpy.ndarray` of shape (Ngrid), optional
            Grid points to evaluate the 1-D PDFs over. Only used when
            `label_dict` is not provided, at which point
            :meth:`~frankenz.pdf.gauss_kde` will be used for KDE.

        kde_args : args, optional
            Arguments to be passed to the KDE function.

        kde_kwargs : kwargs, optional
            Keyword arguments to be passed to the KDE function.

        lprob_args : args, optional
            Arguments to be passed to `lprob_func`.

        lprob_kwargs : kwargs, optional
            Keyword arguments to be passed to `lprob_func`.

        return_gof : bool, optional
            Whether to return a tuple containing the ln(MAP) and
            ln(evidence) values for the predictions
            along with the pdfs. Default is `False`.

        track_scale : bool, optional
            Whether `lprob_func` also returns the scale-factor. Default is
            `False`.

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

        save_fits : bool, optional
            Whether to save fits internally while computing predictions.
            Default is `True`.

        Returns
        -------
        pdfs : `~numpy.ndarray` of shape (Nobj, Ngrid)
            Collection of 1-D PDFs for each object.

        (lmap, levid) : 2-tuple of `~numpy.ndarray` with shape (Nobj), optional
            Set of ln(MAP) and ln(evidence) values for each object.

        """

        # Initialize values.
        if lprob_func is None:
            lprob_func = logprob
        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()
        if kde_args is None:
            kde_args = []
        if kde_kwargs is None:
            kde_kwargs = dict()
        if label_dict is None and label_grid is None:
            raise ValueError("`label_dict` or `label_grid` must be specified.")
        if label_dict is not None:
            Nx = label_dict.Ngrid
        else:
            Nx = len(label_grid)
        Ndata = len(data)
        pdfs = np.zeros((Ndata, Nx))
        if return_gof:
            lmap = np.zeros(Ndata)
            levid = np.zeros(Ndata)

        # Generate predictions.
        for i, res in enumerate(self._fit_predict(data, data_err, data_mask,
                                                  model_labels,
                                                  model_label_errs,
                                                  lprob_func=lprob_func,
                                                  label_dict=label_dict,
                                                  label_grid=label_grid,
                                                  kde_args=kde_args,
                                                  kde_kwargs=kde_kwargs,
                                                  lprob_args=lprob_args,
                                                  lprob_kwargs=lprob_kwargs,
                                                  track_scale=track_scale,
                                                  save_fits=save_fits)):
            pdf, gof = res
            pdfs[i] = pdf
            if return_gof:
                lmap[i], levid[i] = gof  # save gof metrics
            if verbose:
                sys.stderr.write('\rGenerating PDF {0}/{1}'
                                 .format(i+1, Ndata))
                sys.stderr.flush()
        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

        if return_gof:
            return pdfs, (lmap, levid)
        else:
            return pdfs

    def _fit_predict(self, data, data_err, data_mask, model_labels,
                     model_label_errs, lprob_func=None, label_dict=None,
                     label_grid=None, kde_args=None, kde_kwargs=None,
                     lprob_args=None, lprob_kwargs=None,
                     track_scale=False, save_fits=True):
        """
        Internal generator used to fit and compute predictions.

        Parameters
        ----------
        data : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Model values.

        data_err : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Associated errors on the data values.

        data_mask : `~numpy.ndarray` of shape (Ndata, Nfilt)
            Binary mask (0/1) indicating whether the data value was observed.

        model_labels : `~numpy.ndarray` of shape (Nmodel)
            Model values.

        model_label_errs : `~numpy.ndarray` of shape (Nmodel)
            Associated errors on the data values.

        lprob_func : str or func, optional
            Log-posterior function to be used. Must return ln(prior), ln(like),
            ln(post), Ndim, chi2, and (optionally) scale and std(scale).
            If not provided, `~frankenz.pdf.logprob` will be used.

        label_dict : `~frankenz.pdf.PDFDict` object, optional
            Dictionary of pre-computed stationary kernels. If provided,
            :meth:`~frankenz.pdf.gauss_kde_dict` will be used for KDE.

        label_grid : `~numpy.ndarray` of shape (Ngrid), optional
            Grid points to evaluate the 1-D PDFs over. Only used when
            `label_dict` is not provided, at which point
            :meth:`~frankenz.pdf.gauss_kde` will be used for KDE.

        kde_args : args, optional
            Arguments to be passed to the KDE function.

        kde_kwargs : kwargs, optional
            Keyword arguments to be passed to the KDE function.

        lprob_args : args, optional
            Arguments to be passed to `lprob_func`.

        lprob_kwargs : kwargs, optional
            Keyword arguments to be passed to `lprob_func`.

        track_scale : bool, optional
            Whether `lprob_func` also returns the scale-factor. Default is
            `False`.

        save_fits : bool, optional
            Whether to save fits internally while computing predictions.
            Default is `True`.

        Returns
        -------
        pdfs : `~numpy.ndarray` of shape (Ngrid)
            1-D PDF for each object yielded by the generator.

        (lmap, levid) : 2-tuple of floats
            ln(MAP) and ln(evidence) values for each object.

        """

        # Initialize values.
        if lprob_func is None:
            lprob_func = logprob
        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()
        if kde_args is None:
            kde_args = []
        if kde_kwargs is None:
            kde_kwargs = dict()
        if label_dict is None and label_grid is None:
            raise ValueError("`label_dict` or `label_grid` must be specified.")
        Ndata = len(data)
        Nmodels = self.NMODEL
        if save_fits:
            self.fit_lnprior = np.zeros((Ndata, Nmodels), dtype='float')
            self.fit_lnlike = np.zeros((Ndata, Nmodels), dtype='float')
            self.fit_lnprob = np.zeros((Ndata, Nmodels), dtype='float')
            self.fit_Ndim = np.zeros((Ndata, Nmodels), dtype='int')
            self.fit_chi2 = np.zeros((Ndata, Nmodels), dtype='float')
            self.fit_scale = np.ones((Ndata, Nmodels), dtype='float')
            self.fit_scale_err = np.zeros((Ndata, Nmodels), dtype='float')
            self.NDATA = Ndata
        if label_dict is not None:
            y_idx, y_std_idx = label_dict.fit(model_labels, model_label_errs)

        # Run generator.
        for i, (x, xe, xm) in enumerate(zip(data, data_err, data_mask)):

            # Compute fit.
            results = lprob_func(x, xe, xm, self.models, self.models_err,
                                 self.models_mask, *lprob_args, **lprob_kwargs)
            if save_fits:
                self.fit_lnprior[i] = results[0]  # ln(prior)
                self.fit_lnlike[i] = results[1]  # ln(like)
                self.fit_lnprob[i] = results[2]  # ln(prob)
                self.fit_Ndim[i] = results[3]  # dimensionality of fit
                self.fit_chi2[i] = results[4]  # chi2
                if track_scale:
                    self.fit_scale[i] = results[5]  # scale-factor
                    self.fit_scale_err[i] = results[6]  # std(s)
            lnprob = results[2]

            # Compute PDF and GOF metrics.
            lmap, levid = max(lnprob), logsumexp(lnprob)
            wt = np.exp(lnprob - levid)
            if label_dict is not None:
                pdf = gauss_kde_dict(label_dict, y_idx=y_idx,
                                     y_std_idx=y_std_idx, y_wt=wt,
                                     *kde_args, **kde_kwargs)
            else:
                pdf = gauss_kde(model_labels, model_label_errs,
                                label_grid, y_wt=wt,
                                *kde_args, **kde_kwargs)

            yield pdf, (lmap, levid)
