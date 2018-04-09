#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Nearest neighbor searches.

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
from scipy.spatial import KDTree
from pandas import unique

from .pdf import *

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

__all__ = ["BruteForce", "KMCkNN"]


class BruteForce():
    """
    Fits all input data using a simple brute-force approach.

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
        self.fit_lnprior = None
        self.fit_lnlike = None
        self.fit_lnprob = None
        self.fit_Ndim = None
        self.fit_chi2 = None
        self.fit_scale = None

        self.NMODEL, self.NDIM = models.shape

    def fit(self, data, data_err, data_mask, lprob_func,
            lprob_args=None, lprob_kwargs=None, return_scale=False,
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
            ln(post), Ndim, chi2, and (optionally) scale.

        lprob_args : args, optional
            Arguments to be passed to `lprob_func`.

        lprob_kwargs : kwargs, optional
            Keyword arguments to be passed to `lprob_func`.

        return_scale : bool, optional
            Whether `lprob_func` also returns the scale-factor. Default is
            `False`.

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

        """

        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()
        Ndata = len(data)

        # Fit data.
        for i, results in enumerate(self._fit(data, data_err, data_mask,
                                              lprob_func,
                                              lprob_args=lprob_args,
                                              lprob_kwargs=lprob_kwargs,
                                              return_scale=return_scale)):
            if verbose:
                sys.stderr.write('\rFitting object {0}/{1}'.format(i+1, Ndata))
                sys.stderr.flush()
        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

    def _fit(self, data, data_err, data_mask, lprob_func,
             lprob_args=None, lprob_kwargs=None, return_scale=False):
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
            ln(prob), Ndim, chi2, and (optionally) scale.

        lprob_args : args, optional
            Arguments to be passed to `lprob_func`.

        lprob_kwargs : kwargs, optional
            Keyword arguments to be passed to `lprob_func`.

        return_scale : bool, optional
            Whether `lprob_func` also returns the scale-factor. Default is
            `False`.

        """

        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()

        # Initialize values.
        Ndata = len(data)
        Nmodels = self.NMODEL
        self.NDATA = Ndata
        self.fit_lnprior = np.zeros((Ndata, Nmodels), dtype='float')
        self.fit_lnlike = np.zeros((Ndata, Nmodels), dtype='float')
        self.fit_lnprob = np.zeros((Ndata, Nmodels), dtype='float')
        self.fit_Ndim = np.zeros((Ndata, Nmodels), dtype='int')
        self.fit_chi2 = np.zeros((Ndata, Nmodels), dtype='float')
        self.fit_scale = np.ones((Ndata, Nmodels), dtype='float')

        for i, (x, xe, xm) in enumerate(zip(data, data_err, data_mask)):
            results = lprob_func(x, xe, xm, self.models, self.models_err,
                                 self.models_mask, *lprob_args, **lprob_kwargs)
            self.fit_lnprior[i] = results[0]  # ln(prior)
            self.fit_lnlike[i] = results[1]  # ln(like)
            self.fit_lnprob[i] = results[2]  # ln(prob)
            self.fit_Ndim[i] = results[3]  # dimensionality of fit
            self.fit_chi2[i] = results[4]  # chi2
            if return_scale:
                self.fit_scale[i] = results[5]  # scale-factor

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

        """

        if kde_args is None:
            kde_args = []
        if kde_kwargs is None:
            kde_kwargs = dict()
        if logwt is None:
            logwt = self.fit_lnprob
        if label_dict is None and label_grid is None:
            raise ValueError("`label_dict` or `label_grid` must be specified.")

        if label_dict is not None:
            for i, lwt in enumerate(logwt):
                lmap, levid = max(lwt), logsumexp(lwt)
                wt = np.exp(lwt - levid)
                pdf = gauss_kde_dict(label_dict, y=model_labels,
                                     y_std=model_label_errs, y_wt=wt,
                                     *kde_args, **kde_kwargs)
                yield pdf, (lmap, levid)
        else:
            for i, lwt in enumerate(logwt):
                lmap, levid = max(lwt), logsumexp(lwt)
                wt = np.exp(lwt - levid)
                pdf = gauss_kde(model_labels, model_label_errs, label_grid,
                                y_wt=wt, *kde_args, **kde_kwargs)
                yield pdf, (lmap, levid)

    def fit_predict(self, data, data_err, data_mask, lprob_func,
                    model_labels, model_label_errs, label_dict=None,
                    label_grid=None, kde_args=None, kde_kwargs=None, 
                    lprob_args=None, lprob_kwargs=None, return_gof=False,
                    return_scale=False, verbose=True, save_fits=True):
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
            ln(post), Ndim, chi2, and (optionally) scale.

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

        return_scale : bool, optional
            Whether `lprob_func` also returns the scale-factor. Default is
            `False`.

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

        save_fits : bool, optional
            Whether to save fits internally while computing predictions.
            Default is `True`.

        """

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

        for i, res in enumerate(self._fit_predict(data, data_err, data_mask,
                                                 lprob_func, model_labels,
                                                 model_label_errs,
                                                 label_dict=label_dict,
                                                 label_grid=label_grid,
                                                 kde_args=kde_args,
                                                 kde_kwargs=kde_kwargs, 
                                                 lprob_args=lprob_args,
                                                 lprob_kwargs=lprob_kwargs,
                                                 return_scale=return_scale,
                                                 save_fits=save_fits)):
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

    def _fit_predict(self, data, data_err, data_mask, lprob_func,
                     model_labels, model_label_errs, label_dict=None,
                     label_grid=None, kde_args=None, kde_kwargs=None, 
                     lprob_args=None, lprob_kwargs=None,
                     return_scale=False, save_fits=True):
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

        lprob_func : str or func, optional
            Log-posterior function to be used. Must return ln(prior), ln(like),
            ln(post), Ndim, chi2, and (optionally) scale.

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

        kde_args : args, optional
            Arguments to be passed to the KDE function.

        kde_kwargs : kwargs, optional
            Keyword arguments to be passed to the KDE function.

        lprob_args : args, optional
            Arguments to be passed to `lprob_func`.

        lprob_kwargs : kwargs, optional
            Keyword arguments to be passed to `lprob_func`.

        return_scale : bool, optional
            Whether `lprob_func` also returns the scale-factor. Default is
            `False`.

        save_fits : bool, optional
            Whether to save fits internally while computing predictions.
            Default is `True`.

        """

        # Initialize values.
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
            self.NDATA = Ndata

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
                if return_scale:
                    self.fit_scale[i] = results[5]  # scale-factor
            lnprob = results[2]

            # Compute PDF.
            lmap, levid = max(lnprob), logsumexp(lnprob)
            wt = np.exp(lnprob - levid)
            if label_dict is not None:
                pdf = gauss_kde_dict(label_dict, y=model_labels,
                                     y_std=model_label_errs, y_wt=wt,
                                     *kde_args, **kde_kwargs)
                yield pdf, (lmap, levid)
            else:
                pdf = gauss_kde(model_labels, model_label_errs,
                                label_grid, y_wt=wt,
                                *kde_args, **kde_kwargs)
                yield pdf, (lmap, levid)


class KMCkNN():
    """
    Locates a set of `K * k` nearest neighbors using Monte Carlo methods to
    incorporate measurement errors over `K` members of an ensemble.
    Wraps `~scipy.spatial.KDTree`. Note that trees are only trained when
    searching for neighbors to save memory.

    Parameters
    ----------
    leafsize : int, optional
        The number of points where the algorithm switches over to brute force.
        Default is `100`.

    K : int, optional
        The number of members used in the ensemble to incorporate errors using
        Monte Carlo methods. Default is `50`.

    k : int, optional
        The number of nearest neighbors selected by each member. Default is
        `20`.

    eps : float, optional
        If supplied, approximate (rather than exact) nearest neighbor queries
        are returned where the `k`th neighbor is guaranteed to be no further
        than `(1 + eps)` times the distance to the *real* `k`th nearest
        neighbor. Default is `1e-3`.

    p : float, optional
        The Minkowski p-norm that should be used to compute distances. Default
        is `2` (i.e. the Euclidean distance).

    distance_upper_bound : float, optional
        If supplied, return only neighbors within this distance. Default is
        `np.inf`.

    """

    def __init__(self, leafsize=100, K=25, k=20, eps=1e-3, p=2,
                 distance_upper_bound=np.inf):
        # Initialize values.
        self.leafsize = leafsize
        self.K = K
        self.k = k
        self.eps = eps
        self.p = p
        self.dbound = distance_upper_bound

    def query(self, X_train, Xe_train, X_targ, Xe_targ,
              feature_map='asinh_mag', rstate=None):
        """
        Find (at most) `K * k` unique neighbors for each training object.

        Parameters
        ----------
        X_train : `~numpy.ndarray` with shape (Ntrain, Nfilt,)
            Training features.

        Xe_train : `~numpy.ndarray` with shape (Ntrain, Nfilt,)
            Training feature errors.

        X_targ : `~numpy.ndarray` with shape (Ntest, Nfilt,)
            Target features.

        Xe_targ : `~numpy.ndarray` with shape (Ntest, Nfilt,)
            Target feature errors.

        feature_map : str or function, optional
            Function that transforms the input set of features/errors `X, Xe`
            to a new set of features/errors `Y, Ye` to facilitate nearest
            neighbor searches. Built-in options are `None` (the identity
            function) and `'asinh_mag'` (asinh magnitudes).
            Default is `'asinh_mag'`.

        rstate : `~numpy.random.RandomState` instance, optional
            Random state instance. If not passed, the default `~numpy.random`
            instance will be used.

        Returns
        -------
        idxs :  `~numpy.ndarray` with shape (Ntest, M*k,)
            Indices for each target object corresponding to the associated
            `K * k` neighbors among the training objects.

        Nidxs : `~numpy.ndarray` with shape (Ntest,)
            Number of unique indices for each target object.

        """

        # Initialize feature map.
        if feature_map is None or feature_map == 'None':
            # Identity function.
            def feature_map(x):
                return x
        elif feature_map == 'asinh_mag':
            # Asinh mags (Luptitudes).
            feature_map = asinh_mag
        else:
            try:
                # Check if `feature_map` is a function.
                feature_map(X_train[:5], Xe_train[:5])
            except:
                # If all else fails, raise an exception.
                raise ValueError("The provided feature map is not valid.")

        # Initialize RNG.
        if rstate is None:
            rstate = np.random

        # Initialize values.
        Ntrain, Ntarg = len(X_train), len(X_targ)  # train/target size
        Npred = self.K * self.k  # number of total possible neighbors
        indices = np.empty((self.K, Ntarg, self.k), dtype='int')  # all indices
        idxs = np.empty((Ntarg, Npred), dtype='int')  # unique indices
        Nidxs = np.empty(Ntarg, dtype='int')  # number of unique indices

        # Select neighbors.
        for i in range(self.K):
            # Print progress.
            sys.stderr.write("\rFinding neighbors: {0}/{1}          "
                             .format(i + 1, self.K))
            sys.stdout.flush()

            # Monte Carlo data.
            X_train_t = rstate.normal(X_train, Xe_train).astype('float32')
            X_targ_t = rstate.normal(X_targ, Xe_targ).astype('float32')

            # Transform features.
            X_train_t, _ = feature_map(X_train_t, Xe_train)
            X_targ_t, _ = feature_map(X_targ_t, Xe_targ)

            # Construct KDTree.
            kdtree = KDTree(X_train_t, leafsize=self.leafsize)
            _, indices[i] = kdtree.query(X_targ_t, k=self.k,
                                         eps=self.eps, p=self.p,
                                         distance_upper_bound=self.dbound)

        # Select unique neighbors.
        sys.stderr.write('\n')
        for i in range(Ntarg):
            # Print progress.
            sys.stderr.write("\rSelecting unique neighbors: {0}/{1}          "
                             .format(i + 1, Ntarg))
            sys.stdout.flush()

            # Using `pandas.unique` over `np.unique` to avoid additional
            # overhead due to auto-sorting.
            midx_unique = unique(indices[:, i, :].flatten())
            Nidx = len(midx_unique)
            Nidxs[i] = Nidx
            idxs[i, :Nidx] = midx_unique
            idxs[i, Nidx:] = -99

        return idxs, Nidxs
