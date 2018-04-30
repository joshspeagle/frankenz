#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Object used to fit data and compute PDFs using a modified k-nearest neighbors
approach based on Monte Carlo methods.

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

__all__ = ["NearestNeighbors"]


class NearestNeighbors():
    """
    Fits data and generates predictions using a Bayesian-based nearest-neighbor
    approach.

    """

    def __init__(self, models, models_err, models_mask, leafsize=50, K=25,
                 feature_map='luptitude', fmap_args=None, fmap_kwargs=None,
                 rstate=None, verbose=True):
        """
        Load the model data into memory and initialize trees to facilitate
        nearest-neighbor searches.

        Parameters
        ----------
        models : `~numpy.ndarray` of shape (Nmodel, Nfilt)
            Model values.

        models_err : `~numpy.ndarray` of shape (Nmodel, Nfilt)
            Associated errors on the model values.

        models_mask : `~numpy.ndarray` of shape (Nmodel, Nfilt)
            Binary mask (0/1) indicating whether the model value was observed.

        leafsize : int, optional
            The number of points where the algorithm switches over
            to brute force. Default is `50`.

        K : int, optional
            The number of members used in the ensemble to incorporate
            errors using Monte Carlo methods. Default is `25`.

        feature_map : str or function, optional
            Function that transforms the input set of features/errors `X, Xe`
            to a new set of features/errors `Y, Ye` to facilitate nearest
            neighbor searches. Built-in options are `'identity'` (the identity
            function), `'magnitude'` (log10-based magnitudes),
            and `'luptitude'` (asinh-based magnitudes).
            Default is `'luptitude'`.

        fmap_args : args, optional
            Arguments to be passed to `feature_map`.

        fmap_kwargs : kwargs, optional
            Keyword arguments to be passed to `feature_map`.

        rstate : `~numpy.random.RandomState` instance, optional
            Random state instance. If not passed, the default `~numpy.random`
            instance will be used.

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

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

        self.leafsize = leafsize
        self.K = K
        self.KDTrees = None

        self.neighbors = None
        self.Nneighbors = None
        self.k = None
        self.eps = None
        self.p = None
        self.dbound = None

        # Initialize feature map.
        if fmap_args is None:
            fmap_args = []
        if fmap_kwargs is None:
            fmap_kwargs = dict()
        self.fmap_args = fmap_args
        self.fmap_kwargs = fmap_kwargs

        if feature_map == 'identity':
            # Identity function.
            def feature_map(x, xe, *args, **kwargs):
                return x, xe
        elif feature_map == 'magnitude':
            # Magnitude function.
            feature_map = magnitude
        elif feature_map == 'luptitude':
            # Asinh magnitude (Luptitude) function.
            feature_map = luptitude
        else:
            try:
                # Check if `feature_map` is a valid function.
                _ = feature_map(np.atleast_2d(X_train[0]),
                                np.atleast_2d(Xe_train[0]),
                                *fmap_args, **fmap_kwargs)
            except:
                # If all else fails, raise an exception.
                raise ValueError("The provided feature map is not valid.")
        self.feature_map = feature_map

        # Initialize RNG.
        if rstate is None:
            rstate = np.random

        # Build KDTrees.
        self.KDTrees = []
        for i, kdtree in enumerate(self._train_kdtrees(rstate=rstate)):
            if verbose:
                sys.stderr.write("\r{0}/{1} KDTrees constructed"
                                 .format(i+1, self.K))
                sys.stderr.flush()
            self.KDTrees.append(kdtree)
        if verbose:
            sys.stderr.write("\n")
            sys.stderr.flush()

    def _train_kdtrees(self, rstate=None):
        """
        Internal method used to train the `~scipy.spatial.KDTree` used
        for quick nearest-neighbor searches.

        Parameters
        ----------
        rstate : `~numpy.random.RandomState` instance, optional
            Random state instance. If not passed, the default `~numpy.random`
            instance will be used.

        """

        if rstate is None:
            rstate = np.random

        # Build KDTrees.
        for i in range(self.K):
            # Monte Carlo data.
            models_t = np.array(rstate.normal(self.models,
                                              self.models_err),
                                dtype='float32')
            # Transform data using feature map.
            Y_t, Ye_t = np.array(self.feature_map(models_t, self.models_err,
                                                  *self.fmap_args,
                                                  **self.fmap_kwargs),
                                 dtype='float32')
            # Construct KDTree.
            kdtree = KDTree(Y_t, leafsize=self.leafsize)

            yield kdtree

    def fit(self, data, data_err, data_mask, lprob_func=None, rstate=None,
            k=20, eps=1e-3, lp_norm=2, distance_upper_bound=np.inf,
            lprob_args=None, lprob_kwargs=None, track_scale=False,
            verbose=True):
        """
        Fit input models to the input data to compute the associated
        log-posteriors using the KMCkNN approximation.

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

        rstate : `~numpy.random.RandomState` instance, optional
            Random state instance. If not passed, the default `~numpy.random`
            instance will be used.

        k : int, optional
            The number of nearest neighbors selected by each member. Default is
            `20`.

        eps : float, optional
            If supplied, approximate (rather than exact) nearest neighbor
            queries are returned where the `k`th neighbor is guaranteed
            to be no further than `(1 + eps)` times the distance to the
            *real* `k`th nearest neighbor. Default is `1e-3`.

        lp_norm : float, optional
            The Minkowski p-norm that should be used to compute distances.
            Default is `2` (i.e. the Euclidean distance).

        distance_upper_bound : float, optional
            If supplied, return only neighbors within this distance.
            Default is `np.inf`.

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
        if rstate is None:
            rstate = np.random
        Ndata = len(data)
        self.k = k
        self.eps = eps
        self.lp_norm = lp_norm
        self.dbound = distance_upper_bound

        # Fit data.
        for i, blob in enumerate(self._fit(data, data_err, data_mask,
                                           lprob_func=lprob_func,
                                           rstate=rstate,
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

    def _fit(self, data, data_err, data_mask, lprob_func=None, rstate=None,
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

        rstate : `~numpy.random.RandomState` instance, optional
            Random state instance. If not passed, the default `~numpy.random`
            instance will be used.

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
        if rstate is None:
            rstate = np.random

        Ndata = len(data)
        Nmodels = self.K * self.k
        self.NDATA = Ndata

        if save_fits:
            inf = np.inf
            self.Nneighbors = np.zeros(Ndata, dtype='int')
            self.neighbors = np.zeros((Ndata, Nmodels), dtype='int') - 99
            self.fit_lnprior = np.zeros((Ndata, Nmodels), dtype='float') - inf
            self.fit_lnlike = np.zeros((Ndata, Nmodels), dtype='float') - inf
            self.fit_lnprob = np.zeros((Ndata, Nmodels), dtype='float') - inf
            self.fit_Ndim = np.zeros((Ndata, Nmodels), dtype='int')
            self.fit_chi2 = np.zeros((Ndata, Nmodels), dtype='float') + inf
            self.fit_scale = np.ones((Ndata, Nmodels), dtype='float')
            self.fit_scale_err = np.zeros((Ndata, Nmodels), dtype='float')

        # Fit data.
        for i, (x, xe, xm) in enumerate(zip(data, data_err, data_mask)):

            # Nearest-neighbor search.
            x_t = rstate.normal(x, xe)  # monte carlo data
            y_t, ye_t = self.feature_map(x_t, xe, *self.fmap_args,
                                         **self.fmap_kwargs)  # map to features
            y_t = np.atleast_2d(y_t)
            indices = np.array([T.query(y_t, k=self.k, eps=self.eps,
                                        p=self.lp_norm,
                                        distance_upper_bound=self.dbound)[1][0]
                                for T in self.KDTrees]).flatten()  # all idxs

            # Unique neighbor selection.
            idxs = unique(indices)
            Nidx = len(idxs)
            if save_fits:
                self.Nneighbors[i] = Nidx
                self.neighbors[i, :Nidx] = np.array(idxs)

            # Compute posteriors.
            results = lprob_func(x, xe, xm, self.models[idxs],
                                 self.models_err[idxs], self.models_mask[idxs],
                                 *lprob_args, **lprob_kwargs)
            if save_fits:
                self.fit_lnprior[i, :Nidx] = results[0]  # ln(prior)
                self.fit_lnlike[i, :Nidx] = results[1]  # ln(like)
                self.fit_lnprob[i, :Nidx] = results[2]  # ln(prob)
                self.fit_Ndim[i, :Nidx] = results[3]  # dimensionality of fit
                self.fit_chi2[i, :Nidx] = results[4]  # chi2
                if track_scale:
                    self.fit_scale[i, :Nidx] = results[5]  # scale-factor
                    self.fit_scale_err[i, :Nidx] = results[6]  # std(s)

            yield idxs, Nidx, results

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

        # Compute PDFs.
        for i, lwt in enumerate(logwt):
            Nidx = self.Nneighbors[i]  # number of models
            idxs = self.neighbors[i, :Nidx]  # model indices
            lwt_m = lwt[:Nidx]  # reduced set of weights
            lmap, levid = max(lwt_m), logsumexp(lwt_m)
            wt = np.exp(lwt_m - levid)
            if label_dict is not None:
                # Use dictionary if available.
                pdf = gauss_kde_dict(label_dict, y_idx=y_idx[idxs],
                                     y_std_idx=y_std_idx[idxs], y_wt=wt,
                                     *kde_args, **kde_kwargs)
            else:
                # Otherwise just use KDE.
                pdf = gauss_kde(model_labels[idxs], model_label_errs[idxs],
                                label_grid, y_wt=wt, *kde_args, **kde_kwargs)
            pdf /= pdf.sum()

            yield pdf, (lmap, levid)

    def fit_predict(self, data, data_err, data_mask, model_labels,
                    model_label_errs, lprob_func=None, rstate=None,
                    k=20, eps=1e-3, lp_norm=2, distance_upper_bound=np.inf,
                    label_dict=None, label_grid=None, kde_args=None,
                    kde_kwargs=None, lprob_args=None, lprob_kwargs=None,
                    return_gof=False, track_scale=False, verbose=True,
                    save_fits=True):
        """
        Fit input models to the input data to compute the associated
        log-posteriors and 1-D predictions using the KMCkNN approximation.

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

        rstate : `~numpy.random.RandomState` instance, optional
            Random state instance. If not passed, the default `~numpy.random`
            instance will be used.

        k : int, optional
            The number of nearest neighbors selected by each member. Default is
            `20`.

        eps : float, optional
            If supplied, approximate (rather than exact) nearest neighbor
            queries are returned where the `k`th neighbor is guaranteed
            to be no further than `(1 + eps)` times the distance to the
            *real* `k`th nearest neighbor. Default is `1e-3`.

        lp_norm : float, optional
            The Minkowski p-norm that should be used to compute distances.
            Default is `2` (i.e. the Euclidean distance).

        distance_upper_bound : float, optional
            If supplied, return only neighbors within this distance.
            Default is `np.inf`.

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
        if rstate is None:
            rstate = np.random
        self.k = k
        self.eps = eps
        self.lp_norm = lp_norm
        self.dbound = distance_upper_bound

        # Generate PDFs.
        for i, res in enumerate(self._fit_predict(data, data_err, data_mask,
                                                  model_labels,
                                                  model_label_errs,
                                                  lprob_func=lprob_func,
                                                  rstate=rstate,
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
                     model_label_errs, lprob_func=None, rstate=None,
                     label_dict=None, label_grid=None, kde_args=None,
                     kde_kwargs=None, lprob_args=None, lprob_kwargs=None,
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

        rstate : `~numpy.random.RandomState` instance, optional
            Random state instance. If not passed, the default `~numpy.random`
            instance will be used.

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
        if rstate is None:
            rstate = np.random
        Ndata = len(data)
        Nmodels = self.K * self.k
        if save_fits:
            self.Nneighbors = np.zeros(Ndata, dtype='int')
            self.neighbors = np.zeros((Ndata, Nmodels), dtype='int') - 99
            self.fit_lnprior = np.zeros((Ndata, Nmodels), dtype='float')-np.inf
            self.fit_lnlike = np.zeros((Ndata, Nmodels), dtype='float')-np.inf
            self.fit_lnprob = np.zeros((Ndata, Nmodels), dtype='float')-np.inf
            self.fit_Ndim = np.zeros((Ndata, Nmodels), dtype='int')
            self.fit_chi2 = np.zeros((Ndata, Nmodels), dtype='float') + np.inf
            self.fit_scale = np.ones((Ndata, Nmodels), dtype='float')
            self.fit_scale_err = np.zeros((Ndata, Nmodels), dtype='float')
            self.NDATA = Ndata
        if label_dict is not None:
            y_idx, y_std_idx = label_dict.fit(model_labels, model_label_errs)

        # Run generator.
        for i, (x, xe, xm) in enumerate(zip(data, data_err, data_mask)):

            # Nearest-neighbor search.
            x_t = rstate.normal(x, xe)  # monte carlo data
            y_t, ye_t = self.feature_map(x_t, xe, *self.fmap_args,
                                         **self.fmap_kwargs)  # map to features
            y_t = np.atleast_2d(y_t)
            indices = np.array([T.query(y_t, k=self.k, eps=self.eps,
                                        p=self.lp_norm,
                                        distance_upper_bound=self.dbound)[1][0]
                                for T in self.KDTrees]).flatten()  # all idxs

            # Unique neighbor selection.
            idxs = unique(indices)
            Nidx = len(idxs)
            if save_fits:
                self.Nneighbors[i] = Nidx
                self.neighbors[i, :Nidx] = np.array(idxs)

            # Compute posteriors.
            results = lprob_func(x, xe, xm, self.models[idxs],
                                 self.models_err[idxs], self.models_mask[idxs],
                                 *lprob_args, **lprob_kwargs)
            if save_fits:
                self.fit_lnprior[i, :Nidx] = results[0]  # ln(prior)
                self.fit_lnlike[i, :Nidx] = results[1]  # ln(like)
                self.fit_lnprob[i, :Nidx] = results[2]  # ln(prob)
                self.fit_Ndim[i, :Nidx] = results[3]  # dimensionality of fit
                self.fit_chi2[i, :Nidx] = results[4]  # chi2
                if track_scale:
                    self.fit_scale[i, :Nidx] = results[5]  # scale-factor
                    self.fit_scale_err[i, :Nidx] = results[6]  # std(s)
            lnprob = results[2]  # reduced set of posteriors

            # Compute PDF.
            lmap, levid = max(lnprob), logsumexp(lnprob)
            wt = np.exp(lnprob - levid)
            if label_dict is not None:
                pdf = gauss_kde_dict(label_dict, y_idx=y_idx[idxs],
                                     y_std_idx=y_std_idx[idxs], y_wt=wt,
                                     *kde_args, **kde_kwargs)
            else:
                pdf = gauss_kde(model_labels[idxs], model_label_errs[idxs],
                                label_grid, y_wt=wt,
                                *kde_args, **kde_kwargs)
            pdf /= pdf.sum()

            yield pdf, (lmap, levid)
