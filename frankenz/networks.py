#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Objects used to fit data and compute PDFs using a modified nearest-neighbor
approach based on adaptive networks.

"""

from __future__ import (print_function, division)
import six
from six.moves import range
from six import iteritems

import sys
import os
import warnings
import math
import numpy as np
import warnings
from scipy.spatial import KDTree
from pandas import unique
import networkx as nx

from .pdf import *

try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc import logsumexp

__all__ = ["SelfOrganizingMap", "GrowingNeuralGas", "_Network",
           "learn_linear", "learn_geometric", "learn_harmonic",
           "neighbor_gauss", "neighbor_lorentz", "lprob_train"]


def learn_linear(t, start=0.5, end=0.1, *args, **kwargs):
    """
    The linear learning rate between `start` and `end` at time `t`.

    """

    rate = (1. - t) * start + t * end

    return rate


def learn_geometric(t, start=0.5, end=0.1, *args, **kwargs):
    """
    The geometric learning rate between `start` and `end` at time `t`.

    """

    ln_rate = (1. - t) * np.log(start) + t * np.log(end)

    return np.exp(ln_rate)


def learn_harmonic(t, start=0.5, end=0.1, *args, **kwargs):
    """
    The weighted harmonic mean between `start` and `end` at time `t`.

    """

    inv_rate = (1. - t) / start + t / end

    return 1. / inv_rate


def neighbor_gauss(t, pos, positions, nside, start=0.7, end=0.02,
                   rate='harmonic', *args, **kwargs):
    """
    Compute distances between `pos` to `positions` using a Gaussian kernel
    with a standard deviation between `start * nside` to `end * nside`
    at time `t` based on the provided `rate` option.

    """

    if rate == 'linear':
        learn_func = learn_linear
    elif rate == 'geometric':
        learn_func = learn_geometric
    elif rate == 'harmonic':
        learn_func = learn_harmonic
    else:
        raise ValueError("Provided `rate` is not supported.")

    if nside is None:
        nside = np.sqrt(len(positions))
    sqdist = np.sum((pos - positions)**2, axis=1)
    sigma = learn_func(t, start=start, end=end) * nside

    return np.exp(-0.5 * sqdist / sigma**2), sigma


def neighbor_lorentz(t, pos, positions, nside, start=0.7, end=0.02,
                     rate='harmonic', *args, **kwargs):
    """
    Compute distances between `pos` to `positions` using a Lorentzian kernel
    with a standard deviation between `start * nside` to `end * nside`
    at time `t` based on the provided `rate` option.

    """

    if rate == 'linear':
        learn_func = learn_linear
    elif rate == 'geometric':
        learn_func = learn_geometric
    elif rate == 'harmonic':
        learn_func = learn_harmonic
    else:
        raise ValueError("Provided `rate` is not supported.")

    sqdist = np.sum((pos - positions)**2, axis=1)
    sigma = learn_func(t, start=start, end=end) * nside

    return sigma**2 / (sqdist + sigma**2), sigma


class _Network(object):
    """
    Fits data and generates predictions using a network of nodes (models)
    and the data used to train it.

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
        self.models_lmap = np.zeros(self.NMODEL) - np.inf
        self.models_levid = np.zeros(self.NMODEL) - np.inf

        self.fit_lnprior = None
        self.fit_lnlike = None
        self.fit_lnprob = None
        self.fit_Ndim = None
        self.fit_chi2 = None
        self.fit_scale = None
        self.fit_scale_err = None

        self.nodes = None
        self.nodes_pos = None
        self.nodes_idxs = None
        self.nodes_logwts = None
        self.nodes_scales = None
        self.nodes_scales_err = None
        self.nodes_Nmatch = None
        self.nodes_only = None
        self.NNODE, self.NPROJ = None, None

        self.neighbors = None
        self.Nneighbors = None

    def populate_network(self, lprob_func=None, discrete=False,
                         wt_thresh=1e-3, cdf_thresh=2e-4, lprob_args=None,
                         lprob_kwargs=None, track_scale=False, verbose=True):
        """
        Map input models onto the nodes of the network.

        Parameters
        ----------
        lprob_func : str or func, optional
            Log-posterior function to be used. Must return ln(prior), ln(like),
            ln(post), Ndim, chi2, and (optionally) scale and std(scale).
            If not provided, `~frankenz.pdf.loglike` will be used.

        discrete : bool, optional
            Whether to map objects back to the best-fitting node **only**,
            rather than assigning them to multiple nodes along with their
            relevant weights. Default is `False`.

        wt_thresh : float, optional
            The threshold `wt_thresh * max(y_wt)` used to ignore nodes
            with (relatively) negligible weights. Default is `1e-3`.

        cdf_thresh : float, optional
            The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
            nodes with (relatively) negligible weights. This option is only
            used when `wt_thresh=None`. Default is `2e-4`.

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
            def lprob_train(x, xe, xm, ys, yes, yms):
                results = loglike(x, xe, xm, ys, yes, yms)
                lnlike, ndim, chi2 = results
                return np.zeros_like(lnlike), lnlike, lnlike, ndim, chi2
            lprob_func = lprob_train
        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()
        if wt_thresh is None and cdf_thresh is None:
            wt_thresh = -np.inf  # default to no clipping/thresholding

        # Populate network with models.
        Nmodels = self.NMODEL
        percentage = -99
        populate = self._populate_network
        for i, results in enumerate(populate(lprob_func=lprob_func,
                                             discrete=discrete,
                                             wt_thresh=wt_thresh,
                                             cdf_thresh=cdf_thresh,
                                             lprob_args=lprob_args,
                                             lprob_kwargs=lprob_kwargs,
                                             track_scale=track_scale)):
            new_percentage = int((i+1) / Nmodels * 100)
            if verbose and new_percentage != percentage:
                percentage = new_percentage
                sys.stderr.write('\rMapping objects {:d}%'
                                 .format(percentage))
                sys.stderr.flush()
        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

    def _populate_network(self, lprob_func=None, discrete=False,
                          wt_thresh=1e-3, cdf_thresh=2e-4, lprob_args=None,
                          lprob_kwargs=None, track_scale=False):
        """
        Internal generator used by the network to map models onto nodes.

        Parameters
        ----------
        lprob_func : str or func, optional
            Log-posterior function to be used. Must return ln(prior), ln(like),
            ln(post), Ndim, chi2, and (optionally) scale and std(scale).
            If not provided, `~frankenz.pdf.loglike` will be used.

        discrete : bool, optional
            Whether to map objects back to the best-fitting node **only**,
            rather than assigning them to multiple nodes along with their
            relevant weights. Default is `False`.

        wt_thresh : float, optional
            The threshold `wt_thresh * max(y_wt)` used to ignore nodes
            with (relatively) negligible weights. Default is `1e-3`.

        cdf_thresh : float, optional
            The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
            nodes with (relatively) negligible weights. This option is only
            used when `wt_thresh=None`. Default is `2e-4`.

        lprob_args : args, optional
            Arguments to be passed to `lprob_func`.

        lprob_kwargs : kwargs, optional
            Keyword arguments to be passed to `lprob_func`.

        track_scale : bool, optional
            Whether `lprob_func` also returns the scale-factor. Default is
            `False`.

        """

        # Initialize values.
        if lprob_func is None:
            def lprob_train(x, xe, xm, ys, yes, yms):
                results = loglike(x, xe, xm, ys, yes, yms)
                lnlike, ndim, chi2 = results
                return np.zeros_like(lnlike), lnlike, lnlike, ndim, chi2
            lprob_func = lprob_train
        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()
        if wt_thresh is None and cdf_thresh is None:
            wt_thresh = -np.inf  # default to no clipping/thresholding
        self.lprob_func = lprob_func

        Nnodes, Nmodels = self.NNODE, self.NMODEL
        self.nodes_idxs = [[] for i in range(Nnodes)]
        self.nodes_logwts = [[] for i in range(Nnodes)]
        self.nodes_scales = [[] for i in range(Nnodes)]
        self.nodes_scales_err = [[] for i in range(Nnodes)]
        self.nodes_Nmatch = np.zeros(Nnodes, dtype='int')

        y = self.nodes
        ye = np.zeros_like(y)
        ym = np.ones_like(y, dtype='bool')

        # Map models to nodes.
        for i, (x, xe, xm) in enumerate(zip(self.models, self.models_err,
                                            self.models_mask)):

            # Fit network.
            node_results = lprob_func(x, xe, xm, y, ye, ym,
                                      *lprob_args, **lprob_kwargs)
            node_lnprob = node_results[2]

            # Find the set of node(s) the model maps to.
            if discrete:
                # If discrete, assign to the best-matching node.
                n_idxs = np.array([np.argmax(node_lnprob)])
            else:
                # Apply thresholding to get set of probabilistic associations.
                if wt_thresh is not None:
                    # Use relative amplitude to threshold.
                    lwt_min = np.log(wt_thresh) + np.max(node_lnprob)
                    n_idxs = np.arange(Nnodes)[node_lnprob > lwt_min]
                else:
                    # Use CDF to threshold.
                    idx_sort = np.argsort(node_lnprob)
                    node_prob = np.exp(node_lnprob - logsumexp(node_lnprob))
                    node_cdf = np.cumsum(node_prob[idx_sort])
                    n_idxs = idx_sort[node_cdf <= (1. - cdf_thresh)]
            # Compute normalized ln(weights).
            n_lnprobs = node_lnprob[n_idxs]
            n_lmap, n_levid = max(n_lnprobs), logsumexp(n_lnprobs)
            n_lnprobs -= n_levid
            self.models_lmap[i] = n_lmap
            self.models_levid[i] = n_levid
            # Compute scale-factors.
            if track_scale:
                n_scales = node_results[5][n_idxs]
                n_scales_err = node_results[6][n_idxs]
            else:
                n_scales = np.ones_like(n_idxs)
                n_scales_err = np.zeros_like(n_idxs)

            # Assign model to node(s).
            for j, lwt, s, serr in zip(n_idxs, n_lnprobs, n_scales,
                                       n_scales_err):
                self.nodes_idxs[j].append(i)
                self.nodes_logwts[j].append(lwt)
                self.nodes_scales[j].append(s)
                self.nodes_scales_err[j].append(serr)
                self.nodes_Nmatch[j] += 1

            yield n_idxs, n_lnprobs, n_scales, n_scales_err

    def get_node(self, idx=None, pos=None):
        """
        Returns quantities associated with the given node.

        Parameters
        ----------
        idx : int, optional
            The index of the node. Mutually exclusive with `pos`.

        pos : tuple, optional
            The position that will be used to search for the closest
            node. Mutually exclusive with `idx`.

        Returns
        -------
        idx : int
            The index of the node.

        node : `~numpy.ndarray` of shape (Ndim)
            The position of the node in data (observed) space.

        node_pos : `~numpy.ndarray` of shape (Nproj)
            The position of the node in the projected (manifold) space.

        node_idxs : `~numpy.ndarray` of shape (Nmatch)
            The indices of the models assigned to the node.

        node_logwts : `~numpy.ndarray` of shape (Nmatch)
            The associated ln(weights) of the models assigned to the node.

        """

        if idx is None and pos is None:
            raise ValueError("Either `idx` or `pos` must be specified.")
        elif idx is not None and pos is not None:
            raise ValueError("Both `idx` and `pos` cannot be specified.")
        elif pos is not None:
            idx = np.argmin([(pos - p)**2 for p in self.nodes_pos])

        return (idx, self.nodes[idx], self.nodes_pos[idx],
                self.nodes_idxs[idx], self.nodes_logwts[idx])

    def get_pdf(self, idx, model_labels, model_label_errs,
                label_dict=None, label_grid=None, kde_args=None,
                kde_kwargs=None, return_gof=False):
        """
        Returns PDF associated with the given node.

        Parameters
        ----------
        idx : int, optional
            The index of the node.

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

        return_gof : bool, optional
            Whether to return a tuple containing the ln(MAP) and
            ln(evidence) values for the predictions
            along with the pdfs. Default is `False`.

        Returns
        -------
        pdf : `~numpy.ndarray` of shape (Ngrid)
            1-D PDF.

        (lmap, levid) : 2-tuple of floats, optional
            ln(MAP) and ln(evidence) values.

        """

        # Initialize values.
        if kde_args is None:
            kde_args = []
        if kde_kwargs is None:
            kde_kwargs = dict()
        if label_dict is None and label_grid is None:
            raise ValueError("`label_dict` or `label_grid` must be specified.")

        # Compute PDFs.
        Nidx = self.nodes_Nmatch[idx]  # number of models
        idxs = self.nodes_idxs[idx]  # model indices
        lwt = self.nodes_logwts[idx]  # model ln(wts)
        if Nidx > 0:
            lmap, levid = max(lwt), logsumexp(lwt)  # model GOF metrics
            wt = np.exp(lwt - levid)
            if label_dict is not None:
                # Use dictionary if available.
                y_idx, y_std_idx = label_dict.fit(model_labels,
                                                  model_label_errs)
                pdf = gauss_kde_dict(label_dict, y_idx=y_idx[idxs],
                                     y_std_idx=y_std_idx[idxs], y_wt=wt,
                                     *kde_args, **kde_kwargs)
            else:
                # Otherwise just use KDE.
                pdf = gauss_kde(model_labels[idxs], model_label_errs[idxs],
                                label_grid, y_wt=wt, *kde_args, **kde_kwargs)
        else:
            lmap, levid = -np.inf, -np.inf
            if label_dict is not None:
                pdf = np.zeros_like(label_dict.grid)
            else:
                pdf = np.zeros_like(label_grid)

        if return_gof:
            return pdf, (lmap, levid)
        else:
            return pdf

    def get_pdfs(self, model_labels, model_label_errs, label_dict=None,
                 label_grid=None, kde_args=None, kde_kwargs=None,
                 return_gof=False, verbose=True):
        """
        Compute photometric 1-D predictions to the target distribution
        using the models (and possibly associated weights) to each node in
        the network.

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
        if label_dict is None and label_grid is None:
            raise ValueError("`label_dict` or `label_grid` must be specified.")
        if self.nodes_idxs is None:
            raise ValueError("Network has not been trained!")
        if label_dict is not None:
            Nx = label_dict.Ngrid
        else:
            Nx = len(label_grid)
        Nnodes = self.NNODE
        pdfs = np.zeros((Nnodes, Nx))
        if return_gof:
            lmap = np.zeros(Nnodes)
            levid = np.zeros(Nnodes)

        # Compute PDFs.
        for i, res in enumerate(self._get_pdfs(model_labels, model_label_errs,
                                               label_dict=label_dict,
                                               label_grid=label_grid,
                                               kde_args=kde_args,
                                               kde_kwargs=kde_kwargs)):
            pdf, gof = res
            pdfs[i] = pdf
            if return_gof:
                lmap[i], levid[i] = gof  # save gof metrics
            if verbose:
                sys.stderr.write('\rGenerating node PDF {0}/{1}'
                                 .format(i+1, Nnodes))
                sys.stderr.flush()
        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

        if return_gof:
            return pdfs, (lmap, levid)
        else:
            return pdfs

    def _get_pdfs(self, model_labels, model_label_errs, label_dict=None,
                  label_grid=None, kde_args=None, kde_kwargs=None):
        """
        Internal generator used to compute photometric 1-D predictions
        over nodes of the network.

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
        if label_dict is None and label_grid is None:
            raise ValueError("`label_dict` or `label_grid` must be specified.")
        Nnodes = self.NNODE
        if label_dict is not None:
            y_idx, y_std_idx = label_dict.fit(model_labels, model_label_errs)

        # Compute PDFs.
        for i in range(Nnodes):
            Nidx = self.nodes_Nmatch[i]  # number of models
            idxs = self.nodes_idxs[i]  # model indices
            lwt = self.nodes_logwts[i]  # model ln(wts)
            if Nidx > 0:
                lmap, levid = max(lwt), logsumexp(lwt)
                wt = np.exp(lwt - levid)
                if label_dict is not None:
                    # Use dictionary if available.
                    pdf = gauss_kde_dict(label_dict, y_idx=y_idx[idxs],
                                         y_std_idx=y_std_idx[idxs], y_wt=wt,
                                         *kde_args, **kde_kwargs)
                else:
                    # Otherwise just use KDE.
                    pdf = gauss_kde(model_labels[idxs], model_label_errs[idxs],
                                    label_grid, y_wt=wt, *kde_args,
                                    **kde_kwargs)
            else:
                lmap, levid = -np.inf, -np.inf
                if label_dict is not None:
                    pdf = np.zeros_like(label_dict.grid)
                else:
                    pdf = np.zeros_like(label_grid)

            yield pdf, (lmap, levid)

    def fit(self, data, data_err, data_mask, lprob_func=None, nodes_only=False,
            wt_thresh=1e-3, cdf_thresh=2e-4, lprob_args=None,
            lprob_kwargs=None, track_scale=False, verbose=True):
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
            If not provided, `~frankenz.pdf.loglike` will be used.

        nodes_only : bool, optional
            Whether to only fit the nodes of the network, ignoring the
            underlying data. Default is `False`.

        wt_thresh : float, optional
            The threshold `wt_thresh * max(y_wt)` used to ignore nodes
            with (relatively) negligible weights. Default is `1e-3`.

        cdf_thresh : float, optional
            The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
            nodes with (relatively) negligible weights. This option is only
            used when `wt_thresh=None`. Default is `2e-4`.

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
            lprob_func = self.lprob_func
        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()
        if wt_thresh is None and cdf_thresh is None:
            wt_thresh = -np.inf  # default to no clipping/thresholding
        Ndata = len(data)

        # Fit data.
        for i, results in enumerate(self._fit(data, data_err, data_mask,
                                              lprob_func=lprob_func,
                                              nodes_only=nodes_only,
                                              wt_thresh=wt_thresh,
                                              cdf_thresh=cdf_thresh,
                                              lprob_args=lprob_args,
                                              lprob_kwargs=lprob_kwargs,
                                              track_scale=track_scale)):
            if verbose:
                sys.stderr.write('\rFitting object {0}/{1}'.format(i+1, Ndata))
                sys.stderr.flush()
        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

    def _fit(self, data, data_err, data_mask, lprob_func=None,
             nodes_only=False, wt_thresh=1e-3, cdf_thresh=2e-4,
             lprob_args=None, lprob_kwargs=None, track_scale=False):
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
            If not provided, `~frankenz.pdf.loglike` will be used.

        nodes_only : bool, optional
            Whether to only fit the nodes of the network, ignoring the
            underlying data. Default is `False`.

        wt_thresh : float, optional
            The threshold `wt_thresh * max(y_wt)` used to ignore nodes
            with (relatively) negligible weights. Default is `1e-3`.

        cdf_thresh : float, optional
            The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
            nodes with (relatively) negligible weights. This option is only
            used when `wt_thresh=None`. Default is `2e-4`.

        lprob_args : args, optional
            Arguments to be passed to `lprob_func`.

        lprob_kwargs : kwargs, optional
            Keyword arguments to be passed to `lprob_func`.

        track_scale : bool, optional
            Whether `lprob_func` also returns the scale-factor. Default is
            `False`.

        Returns
        -------
        results : tuple
            Output of `lprob_func` yielded from the generator.

        """

        # Initialize values.
        if lprob_func is None:
            lprob_func = self.lprob_func
        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()
        if wt_thresh is None and cdf_thresh is None:
            wt_thresh = -np.inf  # default to no clipping/thresholding

        Ndata = len(data)
        Nnodes, Nmodels = self.NNODE, self.NMODEL
        self.NDATA = Ndata
        self.Nneighbors = np.zeros(Ndata, dtype='int')
        self.neighbors = []
        self.fit_lnprior = []
        self.fit_lnlike = []
        self.fit_lnprob = []
        self.fit_Ndim = []
        self.fit_chi2 = []
        self.fit_scale = []
        self.fit_scale_err = []

        match_sel = np.arange(Nnodes)[self.nodes_Nmatch > 0]
        y = self.nodes[match_sel]
        ye = np.zeros_like(y)
        ym = np.ones_like(y, dtype='bool')

        self.nodes_only = nodes_only

        # Fit data.
        for i, (x, xe, xm) in enumerate(zip(data, data_err, data_mask)):

            # Fit network.
            node_results = lprob_func(x, xe, xm, y, ye, ym,
                                      *lprob_args, **lprob_kwargs)
            node_lnprob = node_results[2]

            # Apply thresholding.
            if wt_thresh is not None:
                # Use relative amplitude to threshold.
                lwt_min = np.log(wt_thresh) + np.max(node_lnprob)
                wsel = node_lnprob > lwt_min
            else:
                # Use CDF to threshold.
                idx_sort = np.argsort(node_lnprob)
                node_prob = np.exp(node_lnprob - logsumexp(node_lnprob))
                node_cdf = np.cumsum(node_prob[idx_sort])
                wsel = idx_sort[node_cdf <= (1. - cdf_thresh)]
            sel_arr = match_sel[wsel]

            if nodes_only:
                # Take our nodes to be our models.
                self.Nneighbors[i] = len(sel_arr)
                self.neighbors.append(sel_arr)
                results = [nr[wsel] for nr in node_results]
            else:
                # Unique neighbor selection based on network fits.
                indices = np.array([idx for sidx in sel_arr
                                    for idx in self.nodes_idxs[sidx]])
                idxs = unique(indices)
                Nidx = len(idxs)
                self.Nneighbors[i] = Nidx
                self.neighbors.append(np.array(idxs))

                # Compute posteriors.
                results = lprob_func(x, xe, xm, self.models[idxs],
                                     self.models_err[idxs],
                                     self.models_mask[idxs],
                                     *lprob_args, **lprob_kwargs)
            self.fit_lnprior.append(results[0])  # ln(prior)
            self.fit_lnlike.append(results[1])  # ln(like)
            self.fit_lnprob.append(results[2])  # ln(prob)
            self.fit_Ndim.append(results[3])  # dimensionality of fit
            self.fit_chi2.append(results[4])  # chi2
            if track_scale:
                self.fit_scale.append(results[5])  # scale-factor
                self.fit_scale_err.append(results[6])  # std(s)

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

        logwt : list of arrays matching saved results, optional
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

        # If the fits were only done to each node, compute the (effective)
        # PDF at each node.
        if self.nodes_only:
            node_pdfs = self.get_pdfs(model_labels, model_label_errs,
                                      label_dict=label_dict,
                                      label_grid=label_grid,
                                      kde_args=kde_args, kde_kwargs=kde_kwargs,
                                      return_gof=False, verbose=verbose)
        else:
            node_pdfs = None

        # Compute PDFs.
        for i, res in enumerate(self._predict(model_labels, model_label_errs,
                                              node_pdfs=node_pdfs,
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

    def _predict(self, model_labels, model_label_errs, node_pdfs=None,
                 label_dict=None, label_grid=None, logwt=None, kde_args=None,
                 kde_kwargs=None):
        """
        Internal generator used to compute photometric 1-D predictions.

        Parameters
        ----------
        model_labels : `~numpy.ndarray` of shape (Nmodel)
            Model values.

        model_label_errs : `~numpy.ndarray` of shape (Nmodel)
            Associated errors on the data values.

        node_pdfs : `~numpy.ndarray` of shape (Nnodes, Ngrid), optional
            The effective PDFs at each node. If `nodes_only=True` when
            computing the fits, these will be used to compute the relevant
            PDFs. Otherwise, these will be ignored.

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

        # Initialize values.
        if kde_args is None:
            kde_args = []
        if kde_kwargs is None:
            kde_kwargs = dict()
        if logwt is None:
            logwt = self.fit_lnprob
        if label_dict is None and label_grid is None:
            raise ValueError("`label_dict` or `label_grid` must be specified.")
        if self.nodes_only and node_pdfs is None:
            raise ValueError("Fits were only computed to nodes in the network "
                             "but the relevant `node_pdfs` are not provided.")
        if label_dict is not None:
            y_idx, y_std_idx = label_dict.fit(model_labels, model_label_errs)

        # Compute PDFs.
        for i, lwt in enumerate(logwt):
            Nidx = self.Nneighbors[i]  # number of models
            idxs = self.neighbors[i]  # model indices
            lmap, levid = max(lwt), logsumexp(lwt)
            wt = np.exp(lwt - levid)
            if node_pdfs is not None:
                # Stack node PDFs based on their relative weights.
                pdf = np.dot(wt, node_pdfs[idxs, :])
            elif label_dict is not None:
                # Otherwise, use a dictionary if available to compute the
                # PDF from the model fits.
                pdf = gauss_kde_dict(label_dict, y_idx=y_idx[idxs],
                                     y_std_idx=y_std_idx[idxs], y_wt=wt,
                                     *kde_args, **kde_kwargs)
            else:
                # Otherwise, just use KDE to compute the PDF from model fits.
                pdf = gauss_kde(model_labels[idxs], model_label_errs[idxs],
                                label_grid, y_wt=wt, *kde_args, **kde_kwargs)

            yield pdf, (lmap, levid)

    def fit_predict(self, data, data_err, data_mask, model_labels,
                    model_label_errs, lprob_func=None, nodes_only=False,
                    wt_thresh=1e-3, cdf_thresh=2e-4,
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
            If not provided, `~frankenz.pdf.loglike` will be used.

        nodes_only : bool, optional
            Whether to only fit the nodes of the network, ignoring the
            underlying data. Default is `False`.

        wt_thresh : float, optional
            The threshold `wt_thresh * max(y_wt)` used to ignore nodes
            with (relatively) negligible weights. Default is `1e-3`.

        cdf_thresh : float, optional
            The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
            nodes with (relatively) negligible weights. This option is only
            used when `wt_thresh=None`. Default is `2e-4`.

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

        """

        # Initialize values.
        if lprob_func is None:
            lprob_func = self.lprob_func
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
        if wt_thresh is None and cdf_thresh is None:
            wt_thresh = -np.inf  # default to no clipping/thresholding
        Ndata = len(data)
        pdfs = np.zeros((Ndata, Nx))
        if return_gof:
            lmap = np.zeros(Ndata)
            levid = np.zeros(Ndata)
        if nodes_only:
            self.nodes_only = True
            node_pdfs = self.get_pdfs(model_labels, model_label_errs,
                                      label_dict=label_dict,
                                      label_grid=label_grid,
                                      kde_args=kde_args, kde_kwargs=kde_kwargs,
                                      return_gof=False, verbose=verbose)
        else:
            self.nodes_only = False
            node_pdfs = None

        # Generate PDFs.
        for i, res in enumerate(self._fit_predict(data, data_err, data_mask,
                                                  model_labels,
                                                  model_label_errs,
                                                  lprob_func=lprob_func,
                                                  node_pdfs=node_pdfs,
                                                  wt_thresh=wt_thresh,
                                                  cdf_thresh=cdf_thresh,
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
                     model_label_errs, lprob_func=None, node_pdfs=None,
                     wt_thresh=1e-3, cdf_thresh=2e-4,
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
            If not provided, `~frankenz.pdf.loglike` will be used.

        node_pdfs : `~numpy.ndarray` of shape (Nnodes, Ngrid), optional
            The effective PDFs at each node. If `nodes_only=True` when
            computing the fits, these will be used to compute the relevant
            PDFs. Otherwise, these will be ignored.

        wt_thresh : float, optional
            The threshold `wt_thresh * max(y_wt)` used to ignore nodes
            with (relatively) negligible weights. Default is `1e-3`.

        cdf_thresh : float, optional
            The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
            nodes with (relatively) negligible weights. This option is only
            used when `wt_thresh=None`. Default is `2e-4`.

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

        """

        # Initialize values.
        if lprob_func is None:
            lprob_func = self.lprob_func
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
            y_idx, y_std_idx = label_dict.fit(model_labels, model_label_errs)
        if wt_thresh is None and cdf_thresh is None:
            wt_thresh = -np.inf  # default to no clipping/thresholding

        Ndata = len(data)
        Nnodes, Nmodels = self.NNODE, self.NMODEL
        match_sel = np.arange(Nnodes)[self.nodes_Nmatch > 0]
        y = self.nodes[match_sel]
        ye = np.zeros_like(y)
        ym = np.ones_like(y, dtype='bool')

        if save_fits:
            self.NDATA = Ndata
            self.Nneighbors = np.zeros(Ndata, dtype='int')
            self.neighbors = []
            self.fit_lnprior = []
            self.fit_lnlike = []
            self.fit_lnprob = []
            self.fit_Ndim = []
            self.fit_chi2 = []
            self.fit_scale = []
            self.fit_scale_err = []

        # Run generator.
        for i, (x, xe, xm) in enumerate(zip(data, data_err, data_mask)):

            # Fit network.
            node_results = lprob_func(x, xe, xm, y, ye, ym,
                                      *lprob_args, **lprob_kwargs)
            node_lnprob = node_results[2]

            # Apply thresholding.
            if wt_thresh is not None:
                # Use relative amplitude to threshold.
                lwt_min = np.log(wt_thresh) + np.max(node_lnprob)
                wsel = node_lnprob > lwt_min
            else:
                # Use CDF to threshold.
                idx_sort = np.argsort(node_lnprob)
                node_prob = np.exp(node_lnprob - logsumexp(node_lnprob))
                node_cdf = np.cumsum(node_prob[idx_sort])
                wsel = idx_sort[node_cdf <= (1. - cdf_thresh)]
            sel_arr = match_sel[wsel]

            if node_pdfs is not None:
                # Take our nodes to be our models.
                idxs = sel_arr
                results = [nr[wsel] for nr in node_results]
                if save_fits:
                    self.Nneighbors[i] = len(idxs)
                    self.neighbors.append(np.array(idxs))
            else:
                # Unique neighbor selection based on network fits.
                indices = np.array([idx for sidx in sel_arr
                                    for idx in self.nodes_idxs[sidx]])
                idxs = unique(indices)
                Nidx = len(idxs)
                if save_fits:
                    self.Nneighbors[i] = Nidx
                    self.neighbors.append(np.array(idxs))

                # Compute posteriors.
                results = lprob_func(x, xe, xm, self.models[idxs],
                                     self.models_err[idxs],
                                     self.models_mask[idxs],
                                     *lprob_args, **lprob_kwargs)
            if save_fits:
                self.fit_lnprior.append(results[0])  # ln(prior)
                self.fit_lnlike.append(results[1])  # ln(like)
                self.fit_lnprob.append(results[2])  # ln(prob)
                self.fit_Ndim.append(results[3])  # dimensionality of fit
                self.fit_chi2.append(results[4])  # chi2
                if track_scale:
                    self.fit_scale.append(results[5])  # scale-factor
                    self.fit_scale_err.append(results[6])  # std(s)

            # Compute PDF.
            lnprob = results[2]  # reduced set of posteriors
            lmap, levid = max(lnprob), logsumexp(lnprob)
            wt = np.exp(lnprob - levid)
            if node_pdfs is not None:
                # Stack node PDFs based on their relative weights.
                pdf = np.dot(wt, node_pdfs[idxs, :])
            elif label_dict is not None:
                # Otherwise, use a dictionary if available to compute the
                # PDF from the model fits.
                pdf = gauss_kde_dict(label_dict, y_idx=y_idx[idxs],
                                     y_std_idx=y_std_idx[idxs], y_wt=wt,
                                     *kde_args, **kde_kwargs)
            else:
                # Otherwise, just use KDE to compute the PDF from model fits.
                pdf = gauss_kde(model_labels[idxs], model_label_errs[idxs],
                                label_grid, y_wt=wt, *kde_args, **kde_kwargs)

            yield pdf, (lmap, levid)


class SelfOrganizingMap(_Network):
    """
    Fits data and generates predictions using a Self-Organizing Map (SOM).

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
        super(SelfOrganizingMap, self).__init__(models, models_err,
                                                models_mask)  # _Network

    def train_network(self, models=None, models_err=None, models_mask=None,
                      nside=50, nproj=2, nodes_init=None, niter=2000,
                      nbatch=50, err_kernel=None, lprob_func=None,
                      learn_func=None, neighbor_func=None,
                      wt_thresh=1e-3, cdf_thresh=2e-4, rstate=None,
                      lprob_args=None, lprob_kwargs=None, track_scale=False,
                      learn_args=None, learn_kwargs=None, neighbor_args=None,
                      neighbor_kwargs=None, verbose=True):
        """
        Train the SOM using the provided set of models.

        Parameters
        ----------
        models : `~numpy.ndarray` of shape (Nmodel, Nfilt), optional
            Model values.

        models_err : `~numpy.ndarray` of shape (Nmodel, Nfilt), optional
            Associated errors on the model values.

        models_mask : `~numpy.ndarray` of shape (Nmodel, Nfilt), optional
            Binary mask (0/1) indicating whether the model value was observed.

        nside : int, optional
            The number of nodes used to specify each side of the SOM.
            Default is `50`.

        nproj : int, optional
            The number of projected dimensions used to specify the positions
            of the nodes of the network. Default is `2`.

        nodes_init : `~numpy.ndarray` of shape (nside**nproj, Nfilt)
            A set of initial values used to initialize each of the nodes.
            If not provided, nodes will be initialized randomly from the data.

        niter : int, optional
            The number of iterations to train the SOM. Default is `2000`.

        nbatch : int, optional
            The number of objects used in a given iteration. Default is `50`.

        err_kernel : `~numpy.ndarray` of shape (Nmodel, Nfilt), optional
            An error kernel added in quadrature to the provided
            `models_err` used when training the SOM.

        lprob_func : str or func, optional
            Log-posterior function to be used when computing fits between
            the network and the models **in the mapped feature space**
            (i.e. via the provided `feature_maps`). Must return ln(prior),
            ln(like), ln(post), Ndim, chi2, and (optionally) scale and
            scale_err. If not provided, `~frankenz.pdf.loglike` will be used.

        learn_func : func, optional
            A function that returns the learning rate as a function of
            fractional iteration (i.e. from `[0., 1.]`). By default,
            the geometric learning rate function `learn_harmonic` is used.

        neighbor_func : func, optional
            A function that returns the weights for nodes in the neighborhood
            of the best-matching node. By default, the Gaussian neighborhood
            function `neighbor_gauss` is used.

        wt_thresh : float, optional
            The threshold `wt_thresh * max(y_wt)` used to ignore nodes
            with (relatively) negligible weights. Default is `1e-3`.

        cdf_thresh : float, optional
            The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
            nodes with (relatively) negligible weights. This option is only
            used when `wt_thresh=None`. Default is `2e-4`.

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

        learn_args : args, optional
            Arguments to be passed to `learn_func`.

        learn_kwargs : kwargs, optional
            Keyword arguments to be passed to `learn_func`.

        neighbor_args : args, optional
            Arguments to be passed to `neighbor_func`.

        neighbor_kwargs : kwargs, optional
            Keyword arguments to be passed to `neighbor_func`.

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

        """

        # Initialize values.
        if lprob_func is None:
            def lprob_train(x, xe, xm, ys, yes, yms):
                results = loglike(x, xe, xm, ys, yes, yms)
                lnlike, ndim, chi2 = results
                return np.zeros_like(lnlike), lnlike, lnlike, ndim, chi2
            lprob_func = lprob_train
        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()
        if learn_func is None:
            learn_func = learn_harmonic
        if learn_args is None:
            learn_args = []
        if learn_kwargs is None:
            learn_kwargs = dict()
        if neighbor_func is None:
            neighbor_func = neighbor_gauss
        if neighbor_args is None:
            neighbor_args = []
        if neighbor_kwargs is None:
            neighbor_kwargs = dict()
        if rstate is None:
            rstate = np.random

        # Load in models.
        if models is None:
            models = self.models
        if models_mask is None:
            models_mask = self.models_mask
        if models_err is None:
            models_err = self.models_err
        if err_kernel is not None:
            models_err = np.sqrt(models_err**2 + err_kernel**2)

        # Train the SOM.
        train = self._train_network
        for i, res in enumerate(train(models, models_err, models_mask,
                                      lprob_func=lprob_func,
                                      nside=nside, nproj=nproj,
                                      nodes_init=nodes_init,
                                      learn_func=learn_func,
                                      neighbor_func=neighbor_func,
                                      niter=niter, nbatch=nbatch,
                                      wt_thresh=wt_thresh,
                                      cdf_thresh=cdf_thresh,
                                      rstate=rstate,
                                      lprob_args=lprob_args,
                                      lprob_kwargs=lprob_kwargs,
                                      track_scale=track_scale,
                                      learn_args=learn_args,
                                      learn_kwargs=learn_kwargs,
                                      neighbor_args=neighbor_args,
                                      neighbor_kwargs=neighbor_kwargs)):
            fits, bmu, learn_rate, learn_sigma = res
            if i % nbatch == 0 and verbose:
                sys.stderr.write('\rIteration {:d}/{:d} '
                                 '[learn={:6.3f}, sigma={:6.3f}]     '
                                 .format(int(i/nbatch) + 1, niter,
                                         learn_rate, learn_sigma))
                sys.stderr.flush()
        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

    def _train_network(self, models, models_err, models_mask, lprob_func=None,
                       nside=50, nproj=2, nodes_init=None, learn_func=None,
                       neighbor_func=None, niter=2000, nbatch=50,
                       wt_thresh=1e-3, cdf_thresh=2e-4,
                       rstate=None, lprob_args=None, lprob_kwargs=None,
                       track_scale=False, learn_args=None, learn_kwargs=None,
                       neighbor_args=None, neighbor_kwargs=None):
        """
        Internal method used to train the SOM.

        Parameters
        ----------
        models : `~numpy.ndarray` of shape (Nmodel, Nfilt)
            Model values.

        models_err : `~numpy.ndarray` of shape (Nmodel, Nfilt)
            Associated errors on the model values.

        models_mask : `~numpy.ndarray` of shape (Nmodel, Nfilt)
            Binary mask (0/1) indicating whether the model value was observed.

        lprob_func : str or func, optional
            Log-posterior function to be used when computing fits between
            the network and the models **in the mapped feature space**
            (i.e. via the provided `feature_maps`). Must return ln(prior),
            ln(like), ln(post), Ndim, chi2, and (optionally) scale and
            scale_err. If not provided, `~frankenz.pdf.loglike` will be used.

        nside : int, optional
            The number of nodes used to specify each side of the SOM.
            Default is `50`.

        nproj : int, optional
            The number of projected dimensions used to specify the positions
            of the nodes of the network. Default is `2`.

        nodes_init : `~numpy.ndarray` of shape (nside**nproj, Nfilt)
            A set of initial values used to initialize each of the nodes.
            If not provided, nodes will be initialized randomly from the data.

        learn_func : func, optional
            A function that returns the learning rate as a function of
            fractional iteration (i.e. from `[0., 1.]`). By default,
            the geometric learning rate function `learn_harmonic` is used.

        neighbor_func : func, optional
            A function that returns the weights for nodes in the neighborhood
            of the best-matching node. By default, the Gaussian neighborhood
            function `neighbor_gauss` is used.

        niter : int, optional
            The number of iterations to train the SOM. Default is `2000`.

        nbatch : int, optional
            The number of objects used in a given iteration. Default is `50`.

        wt_thresh : float, optional
            The threshold `wt_thresh * max(y_wt)` used to ignore nodes
            with (relatively) negligible weights. Default is `1e-3`.

        cdf_thresh : float, optional
            The `1 - cdf_thresh` threshold of the (sorted) CDF used to ignore
            nodes with (relatively) negligible weights. This option is only
            used when `wt_thresh=None`. Default is `2e-4`.

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

        learn_args : args, optional
            Arguments to be passed to `learn_func`.

        learn_kwargs : kwargs, optional
            Keyword arguments to be passed to `learn_func`.

        neighbor_args : args, optional
            Arguments to be passed to `neighbor_func`.

        neighbor_kwargs : kwargs, optional
            Keyword arguments to be passed to `neighbor_func`.

        """

        # Initialize values.
        self.NITER, self.NBATCH = niter, nbatch
        times = np.linspace(0., 1., niter * nbatch)
        if lprob_func is None:
            def lprob_train(x, xe, xm, ys, yes, yms):
                results = loglike(x, xe, xm, ys, yes, yms)
                lnlike, ndim, chi2 = results
                return np.zeros_like(lnlike), lnlike, lnlike, ndim, chi2
            lprob_func = lprob_train
        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()
        if learn_func is None:
            learn_func = learn_harmonic
        if learn_args is None:
            learn_args = []
        if learn_kwargs is None:
            learn_kwargs = dict()
        if neighbor_func is None:
            neighbor_func = neighbor_gauss
        if neighbor_args is None:
            neighbor_args = []
        if neighbor_kwargs is None:
            neighbor_kwargs = dict()
        if rstate is None:
            rstate = np.random
        if wt_thresh is None and cdf_thresh is None:
            wt_thresh = -np.inf  # default to no clipping/thresholding

        # Initialize SOM node positions.
        self.NSIDE, self.NNODE, self.NPROJ = nside, nside**nproj, nproj
        self.nodes_pos = np.zeros((self.NNODE, self.NPROJ))
        for i in range(self.NPROJ):
            counter = int(self.NNODE / self.NSIDE**(i+1))
            n = int(self.NNODE / counter)
            for k, j in enumerate(range(n)):
                self.nodes_pos[j*counter:(j+1)*counter, i] = k % self.NSIDE

        # Initialize SOM node models.
        Nmodel = len(models)
        self.nodes = np.zeros((self.NNODE, self.NDIM))
        if nodes_init is None:
            idxs = rstate.choice(Nmodel, size=self.NNODE, replace=False)
            self.nodes = np.array(models[idxs])
        else:
            self.nodes = nodes_init

        y = self.nodes
        ye = np.zeros_like(y)
        ym = np.ones_like(y, dtype='bool')

        # Train the network.
        for i, t in enumerate(times):

            # Draw object.
            idx = rstate.choice(Nmodel)
            x, xe, xm = models[idx], models_err[idx], models_mask[idx]

            # Fit network.
            node_results = lprob_func(x, xe, xm, y, ye, ym,
                                      *lprob_args, **lprob_kwargs)
            node_lnprob = node_results[2]

            # Rescale models (if needed).
            if track_scale:
                node_scales = node_results[5]
                self.nodes *= node_scales[:, None]  # re-scale node models

            # Find the "best-matching unit".
            bmu = np.argmax(node_lnprob)

            # Compute learning parameters.
            learn_rate = learn_func(t, *learn_args, **learn_kwargs)
            learn_wt, learn_sigma = neighbor_func(t, self.nodes_pos[bmu],
                                                  self.nodes_pos, self.NSIDE,
                                                  *neighbor_args,
                                                  **neighbor_kwargs)

            # Use relative amplitude to threshold.
            if wt_thresh is not None:
                wt_min = wt_thresh * np.max(learn_wt)
                n_idxs = np.arange(self.NNODE)[learn_wt > wt_min]
            else:
                # Use CDF to threshold.
                idx_sort = np.argsort(learn_wt)
                node_prob = learn_wt / np.sum(learn_wt)
                node_cdf = np.cumsum(node_prob[idx_sort])
                n_idxs = idx_sort[node_cdf <= (1. - cdf_thresh)]

            # Update SOM.
            resid = models[idx] - y[n_idxs]
            self.nodes[n_idxs] += learn_rate * learn_wt[n_idxs, None] * resid

            yield node_results, bmu, learn_rate, learn_sigma


class GrowingNeuralGas(_Network):
    """
    Fits data and generates predictions using a Growing Neural Gas (GNG).

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
        super(GrowingNeuralGas, self).__init__(models, models_err,
                                               models_mask)  # _Network

        self.graph = nx.Graph()

    def train_network(self, models=None, models_err=None, models_mask=None,
                      learn_best=0.2, learn_neighbor=0.005, max_age=15,
                      nbatch=50, new_err_dec=0.5, all_err_dec=5e-3,
                      max_nodes=2500, niter=5000, graph_init=None,
                      err_kernel=None, lprob_func=None, rstate=None,
                      lprob_args=None, lprob_kwargs=None, track_scale=False,
                      verbose=True):
        """
        Train the GNG using the provided set of models.

        Parameters
        ----------
        models : `~numpy.ndarray` of shape (Nmodel, Nfilt), optional
            Model values.

        models_err : `~numpy.ndarray` of shape (Nmodel, Nfilt), optional
            Associated errors on the model values.

        models_mask : `~numpy.ndarray` of shape (Nmodel, Nfilt), optional
            Binary mask (0/1) indicating whether the model value was observed.

        learn_best : float, optional
            The fractional amount to adjust the best-fit node based on the
            residual between the node and the data. Default is `0.2`.

        learn_neighbor : float, optional
            The fractional amount to adjust the topological neighbors
            of the best-fit node (i.e. nodes connected by edges) based on the
            residuals between the nodes and the data. Default is `0.005`.

        max_age : int, optional
            The maximum age an edge can be before it is removed from the
            graph. Edges are "aged" each time a node connected to them
            is selected as the best-matching unit. Default is `15`.

        nbatch : int, optional
            The number of iterations before a new node is added to the
            graph and the graph is pruned (i.e. a "batch update").
            Default is `50`.

        new_err_dec : float, optional
            Decrease the accumulated error (chi2) of the nodes in the
            topological vicinity of the new node by a factor of
            `1. - new_err_dec`. Default is `0.5`.

        all_err_dec : float, optional
            Decrease the accumulated error (chi2) of **all** nodes in the
            graph by a factor of `1. - all_err_dec`. Default is `5e-3`.

        max_nodes : int, optional
            The maximum number of allowed nodes in the graph.
            Default is `2500`.

        niter : int, optional
            The maximum number of batches allowed during training.
            Default is `5000`.

        graph_init : `~networkx.Graph` instance, optional
            The graph used to initialize the GNG. If not provided, the GNG
            will be initialized using two random models.

        err_kernel : `~numpy.ndarray` of shape (Nmodel, Nfilt), optional
            An error kernel added in quadrature to the provided
            `models_err` used when training the SOM.

        lprob_func : str or func, optional
            Log-posterior function to be used when computing fits between
            the network and the models **in the mapped feature space**
            (i.e. via the provided `feature_maps`). Must return ln(prior),
            ln(like), ln(post), Ndim, chi2, and (optionally) scale and
            scale_err. If not provided, `~frankenz.pdf.loglike` will be used.

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

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

        """

        # Initialize values.
        if lprob_func is None:
            def lprob_train(x, xe, xm, ys, yes, yms):
                results = loglike(x, xe, xm, ys, yes, yms)
                lnlike, ndim, chi2 = results
                return np.zeros_like(lnlike), lnlike, lnlike, ndim, chi2
            lprob_func = lprob_train
        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()
        if rstate is None:
            rstate = np.random

        # Load in models.
        if models is None:
            models = self.models
        if models_mask is None:
            models_mask = self.models_mask
        if models_err is None:
            models_err = self.models_err
        if err_kernel is not None:
            models_err = np.sqrt(models_err**2 + err_kernel**2)

        # Train the GNG.
        train = self._train_network
        for i, res in enumerate(train(models, models_err, models_mask,
                                      learn_best=learn_best,
                                      learn_neighbor=learn_neighbor,
                                      max_age=max_age, nbatch=nbatch,
                                      new_err_dec=new_err_dec,
                                      all_err_dec=all_err_dec,
                                      max_nodes=max_nodes,
                                      niter=niter,
                                      graph_init=graph_init,
                                      lprob_func=lprob_func,
                                      rstate=rstate, lprob_args=lprob_args,
                                      lprob_kwargs=lprob_kwargs,
                                      track_scale=track_scale)):
            fits, bmu, nnodes, nprune = res
            if i % nbatch == 0 and verbose:
                sys.stderr.write('\rIteration {0}/{1} '
                                 '[nodes={2}, edges pruned={3}] '
                                 .format(int(i/nbatch) + 1, niter, nnodes,
                                         nprune))
                sys.stderr.flush()
        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

    def _train_network(self, models, models_err, models_mask,
                       learn_best=0.2, learn_neighbor=0.005, max_age=15,
                       nbatch=50, new_err_dec=0.5, all_err_dec=5e-3,
                       max_nodes=2500, niter=5000, graph_init=None,
                       lprob_func=None, rstate=None, lprob_args=None,
                       lprob_kwargs=None, track_scale=False, verbose=True):
        """
        Train the GNG using the provided set of models.

        Parameters
        ----------
        models : `~numpy.ndarray` of shape (Nmodel, Nfilt)
            Model values.

        models_err : `~numpy.ndarray` of shape (Nmodel, Nfilt)
            Associated errors on the model values.

        models_mask : `~numpy.ndarray` of shape (Nmodel, Nfilt)
            Binary mask (0/1) indicating whether the model value was observed.

        learn_best : float, optional
            The fractional amount to adjust the best-fit node based on the
            residual between the node and the data. Default is `0.2`.

        learn_neighbor : float, optional
            The fractional amount to adjust the topological neighbors
            of the best-fit node (i.e. nodes connected by edges) based on the
            residuals between the nodes and the data. Default is `0.005`.

        max_age : int, optional
            The maximum age an edge can be before it is removed from the
            graph. Edges are "aged" each time a node connected to them
            is selected as the best-matching unit. Default is `15`.

        nbatch : int, optional
            The number of iterations before a new node is added to the
            graph and the graph is pruned (i.e. a "batch update").
            Default is `50`.

        new_err_dec : float, optional
            Decrease the accumulated error (chi2) of the nodes in the
            topological vicinity of the new node by a factor of
            `1. - new_err_dec`. Default is `0.5`.

        all_err_dec : float, optional
            Decrease the accumulated error (chi2) of **all** nodes in the
            graph by a factor of `1. - all_err_dec`. Default is `5e-3`.

        max_nodes : int, optional
            The maximum number of allowed nodes in the graph.
            Default is `2500`.

        niter : int, optional
            The maximum number of batches allowed during training.
            Default is `5000`.

        graph_init : `~networkx.Graph` instance, optional
            The graph used to initialize the GNG. If not provided, the GNG
            will be initialized using two random models.

        lprob_func : str or func, optional
            Log-posterior function to be used when computing fits between
            the network and the models **in the mapped feature space**
            (i.e. via the provided `feature_maps`). Must return ln(prior),
            ln(like), ln(post), Ndim, chi2, and (optionally) scale and
            scale_err. If not provided, `~frankenz.pdf.loglike` will be used.

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

        """

        # Initialize values.
        if lprob_func is None:
            def lprob_train(x, xe, xm, ys, yes, yms):
                results = loglike(x, xe, xm, ys, yes, yms)
                lnlike, ndim, chi2 = results
                return np.zeros_like(lnlike), lnlike, lnlike, ndim, chi2
            lprob_func = lprob_train
        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()
        if rstate is None:
            rstate = np.random

        # Initialize graph.
        Nmodel = len(models)
        if graph_init is None:
            self.graph = nx.Graph()
            i1, i2 = rstate.choice(Nmodel, size=2, replace=False)
            self.graph.add_node(0, pos=models[i1], error=0.)
            self.graph.add_node(1, pos=models[i2], error=0.)
            self.graph.add_edge(0, 1, age=0)
        else:
            self.graph = graph_init
        nnode_init = self.graph.number_of_nodes()

        # Initialize models.
        self.NNODE = self.graph.number_of_nodes()
        node_idxs = self.graph.nodes()  # grab node indices
        for count, i in enumerate(node_idxs):
            self.graph.add_node(i, count=count)  # add counter labels
        npos = nx.get_node_attributes(self.graph, 'pos')
        self.nodes = np.array([p[1] for p in iteritems(npos)])
        y = self.nodes
        ye = np.zeros_like(y)
        ym = np.ones_like(y, dtype='bool')

        prune_edges = []
        nprune = len(prune_edges)

        # Train the network.
        for i in range(niter * nbatch):

            # Draw object.
            idx = rstate.choice(Nmodel)
            x, xe, xm = models[idx], models_err[idx], models_mask[idx]

            # Fit network.
            node_results = lprob_func(x, xe, xm, y, ye, ym,
                                      *lprob_args, **lprob_kwargs)
            node_lnprob, node_chi2 = node_results[2], node_results[4]

            # Rescale models (if needed).
            if track_scale:
                node_scales = node_results[5]
                self.nodes *= node_scales[:, None]  # re-scale node models

            # Find the "best-matching unit" (BMU) and its closest competitor.
            idx_sort = np.argsort(node_lnprob)
            y_bmu, bmu = idx_sort[-1], node_idxs[idx_sort[-1]]
            bmu2 = node_idxs[idx_sort[-2]]

            # Update the BMU.
            resid = x - self.graph.node[bmu]['pos']
            y[y_bmu] += learn_best * resid
            self.graph.node[bmu]['pos'] += learn_best * resid  # update pos
            self.graph.node[bmu]['error'] += node_chi2[y_bmu]  # add error

            # Update the connection between BMU and BMU2.
            try:
                self.graph.edge[bmu][bmu2]['age'] = 0  # rejuvenate edge
            except:
                self.graph.add_edge(bmu, bmu2, age=0)  # add edge
                pass

            # Update the topological neighbors of the BMU.
            neighbors = self.graph.neighbors(bmu)
            for neighbor in neighbors:
                y_nbr = self.graph.node[neighbor]['count']
                resid = x - self.graph.node[neighbor]['pos']
                y[y_nbr] += learn_neighbor * resid
                self.graph.node[neighbor]['pos'] += learn_neighbor * resid
                self.graph.edge[bmu][neighbor]['age'] += 1  # age edge
                if self.graph.edge[bmu][neighbor]['age'] == max_age:
                    prune_edges.append((bmu, neighbor))

            # End of batch.
            if i % nbatch == 0:

                # Prune the graph and remove any edges that are too old.
                nprune = len(prune_edges)
                for e1, e2 in prune_edges:
                    try:
                        self.graph.remove_edge(e1, e2)
                        # Remove any nodes that become disconnected.
                        if not self.graph.neighbors(e1):
                            self.graph.remove_node(e1)
                        if not self.graph.neighbors(e2):
                            self.graph.remove_node(e2)
                    except:
                        pass
                prune_edges = []

                # Try to add a new node.
                if self.graph.number_of_nodes() < max_nodes:
                    # Find the node with the largest cumulative error.
                    errors = nx.get_node_attributes(self.graph, 'error')
                    errors = np.array([er for er in iteritems(errors)])
                    e1_idx = int(errors[np.argmax(errors[:, 1]), 0])
                    # Find the neighbor with the largest cumulative error.
                    e1_nbrs = self.graph.neighbors(e1_idx)
                    e2_idx = e1_nbrs[np.argmax([self.graph.node[t]['error']
                                                for t in e1_nbrs])]
                    # Adjust errors.
                    self.graph.node[e1_idx]['error'] *= (1. - new_err_dec)
                    self.graph.node[e2_idx]['error'] *= (1. - new_err_dec)
                    # Insert new node halfway between `e1_idx` and `e2_idx`.
                    new_pos = 0.5 * (self.graph.node[e1_idx]['pos'] +
                                     self.graph.node[e2_idx]['pos'])
                    new_err = self.graph.node[e1_idx]['error']
                    new_idx = nnode_init + int(i/nbatch)
                    self.graph.add_node(new_idx, pos=new_pos, error=new_err)
                    # Modify immediate edges.
                    self.graph.remove_edge(e1_idx, e2_idx)
                    self.graph.add_edge(new_idx, e1_idx, age=0)
                    self.graph.add_edge(new_idx, e2_idx, age=0)

                # Re-initialize models.
                self.NNODE = self.graph.number_of_nodes()
                node_idxs = self.graph.nodes()  # grab node indices
                [self.graph.add_node(i, count=count)
                 for count, i in enumerate(node_idxs)]  # add counter labels
                npos = nx.get_node_attributes(self.graph, 'pos')
                self.nodes = np.array([p[1] for p in iteritems(npos)])
                y = self.nodes
                ye = np.zeros_like(y)
                ym = np.ones_like(y, dtype='bool')

            # Decrease the cumulative errors within each node.
            for j in self.graph.nodes():
                self.graph.node[j]['error'] *= (1. - all_err_dec)

            yield node_results, bmu, self.NNODE, nprune
