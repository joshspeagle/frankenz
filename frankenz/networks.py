#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Oject used to fit data and compute PDFs using brute-force methods.

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

__all__ = ["SelfOrganizingMap", "GrowingNeuralGas", "_Network"]


class _Network():
    """
    Fits data and generates predictions using a network of nodes (models)
    and the data used to train it.

    """

    def __init__(self, models, models_err, models_mask):
        """
        Load the model data into memory. This will be replaced by child
        classes.

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

        self.nodes = None
        self.nodes_pos = None
        self.nodes_idxs = None
        self.nodes_logwts = None
        self.nodes_scales = None
        self.nodes_Nmatch = None
        self.nodes_only = None
        self.NNODE = None
        self.NPROJ = None

        self.neighbors = None
        self.Nneighbors = None

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
        Returns quantities associated with the given node.

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
        lmap, levid = max(lwt), logsumexp(lwt)  # model GOF metrics
        wt = np.exp(lwt - levid)
        if label_dict is not None:
            # Use dictionary if available.
            y_idx, y_std_idx = label_dict.fit(model_labels, model_label_errs)
            pdf = gauss_kde_dict(label_dict, y_idx=y_idx[idxs],
                                 y_std_idx=y_std_idx[idxs], y_wt=wt,
                                 *kde_args, **kde_kwargs)
        else:
            # Otherwise just use KDE.
            pdf = gauss_kde(model_labels[idxs], model_label_errs[idxs],
                            label_grid, y_wt=wt, *kde_args, **kde_kwargs)

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
                                label_grid, y_wt=wt, *kde_args, **kde_kwargs)

            yield pdf, (lmap, levid)

    def fit(self, data, data_err, data_mask, lprob_func, nodes_only=False,
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
            ln(post), Ndim, chi2, and (optionally) scale.

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
        if lprob_args is None:
            lprob_args = []
        if lprob_kwargs is None:
            lprob_kwargs = dict()
        if wt_thresh is None and cdf_thresh is None:
            wt_thresh = -np.inf  # default to no clipping/thresholding
        Ndata = len(data)

        # Fit data.
        for i, results in enumerate(self._fit(data, data_err, data_mask,
                                              lprob_func,
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

    def _fit(self, data, data_err, data_mask, lprob_func, nodes_only=False,
             wt_thresh=1e-3, cdf_thresh=2e-4, lprob_args=None,
             lprob_kwargs=None, track_scale=False):
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
            ln(post), Ndim, chi2, and (optionally) scale.

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

        y = self.nodes
        ye = np.zeros_like(y)
        ym = np.ones_like(y, dtype='bool')

        if nodes_only:
            self.nodes_only = True
        else:
            self.nodes_only = False

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
                sel_arr = np.arange(Nnodes)[node_lnprob > lwt_min]
            else:
                # Use CDF to threshold.
                idx_sort = np.argsort(y_wt)
                node_prob = np.exp(node_lnprob - logsumexp(node_lnprob))
                node_cdf = np.cumsum(node_prob[idx_sort])
                sel_arr = idx_sort[node_cdf <= (1. - cdf_thresh)]

            if nodes_only:
                # Take our nodes to be our models.
                self.Nneighbors[i] = len(sel_arr)
                self.neighbors.append(sel_arr)
                results = [x[sel_arr] for x in node_results]
            else:
                # Unique neighbor selection based on network fits.
                indices = np.array([idx for idx in self.nodes_idxs[sidx]
                                    for sidx in sel_arr])
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
            if self.nodes_only:
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

    def fit_predict(self, data, data_err, data_mask, lprob_func,
                    model_labels, model_label_errs, nodes_only=False,
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

        lprob_func : str or func, optional
            Log-posterior function to be used. Must return ln(prior), ln(like),
            ln(post), Ndim, chi2, and (optionally) scale.

        model_labels : `~numpy.ndarray` of shape (Nmodel)
            Model values.

        model_label_errs : `~numpy.ndarray` of shape (Nmodel)
            Associated errors on the data values.

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
                                                  lprob_func, model_labels,
                                                  model_label_errs,
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

    def _fit_predict(self, data, data_err, data_mask, lprob_func,
                     model_labels, model_label_errs, node_pdfs=None,
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

        lprob_func : str or func, optional
            Log-posterior function to be used. Must return ln(prior), ln(like),
            ln(post), Ndim, chi2, and (optionally) scale.

        model_labels : `~numpy.ndarray` of shape (Nmodel)
            Model values.

        model_label_errs : `~numpy.ndarray` of shape (Nmodel)
            Associated errors on the data values.

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
        y = self.nodes
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
                sel_arr = np.arange(Nnodes)[node_lnprob > lwt_min]
            else:
                # Use CDF to threshold.
                idx_sort = np.argsort(y_wt)
                node_prob = np.exp(node_lnprob - logsumexp(node_lnprob))
                node_cdf = np.cumsum(node_prob[idx_sort])
                sel_arr = idx_sort[node_cdf <= (1. - cdf_thresh)]

            if nodes_only:
                # Take our nodes to be our models.
                idxs = sel_arr
                results = [x[idxs] for x in node_results]
                if save_fits:
                    self.Nneighbors[i] = len(idxs)
                    self.neighbors.append(np.array(idxs))
            else:
                # Unique neighbor selection based on network fits.
                indices = np.array([idx for idx in self.nodes_idxs[sidx]
                                    for sidx in sel_arr])
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

            # Compute PDF.
            lnprob = results[2]  # reduced set of posteriors
            lmap, levid = max(lnprob), logsumexp(lnprob)
            wt = np.exp(lnprob - levid)
            if self.nodes_only:
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
        Load the model data into memory and train the SOM.

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

        self.nodes = None
        self.nodes_pos = None
        self.nodes_idxs = None
        self.nodes_logwts = None
        self.nodes_scales = None
        self.nodes_Nmatch = None
        self.nodes_only = None
        self.NNODE = None
        self.NPROJ = None

        self.neighbors = None
        self.Nneighbors = None

    def _train_network(self):
        """
        Internal method used to train the SOM.

        """

        pass

    def _populate_network(self):
        """
        Internal method used to map the models back onto the resulting nodes
        of the SOM.

        """

        pass


class GrowingNeuralGas(_Network):
    """
    Fits data and generates predictions using a Growing Neural Gas (GNG).

    """

    def __init__(self, models, models_err, models_mask):
        """
        Load the model data into memory and trains the GNG.

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

        self.nodes = None
        self.nodes_pos = None
        self.nodes_idxs = None
        self.nodes_logwts = None
        self.nodes_scales = None
        self.nodes_Nmatch = None
        self.nodes_only = None
        self.NNODE = None
        self.NPROJ = None

        self.neighbors = None
        self.Nneighbors = None

    def _train_network(self):
        """
        Internal method used to train the GNG.

        """

        pass

    def _populate_network(self):
        """
        Internal method used to map the models back onto the resulting nodes
        of the GNG.

        """

        pass

    def _project_network(self):
        """
        Internal method used to project the GNG onto a lower dimensional
        subspace.

        """

        pass
