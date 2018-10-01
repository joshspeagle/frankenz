#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Various samplers used for population/hierarchical redshift inference.

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
from scipy import stats

__all__ = ["loglike_nz", "population_sampler", "hierarchical_sampler"]


def loglike_nz(nz, pdfs, overlap=None, return_overlap=False,
               pair=None, pair_step=None):
    """
    Compute the log-likelihood for the provided population redshift
    distribution `nz` given a collection of PDFs `pdfs`. Assumes that the
    distributions both are properly normalized and sum to 1.

    Parameters
    ----------
    nz : `~numpy.ndarray` of shape `(Nbins,)`
        The population redshift distribution.

    pdfs : `~numpy.ndarray` of shape `(Nobs, Nbins,)`
        The individual redshift PDFs that make up the sample.

    overlap : `~numpy.ndarray` of shape `(Nobs,)`
        The overlap integrals (sums) between `pdfs` and `nz`. If not provided,
        these will be computed.

    return_overlap : bool, optional
        Whether to return the overlap integrals. Default is `False`.

    pair : 2-tuple, optional
        A pair of indices `(i, j)` corresponding to a pair of bins that will
        be perturbed by `pair_step`.

    pair_step : float, optional
        The amount by which to perturb the provided pair `(i, j)` in the
        `(+, -)` direction, respectively.

    Returns
    -------
    loglike : float
        The computed log-likelihood.

    """

    # Check for negative values.
    perturb = 0.
    if np.any(~np.isfinite(nz) | (nz < 0.)):
        lnlike, overlap = -np.inf, np.zeros(len(pdfs))
    else:
        # Compute overlap.
        if overlap is None:
            overlap = np.dot(pdfs, nz)
        # Compute perturbation from pair.
        if pair is not None:
            i, j = pair
            if pair_step is not None:
                perturb = pair_step * (pdfs[:, i] - pdfs[:, j])
        # Compute log-likelihood.
        lnlike = np.sum(np.log(overlap + perturb))

    if return_overlap:
        return lnlike, overlap + perturb
    else:
        return lnlike


class population_sampler(object):
    """
    Sampler for drawing redshift population distributions given a set of
    individual redshift PDFs.

    """

    def __init__(self, pdfs):
        """
        Initialize the sampler.

        Parameters
        ----------
        pdfs : `~numpy.ndarray` of shape `(Nobs, Nbins,)`
            The individual redshift PDFs that make up the sample.

        """

        # Initialize values.
        self.pdfs = pdfs
        self.samples = []
        self.samples_lnp = []

    def reset(self):
        """Re-initialize the sampler."""

        self.samples = []
        self.samples_lnp = []

    @property
    def results(self):
        """Return samples."""

        return np.array(self.samples), np.array(self.samples_lnp)

    def run_mcmc(self, Niter, logprior_nz=None, pos_init=None,
                 thin=400, mh_steps=3, rstate=None, verbose=True,
                 prior_args=[], prior_kwargs={}):
        """
        Sample the distribution using MH-in-Gibbs MCMC.

        Parameters
        ----------
        Niter : int
            The number of samples to draw/iterations to run.

        logprior_nz : func, optional
            A function that returns the ln(prior) on `pos`.

        pos_init : `~numpy.ndarray` of shape `(Ndim,)`, optional
            The initial position from where we should start sampling.
            If not provided, the last position available from the previous
            set of samples will be used. If no samples have been drawn, the
            initial position will be the stacked PDFs.

        thin : int, optional
            The number of Gibbs samples (over random pairs) to draw
            before saving a sample. Default is `400`.

        mh_steps : int, optional
            The number of Metropolis-Hastings proposals within each Gibbs
            iteration. Default is `3`.

        rstate : `~numpy.random.RandomState`
            `~numpy.random.RandomState` instance.

        verbose : bool, optional
            Whether or not to output a simple summary of the current run that
            updates with each iteration. Default is `True`.

        prior_args : args, optional
            Optional arguments for `logprior_nz`.

        prior_kwargs : args, optional
            Optional keyword arguments for `logprior_nz`.

        """

        # Initialize values.
        Nobs, Ndim = self.pdfs.shape
        if rstate is None:
            rstate = np.random

        # Initialize prior.
        if logprior_nz is None:
            def logprior_nz(pos, *prior_args, **prior_kwargs):
                return 0.

        # Initialize starting position.
        if pos_init is None:
            try:
                # Try to start from out last position.
                pos = self.samples[-1]
            except:
                # Otherwise, just stack the individual PDFs.
                pos = self.pdfs.sum(axis=0) / self.pdfs.sum()
                pass
        else:
            # Use provided position.
            pos = pos_init

        # Sample.
        for i, (x, lnp) in enumerate(self.sample(Niter,
                                                 logprior_nz=logprior_nz,
                                                 pos_init=pos_init, thin=thin,
                                                 mh_steps=mh_steps,
                                                 rstate=rstate,
                                                 prior_args=prior_args,
                                                 prior_kwargs=prior_kwargs)):

            self.samples.append(np.array(x))
            self.samples_lnp.append(lnp)
            if verbose:
                sys.stderr.write('\r Sample {:d}/{:d} [lnpost = {:6.3f}]      '
                                 .format(i+1, Niter, lnp))
                sys.stderr.flush()

    def sample(self, Niter, logprior_nz=None, pos_init=None, thin=400,
               mh_steps=3, rstate=None, prior_args=[], prior_kwargs={}):
        """
        Internal generator used for MH-in-Gibbs MCMC sampling.

        Parameters
        ----------
        Niter : int
            The number of samples to draw/iterations to run.

        logprior_nz : func, optional
            A function that returns the ln(prior) on `pos`.

        pos_init : `~numpy.ndarray` of shape `(Ndim,)`, optional
            The initial position from where we should start sampling.
            If not provided, the last position available from the previous
            set of samples will be used. If no samples have been drawn, the
            initial position will be the stacked PDFs.

        thin : int, optional
            The number of Gibbs samples (over random pairs) to draw
            before saving a sample. Default is `400`.

        mh_steps : int, optional
            The number of Metropolis-Hastings proposals within each Gibbs
            iteration. Default is `3`.

        rstate : `~numpy.random.RandomState`
            `~numpy.random.RandomState` instance.

        verbose : bool, optional
            Whether or not to output a simple summary of the current run that
            updates with each iteration. Default is `True`.

        prior_args : args, optional
            Optional arguments for `logprior_nz`.

        prior_kwargs : args, optional
            Optional keyword arguments for `logprior_nz`.

        """

        # Initialize values.
        Nobs, Ndim = self.pdfs.shape
        if rstate is None:
            rstate = np.random

        # Initialize prior.
        if logprior_nz is None:
            def logprior_nz(pos, *prior_args, **prior_kwargs):
                return 0.

        # Initialize starting position.
        if pos_init is None:
            pos = self.pdfs.sum(axis=0) / self.pdfs.sum()
        else:
            pos = pos_init
        lnlike, overlap = loglike_nz(pos, self.pdfs, return_overlap=True)
        lnprior = logprior_nz(pos, *prior_args, **prior_kwargs)
        lnpost = lnlike + lnprior

        # Sample.
        for i in range(Niter):
            # Generate random pairs.
            pairs = [rstate.choice(Ndim, size=2, replace=False)
                     for i in range(thin)]
            # Gibbs step.
            for pair in pairs:
                # Generate (i, j) basis vector.
                t = np.zeros_like(pos)
                t[pair] = (1, -1)
                # Compute absolute range.
                scale = 1e-4 * np.min(np.append(pos[pair], 1. - pos[pair]))
                # Compute numerical gradient.
                lnp1 = loglike_nz(pos, self.pdfs, overlap=overlap,
                                  pair=pair, pair_step=scale/2.)
                lnp1 += logprior_nz(pos + t*scale/2.,
                                    *prior_args, **prior_kwargs)
                lnp2 = loglike_nz(pos, self.pdfs, overlap=overlap,
                                  pair=pair, pair_step=-scale/2.)
                lnp2 += logprior_nz(pos - t*scale/2.,
                                    *prior_args, **prior_kwargs)
                grad = (lnp1 - lnp2) / scale
                # Rescale so that we're looking at changes in log(post) of ~ 1.
                if grad != 0.:
                    gscale = min(abs(1. / grad), abs(scale * 1e4))
                else:
                    gscale = abs(scale)

                # Metropolis-Hastings step.
                for k in range(mh_steps):
                    # Generate proposal.
                    z = rstate.randn() * gscale
                    # Generate new proposal.
                    pos_new = pos + (t * z)
                    lnlike_new, overlap_new = loglike_nz(pos_new, self.pdfs,
                                                         overlap=overlap,
                                                         return_overlap=True,
                                                         pair=pair,
                                                         pair_step=z)
                    lnprior_new = logprior_nz(pos_new,
                                              *prior_args, **prior_kwargs)
                    lnpost_new = lnlike_new + lnprior_new
                    # Metropolis update.
                    if -rstate.exponential() < lnpost_new - lnpost:
                        pos, lnpost, overlap = pos_new, lnpost_new, overlap_new

            # Return current position.
            yield pos, lnpost


class hierarchical_sampler(object):
    """
    Sampler for jointly drawing redshift population distributions and
    individual redshift predictions given a set of individual redshift PDFs.
    Note that these must be *likelihoods*, since the prior is being explicitly
    modeled. Assumes a Dirichlet hyper-prior.

    """

    def __init__(self, pdfs):
        """
        Initialize the sampler.

        Parameters
        ----------
        pdfs : `~numpy.ndarray` of shape `(Nobs, Nbins,)`
            The individual redshift PDFs that make up the sample.

        """

        # Initialize values.
        self.pdfs = pdfs
        self.samples = []
        self.samples_lnp = []

    def reset(self):
        """Re-initialize the sampler."""

        self.samples_prior = []
        self.samples_lnp = []
        self.samples_counts = []

    @property
    def results(self):
        """Return samples."""

        return np.array(self.samples), np.array(self.samples_lnp)

    def run_mcmc(self, Niter, alpha=None, pos_init=None,
                 thin=5, ref_sample=None, beta=None, rstate=None,
                 verbose=True):
        """
        Sample the joint distribution using Gibbs MCMC.

        Parameters
        ----------
        Niter : int
            The number of samples to draw/iterations to run.

        alpha : `~numpy.ndarray` of shape `(Ndim,)`, optional
            The concentration parameters for the Dirichlet hyper-prior.
            If not provided, a flat `alpha = 1.` will be assumed.

        pos_init : `~numpy.ndarray` of shape `(Ndim,)`, optional
            The initial position from where we should start sampling.
            If not provided, the last position available from the previous
            set of samples will be used. If no samples have been drawn, the
            initial position will be the stacked PDFs.

        thin : int, optional
            The number of Gibbs samples (over random pairs) to draw
            before saving a sample. Default is `5`.

        ref_sample : `~numpy.ndarray` of shape `(Ndim,)`, optional
            A set of observed counts from a reference sample. If passed,
            these will be included in the hierarchical model.

        beta : `~numpy.ndarray` of shape `(Ndim,)`, optional
            The concentration parameters for the Dirichlet hyper-prior for
            the reference sample. If not provided, a flat `beta = 1.`
            will be assumed.

        rstate : `~numpy.random.RandomState`
            `~numpy.random.RandomState` instance.

        verbose : bool, optional
            Whether or not to output a simple summary of the current run that
            updates with each iteration. Default is `True`.

        """

        # Initialize values.
        Nobs, Ndim = self.pdfs.shape
        if rstate is None:
            rstate = np.random

        # Initialize prior.
        if alpha is None:
            alpha = np.ones(Ndim)
        if beta is None:
            beta = np.ones(Ndim)

        # Initialize starting position.
        if pos_init is None:
            try:
                # Try to start from out last position.
                pos = self.samples[-1]
            except:
                # Otherwise, just stack the individual PDFs.
                pos = self.pdfs.sum(axis=0) / self.pdfs.sum()
                pass
        else:
            # Use provided position.
            pos = pos_init

        # Sample.
        for i, (x, lnp) in enumerate(self.sample(Niter, alpha=alpha, beta=beta,
                                                 pos_init=pos_init, thin=thin,
                                                 ref_sample=ref_sample,
                                                 rstate=rstate)):

            self.samples.append(np.array(x))
            self.samples_lnp.append(lnp)
            if verbose:
                sys.stderr.write('\r Sample {:d}/{:d} [lnpost = {:6.3f}]      '
                                 .format(i+1, Niter, lnp))
                sys.stderr.flush()

    def sample(self, Niter, alpha=None, pos_init=None, thin=5,
               ref_sample=None, beta=None, rstate=None):
        """
        Internal generator used for MH-in-Gibbs MCMC sampling.

        Parameters
        ----------
        Niter : int
            The number of samples to draw/iterations to run.

        alpha : `~numpy.ndarray` of shape `(Ndim,)`, optional
            The concentration parameters for the Dirichlet hyper-prior.
            If not provided, a flat `alpha = 1.` will be assumed.

        pos_init : `~numpy.ndarray` of shape `(Ndim,)`, optional
            The initial position from where we should start sampling.
            If not provided, the last position available from the previous
            set of samples will be used. If no samples have been drawn, the
            initial position will be the stacked PDFs.

        thin : int, optional
            The number of Gibbs samples (over random pairs) to draw
            before saving a sample. Default is `5`.

        ref_sample : `~numpy.ndarray` of shape `(Ndim,)`, optional
            A set of observed counts from a reference sample. If passed,
            these will be included in the hierarchical model.

        beta : `~numpy.ndarray` of shape `(Ndim,)`, optional
            The concentration parameters for the Dirichlet hyper-prior for
            the reference sample. If not provided, a flat `beta = 1.`
            will be assumed.

        rstate : `~numpy.random.RandomState`
            `~numpy.random.RandomState` instance.

        verbose : bool, optional
            Whether or not to output a simple summary of the current run that
            updates with each iteration. Default is `True`.

        """

        # Initialize values.
        Nobs, Ndim = self.pdfs.shape
        if rstate is None:
            rstate = np.random

        # Initialize prior.
        if alpha is None:
            alpha = np.ones(Ndim)
        if beta is None:
            beta = np.ones(Ndim)

        # Initialize reference sample.
        if ref_sample is not None:
            ref_counts = np.array(ref_sample)
            ref_norm = ref_sample + beta
            ref_norm /= ref_norm.sum()
            Nref = sum(ref_counts)
        else:
            ref_counts = np.zeros(Ndim)
            Nref = 0

        # Initialize starting position.
        if pos_init is None:
            pos = self.pdfs.sum(axis=0) / self.pdfs.sum()
        else:
            pos = pos_init
        # Sample redshifts.
        counts = np.sum([rstate.multinomial(1, p * pos / np.dot(p, pos))
                         for p in self.pdfs], axis=0)
        # Sample population.
        pos = rstate.dirichlet(alpha + counts + ref_counts)
        # Sample reference set.
        if ref_sample is not None:
            pcounts = ref_sample + beta + Nobs * pos
            ref_counts = rstate.multinomial(Nref, pcounts / pcounts.sum())
        # Evaluate posterior.
        lnlike = stats.multinomial.logpmf(counts, Nobs, pos)
        lnprior = stats.dirichlet.logpdf(pos, alpha + ref_counts)
        if ref_sample is not None:
            lnpriorref = stats.multinomial.logpmf(ref_counts, Nref, ref_norm)
        else:
            lnpriorref = 0.
        lnpost = lnlike + lnprior + lnpriorref

        # Sample.
        for i in range(Niter):
            for j in range(thin):
                # Sample redshifts.
                counts = np.sum([rstate.multinomial(1, p*pos / np.dot(p, pos))
                                 for p in self.pdfs], axis=0)
                # Sample population.
                pos = rstate.dirichlet(alpha + counts + ref_counts)
                # Sample reference set.
                if ref_sample is not None:
                    pcounts = ref_sample + beta + Nobs * pos
                    ref_counts = rstate.multinomial(Nref, (pcounts /
                                                           pcounts.sum()))
                    lnpriorref = stats.multinomial.logpmf(ref_counts, Nref,
                                                          ref_norm)
                lnlike = stats.multinomial.logpmf(counts, Nobs, pos)
                lnprior = stats.dirichlet.logpdf(pos, alpha + ref_counts)
                lnpost = lnlike + lnprior + lnpriorref

            # Return current position.
            yield pos, lnpost
