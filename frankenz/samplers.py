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

__all__ = ["loglike_nz", "population_sampler"]


def loglike_nz(nz, pdfs):
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

    Returns
    -------
    loglike : float
        The computed log-likelihood.

    """

    # Check for negative values.
    if np.any(nz < 0.):
        lnlike = -np.inf
    else:
        # Compute log-likelihood.
        lnlike = np.sum(np.log(np.dot(pdfs, nz)))

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
                 thin=None, mh_steps=5, rstate=None, verbose=True,
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
            before saving a sample. Default is `Ndim`.

        mh_steps : int, optional
            The number of Metropolis-Hastings proposals within each Gibbs
            iteration. Default is `10`.

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
        if thin is None:
            thin = Ndim
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
        lnlike = loglike_nz(pos, self.pdfs)
        lnprior = logprior_nz(pos, *prior_args, **prior_kwargs)
        lnpost = lnlike + lnprior

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

    def sample(self, Niter, logprior_nz=None, pos_init=None, thin=None,
               mh_steps=5, rstate=None, prior_args=[], prior_kwargs={}):
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
            before saving a sample. Default is `Ndim`.

        mh_steps : int, optional
            The number of Metropolis-Hastings proposals within each Gibbs
            iteration. Default is `10`.

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
        if thin is None:
            thin = Ndim
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
        lnlike = loglike_nz(pos, self.pdfs)
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
                lnp1 = loglike_nz(pos + t*scale/2., self.pdfs)
                lnp1 += logprior_nz(pos + t*scale/2.,
                                    *prior_args, **prior_kwargs)
                lnp2 = loglike_nz(pos - t*scale/2., self.pdfs)
                lnp2 += logprior_nz(pos - t*scale/2.,
                                    *prior_args, **prior_kwargs)
                grad = (lnp1 - lnp2) / scale
                # Rescale so that we're looking at changes in log(post) of ~ 1.
                gscale = min(1. / grad, scale)

                # Metropolis-Hastings step.
                for k in range(mh_steps):
                    # Generate proposal.
                    z = rstate.randn() * gscale
                    # Generate new proposal.
                    pos_new = pos + (t * z)
                    lnlike_new = loglike_nz(pos_new, self.pdfs)
                    lnprior_new = logprior_nz(pos_new,
                                              *prior_args, **prior_kwargs)
                    lnpost_new = lnlike_new + lnprior_new
                    # Metropolis update.
                    if -rstate.exponential() < lnpost_new - lnpost:
                        pos, lnpost = pos_new, lnpost_new

            # Return current position.
            yield pos, lnpost
