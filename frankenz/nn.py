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

__all__ = ["kNNMC"]


class kNNKMC():
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
        Monte Carlo methods. Default is `25`.

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
