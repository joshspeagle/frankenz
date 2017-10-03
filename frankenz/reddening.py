#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Reddening functions used when simulating observations.

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

__all__ = ["_madau_t1", "_madau_tau1", "_madau_tau2", "madau_teff"]


def _madau_t1(wave, z, l, coeff):
    """
    Applies attenuation from the IGM at a particular set of wavelengths
    `wave` at redshift `z` from an line with wavelength `l` based on
    coefficients `coeff`.

    """

    zlambda = l * (1 + z)  # redshift wavelength
    tau = np.zeros_like(wave)
    sel = wave < zlambda  # selecting all wavelengths < zlambda
    tau[sel] = coeff * (wave[sel] / l) ** 3.46  # optical depth contribution

    return tau


def _madau_tau1(wave, z):
    """
    Applies attenuation from the IGM from 912-1216 Angstroms at a particular
    set of wavelengths `wave` at redshift `z`.

    """

    tau1 = np.zeros_like(wave)
    lines = [1216.0, 1026.0, 973.0, 950.0, 938.1, 931.0, 926.5, 923.4, 921.2,
             919.6, 918.4]  # n->1 transitions (n=2,3,...,12)
    coeffs = [0.0037, 0.00177, 0.00106, 0.000584, 0.00044, 0.00040, 0.00037,
              0.00035, 0.00033, 0.00032, 0.00031]  # coefficients
    
    # Apply attenuation from absorption lines.
    for l, c in zip(lines, coeffs):
        tau1 += _madau_t1(wave, z, l, c)

    return tau1

def _madau_tau2(wave, z):
    """
    Applies attenuation from the IGM at <912 Angstroms at a particular
    set of wavelengths `wave` at redshift `z`.

    """

    zlambda = 912.0 * (1 + z)
    tau2 = np.zeros_like(wave)
    sel = wave < zlambda

    xc = wave[sel] / 912.0  # (1+z) factors assuming this was at 912A
    xem = 1. + z  # observed (1+z)
    tau2[sel] = ((0.25 * (xc**3) * (xem**0.46 - xc**0.46)) +
                 (9.4 * (xc**1.5) * (xem**0.18 - xc**0.18)) -
                 (0.7 * (xc**3) * (xc**-1.32 - xem**-1.32)) -
                 (0.023 * (xem**1.68 - xc**1.68)))  # compute optical depth
    tau2[tau2 < 0.] = 0.  # set floor to 0.

    return tau2

def madau_teff(wave, z):
    """
    Applies attenuation from the IGM at <1216 Angstroms at a particular
    set of wavelengths `wave` at redshift `z`. Returns the **effective
    transmission**.

    """

    # Compute optical depth by splicing together <912 and 912-1216 components.
    tau = _madau_tau1(wave, z) + _madau_tau2(wave,z)

    # Convert to **effective transmission**.
    teff = np.exp(-tau)

    return teff
