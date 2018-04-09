#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate photometric observations.

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
from . import priors
from . import reddening

__all__ = ["mag_err", "draw_mag", "draw_type_given_mag",
           "draw_redshift_given_type_mag", "draw_ztm", "MockSurvey"]

# Filter lists for pre-set surveys.
_FILTERS = {'cosmos': 'COSMOS.list',
            'euclid': 'Euclid.list',
            'hsc': 'HSC.list',
            'lsst': 'LSST.list',
            'sdss': 'SDSS.list'}

# Reference magnitudes for pre-set surveys.
_REFMAGS = {'cosmos': 'i+',
            'euclid': 'VIS',
            'hsc': 'i',
            'lsst': 'r',
            'sdss': 'r'}

# Pre-set collection of templates.
_TEMPLATES = {'brown': 'BROWN.list',
              'cww+': 'CWWSB4.list',
              'polletta+': 'POLLETTASB.list'}

# Pre-set P(z,t,m) priors.
_PRIORS = {'bpz': (priors.pmag, priors.bpz_pt_m, priors.bpz_pz_tm)}

# Pre-set IGM attenuation curves.
_IGM = {'madau+99': reddening.madau_teff}

# Useful constants.
c = 299792458.0  # speed of light in m/s


def mag_err(mag, maglim, sigdet=5., params=(4.56, 1., 1.)):
    """
    Compute the magnitude error as a function of a given detection limit
    following Rykoff et al. (2015).

    Parameters
    ----------
    mag : float or `~numpy.ndarray`
        Target magnitude.

    maglim : float
        Magnitude limit.

    sigdet : float, optional
        The `sigdet`-sigma detection limit used for `maglim`. Default is `5.`.

    params : tuple of shape (3,), optional
        Free parameters `(a, b, k)` used in the functional form given by
        Rykoff et al. (2015). Default is `(4.56, 1., 1.)`.

    Returns
    -------
    magerr : float or `~numpy.ndarray`
        Corresponding magnitude error.

    """

    # Set parameters.
    a, b, k = params
    teff = np.exp(a + b * (maglim - 21.))

    # Compute flux/limit.
    F = 10**(-0.4 * (m - 22.5))
    Flim = 10**(-0.4 * (mlim - 22.5))

    # Compute noise.
    Fnoise = (Flim / sigmadet)**2 * k * teff - Flim
    magerr = 2.5 / np.log(10.) * np.sqrt((1. + Fnoise / F) / (F * k * teff))

    return magerr


def draw_mag(Nobj, pmag, rstate=None, pmag_kwargs=None, mbounds=(10, 28),
             Npoints=1000):
    """
    Draw `Nobj` magnitudes from the P(mag) function :meth:`pmag`.

    Parameters
    ----------
    Nobj : int
        The number of objects to draw.

    pmag : function
        The P(mag) function that magnitudes will be drawn from.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance. If not given, the
         global random state of the `~numpy.random` module will be used.

    pmag_kwargs : dict, optional
        Additional keyword arguments to be passed to :meth:`pmag`.

    mbounds : tuple of length 2, optional
        The minimum/maximum magnitude used to truncate :meth:`pmag`. Default is
        `(10, 28)`.

    Npoints : int, optional
        The number of points used when interpolating the inverse cumulative
        distribution function (CDF). Default is `1000`.

    Returns
    -------
    mags : `~numpy.ndarray` of shape (Nobj,)
        The magnitudes of the simulated objects drawn from :meth:`pmag`.

    """

    if pmag_kwargs is None:
        pmag_kwargs = dict()
    if mbounds[0] >= mbounds[1]:
        raise ValueError("The values {0} in `mbounds` are incorrectly "
                         "ordered.".format(mbounds))
    if rstate is None:
        rstate = np.random

    # Construct the CDF.
    mgrid = np.linspace(mbounds[0], mbounds[1], Npoints)  # compute mag grid
    pdf_m = pmag(mgrid, **pmag_kwargs)  # compute P(m) over grid
    cdf_m = pdf_m.cumsum()  # compute unnormalized CDF F(x)
    cdf_m = np.append(0, cdf_m) / cdf_m[-1]  # normalize and left-pad F(x)
    lpad = 1e-5 * (mbounds[1] - mbounds[0])  # compute left padding for x
    mgrid = np.append(mgrid[0] - lpad, mgrid)  # left pad x to match F(x)

    # Sample from the inverse CDF F^-1(x).
    mags = np.interp(rstate.rand(Nobj), cdf_m, mgrid)

    return mags


def draw_type_given_mag(p_type_given_mag, mags, Ntypes, rstate=None,
                        ptm_kwargs=None):
    """
    Draw corresponding types from P(type | mag) using the
    :meth:`p_type_given_mag` function. Returns a generator.

    Parameters
    ----------
    p_type_mag : function
        Function that returns the probability of an object's type at a
        given magnitude. Output should be an `~numpy.ndarray` with shape
        (`Ntypes`,).

    mags : iterable of shape (N,)
        Set of input magnitudes.

    Ntypes : int
        The number of types we can draw.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance. If not given, the
         global random state of the `~numpy.random` module will be used.

    ptm_kwargs : dict, optional
        Additional keyword arguments to be passed to :meth:`p_type_given_mag`.

    Returns
    -------
    types : `~numpy.ndarray` of shape (N,)
        The types of the simulated objects drawn from :meth:`p_type_given_mag`
        given `mags`.

    """

    if ptm_kwargs is None:
        ptm_kwargs = dict()
    if rstate is None:
        rstate = np.random

    # Draw types.
    types = np.arange(-1, Ntypes)
    for i, m in enumerate(mags):
        prob = np.array([p_type_given_mag(t, m, **ptm_kwargs)
                         for t in range(Ntypes)])
        cdf = np.append(0., prob).cumsum()  # compute augmented CDF
        cdf /= cdf[-1]  # normalize
        draw = int(np.interp(rstate.rand(), cdf, types) + 1)  # index
        yield draw  # pass back draw as a generator


def draw_redshift_given_type_mag(p_z_tm, types, mags, rstate=None,
                                 pztm_kwargs=None, zbounds=(0, 15),
                                 Npoints=1000):
    """
    Draw corresponding redshifts from P(z | type, mag) using the
    :meth:`p_ztm` function. Returns a generator.

    Parameters
    ----------
    p_z_tm : function
        Function that takes in `z`, `t`, and `m` and returns a
        probability P(z | t, m).

    types : iterable of shape (N,)
        Set of input types.

    mags : iterable of shape (N,)
        Set of input magnitudes.

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance. If not given, the
         global random state of the `~numpy.random` module will be used.

    pztm_kwargs : dict, optional
        Additional keyword arguments to be passed to :meth:`p_ztm`.

    zbounds : tuple of length 2, optional
        The minimum/maximum redshift allowed. Default is
        `(0, 15)`.

    Npoints : int, optional
        The number of points used when interpolating the inverse cumulative
        distribution function (CDF). Default is `1000`.

    Returns
    -------
    redshifts : `~numpy.ndarray` of shape (Nobj,)
        The redshifts of the simulated objects drawn from :meth:`p_ztm`.

    """

    if pztm_kwargs is None:
        pztm_kwargs = dict()
    if zbounds[0] >= zbounds[1]:
        raise ValueError("The values {0} in `zbounds` are incorrectly "
                         "ordered.".format(zbounds))
    if rstate is None:
        rstate = np.random

    # Compute the redshift grid.
    zgrid = np.linspace(zbounds[0], zbounds[1], Npoints)
    lpad = 1e-5 * (zbounds[1] - zbounds[0])  # compute left padding for z
    zgrid2 = np.append(zgrid[0] - lpad, zgrid)  # zgrid with left padding

    # Draw redshifts.
    for i, (t, m) in enumerate(zip(types, mags)):

        # Compute PDF.
        try:
            pdf_z = p_z_tm(z=zgrid, t=t, m=m, **pztm_kwargs)
        except:
            pdf_z = np.array([p_z_tm(z=z, t=t, m=m, **pztm_kwargs)
                              for z in zgrid])

        # Compute (augmented) CDF.
        cdf_z = pdf_z.cumsum()
        cdf_z = np.append(0, cdf_z) / cdf_z[-1]  # left pad and normalize

        # Draw redshift from inverse CDF F^-1(x).
        redshift = max(0., np.interp(rstate.rand(), cdf_z, zgrid2))
        yield redshift


def draw_ztm(pmag, p_tm, p_ztm, Nobj, pm_kwargs=None, ptm_kwargs=None,
             pztm_kwargs=None, mbounds=(10, 28), zbound=(0, 15), Npoints=1000):
    """
    Draw `Nobj` redshifts, types, and magnitudes from P(z, type, mag) using the
    input P(m) :meth:`pmag`, P(t | m) :meth:`p_tm`, and P(z | t, m)
    :meth:`p_ztm` functions.

    Parameters
    ----------
    pmag : function
        Function that returns P(mag).

    p_tm : function
        Function that takes in `mag` and returns an `~numpy.ndarray`
        of shape (`Ntypes`,) corresponding to P(type | mag).

    p_ztm : function
        Function that takes in `z`, `t`, and `m` and returns P(z | t, m).

    Nobj : int
        The number of instances that should be returned.

    pm_kwargs : dict, optional
        Additional keyword arguments to be passed to :meth:`pmag`.

    ptm_kwargs : dict, optional
        Additional keyword arguments to be passed to :meth:`p_tm`.

    pztm_kwargs : dict, optional
        Additional keyword arguments to be passed to :meth:`p_ztm`.

    mbounds : tuple of length 2, optional
        The minimum/maximum magnitude allowed. Default is
        `(10, 28)`.

    zbounds : tuple of length 2, optional
        The minimum/maximum redshift allowed. Default is
        `(0, 15)`.

    Npoints : int, optional
        The number of points used when interpolating the inverse cumulative
        distribution function (CDF). Default is `1000`.

    Returns
    -------
    mags : `~numpy.ndarray` of shape (Nobj,)
        Magnitudes of the simulated objects.

    types : `~numpy.ndarray` of shape (Nobj,)
        Types of the simulated objects.

    redshifts : `~numpy.ndarray` of shape (Nobj,)
        Redshifts of the simulated objects.

    """

    # Draw magnitudes.
    mags = np.zeros(Nobj, dtype='float')
    mags = draw_mag(Nobj, pmag, pmag_kwargs=pm_kwargs, mbounds=mbounds,
                    Npoints=Npoints)

    # Draw types.
    types = np.zeros(Nobj, dtype='int')
    for i, t in enumerate(draw_type_given_mag(p_tm, mags,
                          ptm_kwargs=ptm_kwargs)):
        types[i] = t

    # Draw redshifts.
    redshifts = np.zeros(Nobj, dtype='float')
    for i, z in enumerate(draw_redshift_given_type_mag(p_ztm, types, mags,
                                                       pztm_kwargs=pztm_kwargs,
                                                       zbounds=zbounds,
                                                       Npoints=Npoints)):
        redshifts[i] = z

    return mags, types, redshifts


class MockSurvey(object):
    """
    A mock survey object used to generate and store mock data.

    Parameters
    ----------
    survey : str, optional
        If provided, will initialize the `MockSurvey` using one of several
        built-in presets:

        * COSMOS (`'cosmos'`),
        * *Euclid* (`'euclid'`),
        * HSC SSP (`'hsc'`),
        * LSST (`'lsst'`), and
        * SDSS (`'sdss'`).

    templates : str, optional
        If provided, will initialize the `MockSurvey` using one of several
        built-in template libraries:

        * 129 galaxies from Brown et al. (2014) (`'brown'`),
        * 8 templates generated using a combination of galaxies
          from Coleman, Wu & Weeman (1980) and synthetic spectra from Bruzual
          & Charlot (2003) spectral models (`'cww+'`), and
        * 31 templates generated using a combination of galaxies from
          Polletta et al. (2006) and synthetic spectra from Bruzual & Charlot
          (2003) (`'polletta+'`).

    prior : str or tuple of shape (3,), optional
        If a string provided, will initialize the `MockSurvey` using a preset
        P(z, type, mag) prior. Otherwise, if a tuple containing P(mag),
        P(type | mag), and P(z | type, mag) functions of the form
        `(p_m, p_tm, p_ztm)` is provided, those functions will be initialized
        instead. Current presets include:

        * The Bayesian Photo-Z (BPZ) prior described in Benitez (2000)
          (`'bpz'`).

    rstate : `~numpy.random.RandomState`, optional
        `~numpy.random.RandomState` instance. If not given, the
         global random state of the `~numpy.random` module will be used.

    """

    def __init__(self, survey=None, templates=None, prior=None, rstate=None):

        # filters
        self.filters = None
        self.NFILTER = None
        self.ref_filter = None

        # templates
        self.templates = None
        self.NTEMPLATE = None
        self.TYPES = None
        self.TYPE_COUNTS = None
        self.NTYPE = None

        # priors
        self.pm = None
        self.ptm = None
        self.pztm = None

        # mock data
        self.data = None

        if survey is not None:
            if survey in _FILTERS:
                self.load_survey(survey)
                self.set_refmag(_REFMAGS[survey])
            else:
                raise ValueError("{0} does not appear to be valid survey "
                                 "preset.".format(survey))
        if templates is not None:
            if templates in _TEMPLATES:
                self.load_templates(templates)
            else:
                raise ValueError("{0} does not appear to be valid template "
                                 "preset.".format(templates))

        if prior is not None:
            if prior in _PRIORS:
                self.load_prior(prior)
            else:
                raise ValueError("{0} does not appear to be valid prior "
                                 "preset.".format(prior))

        if rstate is None:
            self.rstate = np.random

    def load_survey(self, filter_list, path='', Npoints=5e4):
        """
        Load an input filter list and associated depths for a particular
        survey. Results are stored internally under `filters`.

        Parameters
        ----------
        filter_list : str
            A list of filters to import. This can be a string from a
            collection of built-in surveys or a corresponding file in the
            proper format (see `frankenz/filters/README.txt`).

        path : str, optional
            The filepath appended to `filter_list`. Also used to search for
            filters. If `filter_list` is one of the pre-specified options
            above, this defaults to `None`.

        Npoints : int, optional
            The number of points used to interpolate the filter transmission
            curves when computing the effective wavelength. Default is `5e4`.

        """

        # Get filter list.
        try:
            filter_list = _FILTERS[filter_list]
            path = os.path.dirname(os.path.realpath(__file__)) + '/filters/'
        except:
            pass

        # Load filter list.
        f = open(path + filter_list)
        self.filters = []
        filter_paths = []
        for line in f:
            index, name, fpath, fdepth_mag = line.split()
            fdepth_mag = float(fdepth_mag)
            fdepth_flux = 10**((fdepth_mag - 23.9) / -2.5) / 5.  # noise [uJy]
            fltr = {'index': int(index), 'name': name,
                    'depth_mag5sig': fdepth_mag, 'depth_flux1sig': fdepth_flux}
            self.filters.append(fltr)
            filter_paths.append(fpath)
        f.close()

        self.NFILTER = len(self.filters)  # number of filters

        # Extract filters.
        for fpath, fltr in zip(filter_paths, self.filters):
            wavelength, transmission = np.loadtxt(path + fpath).T
            fltr['wavelength'] = wavelength
            fltr['transmission'] = transmission
            fltr['frequency'] = c / (1e-10 * wavelength)

        # Compute effective wavelengths.
        for fltr in self.filters:
            nuMax = 0.999 * c / (min(fltr['wavelength']) * 1e-10)  # max nu
            nuMin = 1.001 * c / (max(fltr['wavelength']) * 1e-10)  # min nu
            nu = np.linspace(nuMin, nuMax, Npoints)  # frequency array
            lnu = np.log(nu)  # ln(frequency)
            wave = c / nu  # convert to wavelength
            lwave = np.log(wave)  # ln(wavelength)
            trans = np.interp(1e10 * wave, fltr['wavelength'],
                              fltr['transmission'])  # interp transmission
            lambda_eff = np.exp(np.trapz(trans * lwave, lnu) /
                                np.trapz(trans, lnu)) * 1e10  # integrate
            fltr['lambda_eff'] = lambda_eff

    def load_templates(self, template_list, path='', wnorm=7000.):
        """
        Load an input template list. Results are stored internally under
        `templates`.

        Parameters
        ----------
        template_list : str
            A list of templates to import. This can be a string from a
            collection of built-in template lists or a corresponding file in
            the proper format (see `frankenz/templates/README.txt`).

        path : str, optional
            The filepath appended to `template_list`. Also used to search for
            templates. If `template_list` is one of the pre-specified options
            above, this defaults to `None`.

        wnorm : float, optional
            The "pivot wavelength" [A] where templates will be normalized.
            Default is `7000.`.

        """

        # Get template list.
        try:
            template_list = _TEMPLATES[template_list]
            path = os.path.dirname(os.path.realpath(__file__)) + '/seds/'
        except:
            pass

        # Load template list.
        f = open(path + template_list)
        self.templates = []
        template_paths = []
        for line in f:
            index, name, obj_type, fpath = line.split()
            tmp = {'index': int(index), 'name': name, 'type': obj_type}
            self.templates.append(tmp)
            template_paths.append(fpath)
        f.close()

        self.NTEMPLATE = len(self.templates)  # number of templates

        # Divide our templates into groups.
        ttypes = [t['type'] for t in self.templates]
        _, idx, self.TYPE_COUNTS = np.unique(ttypes, return_index=True,
                                             return_counts=True)
        self.TYPES = np.array(ttypes)[np.sort(idx)]
        if len(self.TYPES) == 1:  # if no types provided, all are unique
            self.TYPES = np.arange(self.NTEMPLATE).astype('str')
            self.TYPE_COUNTS = np.ones(self.NTEMPLATE)
        self.NTYPE = len(self.TYPES)
        self.TTYPE = np.array([np.arange(self.NTYPE)[t['type'] == self.TYPES]
                               for t in self.templates], dtype='int').flatten()

        # Extract templates.
        for fpath, tmp in zip(template_paths, self.templates):
            wavelength, flambda = np.loadtxt(path + fpath).T
            tmp['wavelength'] = wavelength
            tmp['frequency'] = c / (1e-10 * wavelength)
            tmp['flambda'] = flambda
            tmp['fnu'] = (wavelength * 1e-10)**2 / c * (flambda * 1e10)

        # Normalize flux densities at the pivot wavelength.
        for tmp in self.templates:
            tmp['flambda'] /= np.interp(wnorm, tmp['wavelength'],
                                        tmp['flambda'])
            tmp['fnu'] /= np.interp(wnorm, tmp['wavelength'], tmp['fnu'])

    def load_prior(self, prior):
        """
        Load the P(z, t, m) prior characterizing the mock survey. Results are
        stored internally under `pm`, `ptm`, and `pztm`.

        Parameters
        ----------
        prior : str or tuple of shape (3,), optional
            If a string provided, will initialize the `MockSurvey` using a
            preset P(z, type, mag) prior. Otherwise, if a tuple containing
            P(mag), P(type | mag), and P(z | type, mag) functions of the form
            `(p_m, p_tm, p_ztm)` is provided, those will be initialized.

        """

        try:
            self.pm, self.ptm, self.pztm = _PRIORS[prior]
        except:
            self.pm, self.ptm, self.pztm = prior

    def set_refmag(self, ref, mode='name'):
        """
        Set the reference magnitude used by the magnitude prior P(mag).
        Results are stored internally under `ref_filter`.

        Parameters
        ----------
        ref : str or int
            Either the name, index, or counter (native position in list) of
            the filter.

        mode : {`'name'`, `'index'`, `'counter'`}
            Whether to search among the provided names/indices (from the
            `filter_list` file) or the native position in the filters in the
            stored list. Default is `'name'`.

        """

        if mode not in {'name', 'index', 'counter'}:
            raise ValueError("{0} is not an allowed category.".format(mode))

        if mode == 'counter':
            self.ref_filter = ref
        else:
            sel = [fltr[mode] == ref for fltr in self.filters]
            if len(sel) == 0:
                raise ValueError("{0} does not match any {1} among the "
                                 "filters.".format(ref, mode))
            self.ref_filter = np.arange(self.NFILTER)[sel][0]

    def sample_params(self, Nobj, rstate=None, mbounds=None, zbounds=(0, 15),
                      Nm=1000, Nz=1000, pm_kwargs=None, ptm_kwargs=None,
                      pztm_kwargs=None, verbose=True):
        """
        Draw `Nobj` samples from the joint P(z, t, m) prior. Results are
        stored internally under `data`.

        Parameters
        ----------
        Nobj : int
            The number of objects to be simulated.

        mbounds : tuple of length 2, optional
            The minimum/maximum magnitude allowed. Default is `(10,
            maglim + 2.5 * np.log10(5))` where `maglim` is the 5-sigma limiting
            magnitude in the reference filter.

        rstate : `~numpy.random.RandomState`, optional
            `~numpy.random.RandomState` instance. If not given, the
             random state used to initialize the `MockSurvey` object will be
             used.

        zbounds : tuple of length 2, optional
            The minimum/maximum redshift allowed. Default is `(0, 10)`.

        Nm : int, optional
            The number of points used when interpolating the inverse cumulative
            distribution function (CDF) to sample magnitudes.
            Default is `1000`.

        Nz : int, optional
            The number of points used when interpolating the inverse cumulative
            distribution function (CDF) to sample redshifts.
            Default is `1000`.

        pm_kwargs : dict, optional
            Additional keyword arguments to be passed to :meth:`pmag`.

        ptm_kwargs : dict, optional
            Additional keyword arguments to be passed to :meth:`p_tm`.

        pztm_kwargs : dict, optional
            Additional keyword arguments to be passed to :meth:`p_ztm`.

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

        """

        if pm_kwargs is None:
            pm_kwargs = dict()
        if ptm_kwargs is None:
            ptm_kwargs = dict()
        if pztm_kwargs is None:
            pztm_kwargs = dict()
        if rstate is None:
            rstate = self.rstate

        maglim = pm_kwargs.get('maglim',
                               self.filters[self.ref_filter]['depth_mag5sig'])
        pm_kwargs['maglim'] = maglim  # get 5-sigma limiting reference mag
        if mbounds is None:
            mbounds = (10, pm_kwargs['maglim'] + 2.5 * np.log10(5))

        # Sample magnitudes.
        if verbose:
            sys.stderr.write('Sampling mags: ')
            sys.stderr.flush()
        mags = np.zeros(Nobj, dtype='float')
        mags = draw_mag(Nobj, self.pm, pmag_kwargs=pm_kwargs, rstate=rstate,
                        mbounds=mbounds, Npoints=Nm)  # sample magnitudes
        if verbose:
            sys.stderr.write('{0}/{1}'.format(Nobj, Nobj))
            sys.stderr.flush()

        # Sample types.
        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()
        types = np.zeros(Nobj, dtype='int')
        generator = draw_type_given_mag  # alias for generator
        for i, t in enumerate(generator(self.ptm, mags, self.NTYPE,
                                        ptm_kwargs=ptm_kwargs,
                                        rstate=rstate)):
            types[i] = t  # assign type draw
            if verbose:
                sys.stderr.write('\rSampling types: {0}/{1}'
                                 .format(i+1, Nobj))
                sys.stderr.flush()

        # Re-label templates by type and construct probability vectors.
        tmp_types = np.array([tmp['type'] for tmp in self.templates])
        tmp_p = [np.array(t == tmp_types, dtype='float') / sum(t == tmp_types)
                 for t in self.TYPES]

        # Sample templates from types.
        templates = np.empty(Nobj, dtype='int')
        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()
        for i in range(self.NTYPE):
            n = int(sum(types == i))  # number of objects of a given type
            templates[types == i] = rstate.choice(self.NTEMPLATE, size=n,
                                                  p=tmp_p[i])
            if verbose:
                sys.stderr.write('\rSampling templates within each type: '
                                 '{0}/{1}'.format(i+1, self.NTYPE))
                sys.stderr.flush()

        # Sample redshifts.
        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()
        redshifts = np.zeros(Nobj, dtype='float')
        generator = draw_redshift_given_type_mag
        for i, z in enumerate(generator(self.pztm, types, mags,
                                        pztm_kwargs=pztm_kwargs,
                                        zbounds=zbounds, Npoints=Nz,
                                        rstate=rstate)):
            redshifts[i] = z  # assign redshift draw
            if verbose:
                sys.stderr.write('\rSampling redshifts: {0}/{1}'
                                 .format(i+1, Nobj))
                sys.stderr.flush()
        if verbose:
            sys.stderr.write('\n')
            sys.stderr.flush()

        # Save data.
        self.data = {'refmags': mags, 'types': types,
                     'templates': templates, 'redshifts': redshifts}
        self.NOBJ = Nobj

    def sample_phot(self, red_fn='madau+99', rnoise_fn=None, rstate=None,
                    verbose=True):
        """
        Generate noisy photometry from `(t, z, m)` samples. **Note that this
        ignores Poisson noise**. Results are added internally to `data`.

        Parameters
        ----------
        red_fn : function, optional
            A function that adds in reddening from the intergalactic medium
            (IGM). Default is `'madau+99'`, which uses the parametric form
            from Madau et al. (1999). If `None` is passed, no reddening will
            be applied.

        rnoise_fn : function, optional
            A function that takes the average noise (computed from the
            provided survey depths) and jitters them to mimic spatial
            background variation.

        rstate : `~numpy.random.RandomState`, optional
            `~numpy.random.RandomState` instance. If not given, the
             random state used to initialize the `MockSurvey` object will be
             used.

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

        """

        if rstate is None:
            rstate = self.rstate

        # Grab data.
        try:
            mags = self.data['refmags']
            types = self.data['types']
            templates = self.data['templates']
            redshifts = self.data['redshifts']
        except:
            raise ValueError("No mock data has been generated.")

        # Extract reddening function.
        try:
            red_fn = _IGM[red_fn]
        except:
            pass

        # Initialize useful quantities.
        tlw = [np.log(t['wavelength']) for t in self.templates]  # ln(tmp wave)
        flw = [np.log(f['wavelength']) for f in self.filters]  # ln(flt wave)
        filt_nu = [f['frequency'] for f in self.filters]  # filt nu
        filt_t = [f['transmission'] for f in self.filters]  # filt nu
        norm = [np.trapz(ft / fn, fn)
                for ft, fn in zip(filt_t, filt_nu)]  # filter normalization
        tfnu = [t['fnu'] for t in self.templates]

        # Compute unnormalized photometry.

        phot = np.zeros((self.NOBJ, self.NFILTER))  # photometry array
        for i, (t, z) in enumerate(zip(templates, redshifts)):
            # Compute reddening.
            if red_fn is not None:
                igm_teff = [red_fn(np.exp(f_lw), z) for f_lw in flw]
            else:
                igm_teff = [np.ones_like(f_lw) for f_lw in flw]

            # Integrate the flux over the filter. Interpolation is performed
            # using the arcsinh transform for improved numerical stability.
            phot[i] = [np.trapz(np.sinh(np.interp(f_lw, tlw[t] + np.log(1 + z),
                                                  np.arcsinh(tfnu[t]))) *
                                f_t / f_nu * te, f_nu) / f_n
                       for f_t, f_nu, f_lw, f_n, te in zip(filt_t, filt_nu,
                                                           flw, norm,
                                                           igm_teff)]
            if verbose:
                sys.stderr.write('\rGenerating photometry: {0}/{1}'
                                 .format(i+1, self.NOBJ))
                sys.stderr.flush()

        # Normalize photometry to reference magnitude.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            fluxes = 10**((mags - 23.9) / -2.5)  # convert to flux
            phot /= phot[:, self.ref_filter][:, None]  # norm to ref_flux=1.
            phot *= fluxes[:, None]  # multiply by actual ref_flux value

        # Deal with "bad" (nonsensical) photometry.
        sel_badphot = np.unique(np.nonzero(~np.isfinite(phot))[0])
        self.data['refmags'][sel_badphot] = np.inf  # fix magnitudes
        phot[sel_badphot] = -np.inf  # fix fluxes

        # Compute errors.
        if verbose:
            sys.stderr.write('\nSampling errors: ')
            sys.stderr.flush()
        fnoise = np.array([np.ones(self.NOBJ) * f['depth_flux1sig']
                           for f in self.filters]).T
        if rnoise_fn is not None:
            fnoise = rnoise_fn(fnoise, rstate=rstate)  # add noise variability
        if verbose:
            sys.stderr.write('{0}/{1}'.format(self.NOBJ, self.NOBJ))
            sys.stderr.flush()

        # Jittering fluxes.
        if verbose:
            sys.stderr.write('\nSampling photometry: ')
            sys.stderr.flush()
        phot_obs = rstate.normal(phot, fnoise)
        if verbose:
            sys.stderr.write('{0}/{1}\n'.format(self.NOBJ, self.NOBJ))
            sys.stderr.flush()

        # Save results.
        self.data['phot_true'] = phot
        self.data['phot_obs'] = phot_obs
        self.data['phot_err'] = fnoise

    def make_mock(self, Nobj, mbounds=None, zbounds=(0, 15),
                  Nm=1000, Nz=1000, pm_kwargs=None, ptm_kwargs=None,
                  pztm_kwargs=None, red_fn='madau+99', rnoise_fn=None,
                  rstate=None, verbose=True):
        """

        Generate (noisy) photometry for `Nobj` objects sampled from the
        prior. Wraps :meth:`sample_params` and :meth:`sample_phot`. Results are
        stored internally under `data`.

        Parameters
        ----------
        Nobj : int
            The number of objects to be simulated.

        mbounds : tuple of length 2, optional
            The minimum/maximum magnitude allowed. Default is `(10,
            maglim + 2.5 * np.log10(5))` where `maglim` is the 5-sigma limiting
            magnitude in the reference filter.

        zbounds : tuple of length 2, optional
            The minimum/maximum redshift allowed. Default is `(0, 10)`.

        Nm : int, optional
            The number of points used when interpolating the inverse cumulative
            distribution function (CDF) to sample magnitudes.
            Default is `1000`.

        Nz : int, optional
            The number of points used when interpolating the inverse cumulative
            distribution function (CDF) to sample redshifts.
            Default is `1000`.

        pm_kwargs : dict, optional
            Additional keyword arguments to be passed to :meth:`pmag`.


        ptm_kwargs : dict, optional
            Additional keyword arguments to be passed to :meth:`p_tm`.

        pztm_kwargs : dict, optional
            Additional keyword arguments to be passed to :meth:`p_ztm`.

        red_fn : function, optional
            A function that adds in reddening from the intergalactic medium
            (IGM). Default is `'madau+99'`, which uses the parametric form
            from Madau et al. (1999). If `None` is passed, no reddening will
            be applied.

        rnoise_fn : function, optional
            A function that takes the average noise (computed from the
            provided survey depths) and jitters them to mimic spatial
            background variation.

        rstate : `~numpy.random.RandomState`, optional
            `~numpy.random.RandomState` instance. If not given, the
             random state used to initialize the `MockSurvey` object will be
             used.

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

        """

        # Sample parameters.
        self.sample_params(Nobj, mbounds=mbounds, zbounds=zbounds,
                           Nm=Nm, Nz=Nz, pm_kwargs=pm_kwargs, rstate=rstate,
                           ptm_kwargs=ptm_kwargs, pztm_kwargs=pztm_kwargs,
                           verbose=verbose)

        # Sample photometry.
        self.sample_phot(red_fn=red_fn, rnoise_fn=rnoise_fn,
                         rstate=rstate, verbose=verbose)

    def make_model_grid(self, redshifts, red_fn='madau+99', verbose=True):
        """
        Generate photometry for input set of templates over the input
        `redshifts` grid. Results are stored internally under `models` as an
        `(Nz, Nt, Nf)` `~numpy.ndarray` with `Nz` redshifts, `Nt` templates,
        and `Nf` filters.

        Parameters
        ----------
        redshifts : iterable of shape (N,)
            Input redshift grid.

        red_fn : function, optional
            A function that adds in reddening from the intergalactic medium
            (IGM). Default is `'madau+99'`, which uses the parametric form
            from Madau et al. (1999). If `None` is passed, no reddening will
            be applied.

        verbose : bool, optional
            Whether to print progress to `~sys.stderr`. Default is `True`.

        """

        Nz = len(redshifts)

        # Extract reddening function.
        try:
            red_fn = _IGM[red_fn]
        except:
            pass

        # Initialize useful quantities.
        tlw = [np.log(t['wavelength']) for t in self.templates]  # ln(tmp wave)
        flw = [np.log(f['wavelength']) for f in self.filters]  # ln(flt wave)
        filt_nu = [f['frequency'] for f in self.filters]  # filt nu
        filt_t = [f['transmission'] for f in self.filters]  # filt nu
        norm = [np.trapz(ft / fn, fn)
                for ft, fn in zip(filt_t, filt_nu)]  # filter normalization
        tfnu = [t['fnu'] for t in self.templates]

        # Compute unnormalized photometry.
        phot = np.zeros((Nz, self.NTEMPLATE, self.NFILTER))
        for i, z in enumerate(redshifts):
            for j in range(self.NTEMPLATE):
                # Compute reddening.
                if red_fn is not None:
                    igm_teff = [red_fn(np.exp(f_lw), z) for f_lw in flw]
                else:
                    igm_teff = [np.ones_like(f_lw) for f_lw in flw]

                # Integrate the flux over the filter. Interpolation is done
                # using the arcsinh transform for improved numerical stability.
                phot[i, j] = [np.trapz(f_t / f_nu * te *
                                       np.sinh(np.interp(f_lw, tlw[j] +
                                                         np.log(1 + z),
                                                         np.arcsinh(tfnu[j]))),
                                       f_nu) / f_n
                              for f_t, f_nu, f_lw, f_n, te in zip(filt_t,
                                                                  filt_nu,
                                                                  flw, norm,
                                                                  igm_teff)]
            if verbose:
                sys.stderr.write('\rGenerating model photometry grid: {0}/{1}'
                                 .format(i+1, len(redshifts)))
                sys.stderr.flush()

        # Save results.
        self.models = {'data': phot, 'zgrid': redshifts}
