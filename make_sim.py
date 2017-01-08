##############################
########## MAKE_SIM ##########
##############################

# A collection of simple functions used to simulate a photometric survey.
# Current version (v1) by Josh Speagle (Harvard University; jspeagle@cfa.harvard.edu)




########## SETUP ##########

# general environment
import numpy as np
import matplotlib
import scipy
from numpy import *
from numpy.random import *
from numpy.random import choice
from matplotlib import *
from matplotlib.pyplot import *
from scipy import *

# general functions
import sys # print statements
from astropy.io import fits # I/O on fits
import os # used to check for files
from scipy import interpolate # interpolation

# statistics
from scipy import stats
from scipy import random

# additional memory management
import gc

# confidence intervals
SIG1 = 68.2689492/100.
SIG2 = 95.4499736/100.
SIG3 = 99.7300204/100.






########## MAGNITUDE FUNCTIONS ##########

def draw_mag(Nobj, pmag, mlim, mbounds=[16,28], Npoints=1000):
    """
    Draw magnitudes from P(m).

    Keyword arguments:
    Nobj -- number of objects
    pmag -- P(mag)
    mlim -- 5-sigma magnitude limit
    mbounds -- magnitude bounds (low, high)
    Npoints -- number of points used in grid (default=1000)
    """

    mgrid = linspace(mbounds[0], mbounds[1], Npoints) # compute magnitude grid
    pdf_m = pmag(mgrid, mlim) # compute P(m) over grid
    cdf_m = append(0, pdf_m.cumsum()) / pdf_m.sum() # compute CDF (left pad 0)
    mgrid = append(mgrid[0]-1e-2, mgrid) # left pad edge
    draws = interp(rand(Nobj), cdf_m, mgrid) # sample from inverse CDF

    return draws
    

def draw_type_mag(p_t_m, mags, Ntypes):
    """
    Draw types from P(t|m). 

    Keyword arguments:
    p_t_m -- our conditional P(t|m) prior
    mags -- series of magnitudes
    Ntypes -- number of galaxy types
    """

    t = arange(Ntypes)
    
    draws = dot(array([multinomial(1, p_t_m(t,m)) for m in mags]), t) # draw type

    return draws


def draw_redshift_type_mag(p_tz_m, types, mags, zbounds=[0,15], Npoints=1000):
    """
    Draw redshifts from P(z|t,m).

    Keyword arguments:
    p_tz_m -- our conditional prior P(z,t|m) (which we evaluate at fixed t)
    types -- series of types
    mags -- series of magnitudes
    """

    zgrid = linspace(zbounds[0], zbounds[1], Npoints) # compute redshift grid
    zgrid2 = append(zgrid[0]-1e-5, zgrid) # left-padded grid
    
    Nobj = len(types) # number of draws
    draws = empty(Nobj) # redshift draws
    for i in xrange(Nobj):
        t, m = types[i], mags[i]
        pdf_z = p_tz_m(t,zgrid,m) # PDF
        cdf_z = append(0, pdf_z.cumsum()) / pdf_z.sum() # CDF (left padded)
        draws[i] = interp(rand(), cdf_z, zgrid2) # sample from inverse CDF

    return draws


# imag error distribution as function of mag limit, as in Rykoff et al. (2015)
def mag_err(m, mlim):
    a,b=4.56,1
    k=1
    sigmadet=5
    teff=exp(a+b*(mlim-21.))
    F=10**(-0.4*(m-22.5))
    Flim=10**(-0.4*(mlim-22.5))
    Fnoise=(Flim/sigmadet)**2*k*teff-Flim
    return 2.5/log(10)*sqrt((1+Fnoise/F)/(F*k*teff))









########## SURVEY ##########

class survey():
    """
    Our class containing the relevant information characterizing our survey.
    """

    def LoadFilters(self, filter_list, mag_ref, path='', Npoints=5e4):
        """
        Load an input filter list.

        Keyword arguments:
        filter_list -- file list
        mag_ref -- reference filter magnitude
        path -- home path
        Npoints -- number of points used to interpolate filter transmission curves
        """

        c = 299792458.0 # speed of light in m/s

        self.mag_ref = mag_ref # reference magnitude

        # load filter names and files
        f = open(filter_list)
        self.filters = []
        self.fnames = []
        for line in f:
            lsplit = line.split()
            self.filters.append(lsplit[0]) # filter names
            self.fnames.append(lsplit[1]) # file names
        f.close()

        self.NFILTER = len(self.filters) # number of filters
        
        self.fw = [0.]*self.NFILTER # wavelength [A]
        self.fn = [0.]*self.NFILTER # frequency [Hz]
        self.ft = [0.]*self.NFILTER # transmission

        # extract filters
        for i in xrange(self.NFILTER):
            self.fw[i], self.ft[i] = swapaxes(loadtxt(path+self.fnames[i]), 0, 1)
            self.fn[i] = c / (self.fw[i]*1e-10)

        # compute effective wavelengths
        self.lambda_eff = zeros(self.NFILTER) # initialize
        for i in xrange(self.NFILTER):
            nuMax = 0.999*c / (min(self.fw[i])*1e-10) # max frequency
            nuMin = 1.001*c / (max(self.fw[i])*1e-10) # min frequency
            nu = linspace(nuMin, nuMax, Npoints) # frequency array
            lnu = log(nu) # ln(frequency)
            wave = c/nu # convert to lambda
            lwave = log(wave) # ln(wavelength)
            temp = interp(1e10*wave, self.fw[i], self.ft[i]) # interpolated transmission [A]
            self.lambda_eff[i] = 1e10 * exp( trapz(temp*lwave,lnu) / trapz(temp,lnu) ) # effective wavelength

            
    def LoadDepth(self, depth_list, depths):
        """
        Load a set of depths corresponding to the observed filters.

        Keyword arguments:
        depth_list -- list of depths (in AB magnitudes)
        depths -- number of sigma used to define depths
        """

        self.MDEPTHS = depth_list + 2.5*log10(depths/5.) # 5-sigma depths [AB mag]
        self.FDEPTHS = 10**((depth_list-23.9) / -2.5) / depths # 1-sigma noise level [uJy]


    def LoadTemplates(self, template_list, path='', lnorm=7e3):
        """
        Load an input template list.

        Keyword arguments:
        template_list -- name of template list file
        path -- home path
        lnorm -- wavelength where F_nu is normalized
        """

        # load template list
        self.tidx, self.ttype, self.tfile = [], [], []
        f = open(template_list)
        for line in f:
            lsplit = line.split()
            self.tidx.append(lsplit[0]) # template index
            self.ttype.append(lsplit[1]) # template type
            self.tfile.append(lsplit[2]) # template filename
        f.close()
        self.templates=[i.replace('.sed','') for i in self.tfile] # template names

        self.NTEMPLATES = len(self.templates) # number of templates
        self.TYPES, self.NTYPES = unique(self.ttype, return_counts=True) # types and number per type
        self.types = array([arange(len(self.TYPES))[self.ttype[i] == self.TYPES][0] for i in xrange(self.NTEMPLATES)], dtype='int') # convert to ints

        # extract templates
        self.tw, self.tflambda, self.tfnu = [], [], []
        for i in self.tfile:
            data = loadtxt('seds/'+i)
            self.tw.append(data[:,0]) # wavelength [A]
            self.tflambda.append(data[:,1]) # F_lambda [per unit wavelength]
            self.tfnu.append(data[:,1]*data[:,0]**2) # F_nu [per unit frequency]

        # normalize F_nu at lnorm
        for i in xrange(self.NTEMPLATES):
            self.tfnu[i] /= interp(lnorm, self.tw[i], self.tfnu[i])

    def LoadPrior(self, p_tz_m, p_t_m, p_m):
        """
        Load input priors. This must be a function of galaxy type, redshift, and magnitude. 

        Keyword arguments:
        p_tz_m -- P(t,z|m)
        p_t_m -- P(t|m)
        p_m -- P(m)
        mag_ref -- Reference magnitude
        """

        self.prior_tz_m = p_tz_m # P(t,z|m)
        self.prior_t_m = p_t_m # P(t|m)
        self.prior_p_m = p_m # P(m)

        
    def SamplePrior(self, Ndraws, mbounds=[16,28], Nm=1000, zbounds=[0,15], Nz=1000):
        """
        Draw Ndraws samples from the joint P(t,z,m) prior.

        Keyword arguments:
        Ndraws -- number of samples
        mbounds -- magnitude bounds (default=16-28)
        Nm -- number of magnitude grid points (default=1000)
        zbounds -- redshift bounds (default=0-15)
        Nz -- number of redshift grid points (default=1000)
        """

        sys.stdout.write('Sampling mags...')
        mags = draw_mag(Ndraws, self.prior_p_m, self.MDEPTHS[self.mag_ref], mbounds, Nm)
        sys.stdout.write('done!\n')
        
        sys.stdout.write('Sampling types...')
        types = draw_type_mag(self.prior_t_m, mags, len(self.NTYPES))
        sys.stdout.write('done!\n')
        
        sys.stdout.write('Sampling redshifts...')
        redshifts = draw_redshift_type_mag(self.prior_tz_m, types, mags, zbounds, Nz)
        sys.stdout.write('done!\n')

        self.samples = array([types, redshifts, mags])
        self.NSAMPLES = Ndraws
        
        return self.samples


    def SamplePhot(self, bg_lognorm_std=None):
        """
        Generate noisy photometry from (t,z,m) samples.

        Keyword arguments:
        bg_lognorm_std -- fractional variation in background noise (default=None)
        """

        types, redshifts, mags = self.samples

        # convert from types to templates
        templates = empty(self.NSAMPLES, dtype='int')
        counts = zeros(self.NTEMPLATES)
        for i in unique(self.types): # for each type
            n = int((types==i).sum()) # number of draws
            p = 1.*(self.types==i) / self.NTYPES[i] # corresponding multinomial probability vector
            templates[types==i] = choice(self.NTEMPLATES, size=n, p=p)
        self.samples_t = templates

        # initialize useful quantities
        tlw=[log(self.tw[i]) for i in xrange(self.NTEMPLATES)] # log(template wavelength)
        flw=[log(self.fw[i]) for i in xrange(self.NFILTER)] # log(filter wavelength)
        norm=[trapz(self.ft[i]/self.fn[i],self.fn[i]) for i in xrange(self.NFILTER)] # filter normalization

        # compute unnormalized photometry
        self.phot=empty((self.NSAMPLES,self.NFILTER)) # initialize photometry array
        for i in xrange(self.NSAMPLES):
            t, z = templates[i], redshifts[i] # type, redshift
            self.phot[i] = [trapz(sinh(interp(flw[j], tlw[t]+log(1+z), arcsinh(self.tfnu[t])))
                                  * self.ft[j] / self.fn[j], self.fn[j])
                            / norm[j] for j in xrange(self.NFILTER)] # integrated flux over filter

        # normalize photometry to reference magnitude
        fluxes = 10**((mags-23.9) / -2.5) # compute reference mag fluxes
        self.phot /= self.phot[:,self.mag_ref][:,None] # normalize to reference mag
        self.phot *= fluxes[:,None]

        # compute errors
        self.noise = array([self.FDEPTHS for i in xrange(self.NSAMPLES)])
        if bg_lognorm_std is not None:
            self.noise *= lognormal(0, log(bg_lognorm_std), size=(self.NSAMPLES, self.NFILTER))

        # jittering fluxes
        self.phot_obs = normal(self.phot, self.noise)
            
        return self.phot, self.noise, self.phot_obs


    def GenModelGrid(self, templates, redshifts):
        """
        Generate a grid of models over the corresponding templates and redshifts.
        """

        Nt, Nz = len(templates), len(redshifts)

        # initialize useful quantities
        tlw=[log(self.tw[i]) for i in xrange(self.NTEMPLATES)] # log(template wavelength)
        flw=[log(self.fw[i]) for i in xrange(self.NFILTER)] # log(filter wavelength)
        norm=[trapz(self.ft[i]/self.fn[i],self.fn[i]) for i in xrange(self.NFILTER)] # filter normalization

        # compute unnormalized photometry
        self.models_t, self.models_z = templates, redshifts
        self.models_phot = empty((Nt, Nz, self.NFILTER)) # initialize photometry array
        for tidx in xrange(Nt):
            for zidx in xrange(Nz):
                t, z = templates[tidx], redshifts[zidx]
                # integrate flux over filter
                self.models_phot[tidx][zidx] = [trapz(sinh(interp(flw[j], tlw[t]+log(1+z), arcsinh(self.tfnu[t])))
                                                      * self.ft[j] / self.fn[j], self.fn[j])
                                                / norm[j] for j in xrange(self.NFILTER)]
                
        return self.models_phot
    
