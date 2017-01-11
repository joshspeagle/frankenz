###############################
########## FRANKEN-Z ##########
###############################

# Flexible Regression over Associated Neighbors with Kernel dEnsity estimatioN for Redshift (Z)
# Authors: Josh Speagle (Harvard University; jspeagle@cfa.harvard.edu).
# Released under MIT License.
# Please see Speagle et al. (2017) arxiv:XXXX.XXXXX for more details.


####################################################################################################


########## MODULES ##########

# initializing Pylab environment
import numpy as np
import matplotlib as mpl
import scipy
from numpy import *
from matplotlib import *
from matplotlib.pyplot import *
from scipy import *

# general functions
import pandas # uniqueness checks
import sys # outputs
import os # used to check for files
from astropy.io import fits # I/O on fits
from sklearn.externals import joblib # I/O on ML models
from scipy import interpolate # interpolation

# machine learning
from sklearn import neighbors # nearest neighbors
from sklearn import base # additional methods

# statistics
from scipy import stats
from scipy import random

# random number generation
from numpy.random import *
from numpy.random import choice

# manual memory management
import gc

# confidence intervals
SIG1 = 68.2689492/100.
SIG2 = 95.4499736/100.
SIG3 = 99.7300204/100.

# useful constants
c = 299792458.0 # speed of light in m/s


########## PLOTTING DEFAULTS ##########

# declaring plotting stuff
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
rcParams.update({'xtick.major.pad': '7.0'})
rcParams.update({'xtick.major.size': '7.5'})
rcParams.update({'xtick.major.width': '1.5'})
rcParams.update({'xtick.minor.pad': '7.0'})
rcParams.update({'xtick.minor.size': '3.5'})
rcParams.update({'xtick.minor.width': '1.0'})
rcParams.update({'ytick.major.pad': '7.0'})
rcParams.update({'ytick.major.size': '7.5'})
rcParams.update({'ytick.major.width': '1.5'})
rcParams.update({'ytick.minor.pad': '7.0'})
rcParams.update({'ytick.minor.size': '3.5'})
rcParams.update({'ytick.minor.width': '1.0'})
rcParams.update({'xtick.color': 'k'})
rcParams.update({'ytick.color': 'k'})
rcParams.update({'font.size': 30})


####################################################################################################


########## LIKELIHOODS ##########


def loglikelihood(data, data_err, data_mask, models, models_err, models_mask):
    """
    Compute -2lnL w/ FIXED scaling using a set of models W/ ERRORS.

    Keyword arguments:
    data -- input values
    data_err -- input errors
    data_mask -- mask for missing input data
    models -- collection of comparison models
    models_err -- model errors
    models_mask -- mask for missing model data

    Outputs:
    chi2_mod -- -2lnL for each model
    Ndim -- number of observations used in fit
    """

    tot_var = square(data_err) + square(models_err) # combined variance
    tot_mask = data_mask * models_mask # combined binary mask
    Ndim = tot_mask.sum(axis=1) # number of bands

    # compute ln(likelihood)    
    resid = data - models # residuals
    chi2 = (tot_mask * square(resid) / tot_var).sum(axis=1) # compute standard chi2
    chi2_mod = chi2 - Ndim # normalize by E[chi2(N)]
    
    return chi2_mod, Ndim


def loglikelihood_s(data, data_err, data_mask, models, models_err, models_mask, return_s=False):
    """
    Compute -2lnL W/ FREE scaling using a set of models W/O ERRORS.

    Keyword arguments:
    data -- input values
    data_err -- input errors
    data_mask -- mask for missing input data
    models -- collection of comparison models
    models_err -- model errors
    models_mask -- mask for missing model data
    return_s -- return the maximum-likelihood scalefactor (default=False)

    Outputs:
    chi2_mod -- -2lnL for each model
    Ndim -- number of observations used in fit
    scale_vals -- maximum-likelihood model scalefactor
    """

    tot_mask = data_mask * models_mask # combined binary mask
    data_var = square(data_err) # data variance
    Ndim = tot_mask.sum(axis=1) # number of bands
    
    # derive scalefactors between data and models
    inter_vals = (tot_mask * models * data[None,:] / data_var[None,:]).sum(axis=1) # interaction term
    shape_vals = (tot_mask * square(models) / data_var[None,:]).sum(axis=1) # model-dependent term (i.e. quadratic 'steepness' of chi2)
    scale_vals = inter_vals / shape_vals # maximum-likelihood scalefactors

    # compute ln(likelihood)
    resid = data - scale_vals[:,None]*models # compute scaled residuals
    
    chi2 = (tot_mask * square(resid) / data_var[None,:]).sum(axis=1) # compute chi2
    chi2_mod = chi2 - (Ndim-1) # normalize by E[chi2(N-1)]

    if return_s:
        return chi2_mod, Ndim, scale_vals
    else:
        return chi2_mod, Ndim



########## MAGNITUDE MAPS ##########


def asinh_mag_map(phot, err, skynoise=None, zeropoints=1.):
    """
    Map flux density to asinh magnitude (i.e. "Luptitude"; Lupton et al. 1999).

    Keyword arguments:
    phot -- flux densities
    err -- associated errors
    skynoise -- background sky noise (i.e. softening parameter) (default=median(err))
    zeropoints -- flux zero-points (default=1.)
    
    Outputs:
    asinh_mag -- asinh magnitudes
    asinh_mag_err -- associated transformed errors
    [skynoise] -- softening parameter (if not originally provided)
    """

    if skynoise is None:
        skynoise = median(err, axis=0) # assign softening as median error

    # compute Luptitudes 
    mag_asinh = -2.5/log(10) * ( arcsinh(phot/(2*skynoise)) + log(skynoise/zeropoints) ) # mag
    mag_asinh_err = sqrt( square(2.5*log10(e)*err) / (square(2*skynoise) + square(phot)) ) # err

    return mag_asinh, mag_asinh_err


def inv_asinh_mag_map(mag, magerr, skynoise, zeropoints=1.):
    """
    Map asinh magnitude to flux density.

    Keyword arguments:
    mag -- asinh magnitudes
    magerr -- associated errors
    skynoise -- background sky noise (i.e. softening parameter)
    zeropoints -- flux zero-points (default=1.)

    Outputs: 
    phot -- flux densities
    err -- associated transformed errors
    """

    # compute flux densities
    phot = sinh( log(10)/-2.5 * mag - log(skynoise/zeropoint) ) * (2*skynoise)
    err = sqrt( square(magerr) * (square(2*skynoise) + square(phot)) ) / (2.5*log10(e))

    return phot, err



########## QUANTIFYING PERFORMANCE ##########


def compute_score(y, pred, weights=None, cat_thresh=0.15):
    """
    Compute quality metrics (scores) between two sets of predictions.

    Keyword arguments:
    y -- original targets
    pred -- associated predictions
    weights -- weights for computing modified statistics (default=None)
    cat_thresh -- threshold used for computing catastrophic outliers (|pred-y|/|y| > cat_thresh) (default=0.15)
    
    Outputs:
    mean scores -- [mean(dy), std(dy), mean(dy/y), std(dy/y)]
    median scores -- [median(dy), MAD(dy; 68%), median(dy/y), MAD(dy/y; 68%)]
    f_cat -- catastrophic outlier fraction
    """

    Ny = len(y) # total number of objects
                  
    # initialize weights
    if weights is None:
        weights = ones(Ny)

    Nobj = weights.sum() # effective number of objects
    sig1 = int(SIG1*Nobj) # defining 1-sigma boundary
    dy, dyp = (pred-y), (pred-y)/y # defining offsets and fractional offsets

    # mean stats
    mean_dy, mean_dyp = average(dy, weights=weights), average(dyp, weights=weights) # mean
    std_dy, std_dyp = sqrt( average(square(dy-mean_dy), weights=weights) ), sqrt( average(square(dyp-mean_dyp), weights=weights) ) # standard deviation
    mean_stat = array([mean_dy, std_dy, mean_dyp, std_dyp]) # bundle scores
    
    # median stats
    s1, s2 = argsort(dy), argsort(dyp) # sorted list of deviations (low->high)
    cdf1, cdf2 = cumsum(weights[s1]), cumsum(weights[s2]) # CDF of sorted weights
    med_dy, med_dyp = dy[s1][argmin(abs(cdf1-0.5*Nobj))], dy[s2][argmin(abs(cdf2-0.5*Nobj))] # median
    
    mad1, mad2 = abs(dy-med_dy), abs(dyp-med_dyp) # median absolute deviations (MADs)
    s1, s2 = argsort(mad1), argsort(mad2) # sorted list of MADs
    cdf1, cdf2 = cumsum(weights[s1]), cumsum(weights[s2]) # CDF of sorted weights
    mad_dy, mad_dyp = mad1[s1][argmin(abs(cdf1-SIG1*Nobj))], mad2[s2][argmin(abs(cdf2-SIG1*Nobj))] # 68% (1-sigma) interval
    med_stat = array([med_dy, mad_dy, med_dyp, mad_dyp]) # bundle scores
    
    # catastrophic outliers
    f_cat = 1.*(weights[abs(dyp)>cat_thresh]).sum() / Nobj
    
    return mean_stat, med_stat, f_cat



def compute_density_score(dist_base, dist_test, dist_test_err=None):
    """
    Compute quality metrics (scores) between two sets of matched densities.

    Keyword arguments:
    dist_base -- the baseline distribution
    dist_test -- the distribution to be tested
    dist_test_err -- associated errors of dist_test

    Outputs:
    Pois_statistic, AD_statistic, [err_statistic]
    Pois_statistic -- Poisson test statistics [N_pois, arr_pois, a_sub, p_pois], where:
        N_pois -- total Poisson deviation
        arr_pois -- running Poisson deviation
        a_sub -- sub-selected array results have been computed over
        p_pois -- p-value (Poisson)
    AD_statistic -- the Anderson-Darling 2-sample test statistics [ad_stat, ad_crit, p_ad], where:
        ad_stat -- AD test statistic
        ad_crit -- critical values (25%, 10%, 5%, 2.5%, 1%)
        p_ad -- approximate p-value
    err_statistic -- error-weighted test statistics (same format as Pois_statistic)
    """

    # flatten arrays to 1-D
    a, b = dist_base.flatten(), dist_test.flatten()
    if dist_test_err is not None: be = dist_test_err.flatten()

    N, Na, Nb = len(a), 1.*a.sum(), 1.*b.sum() # number and normalizations
    a_sub = a > 0 # find all strictly positive terms
    dof = sum(a_sub) # degrees of freedom
    dev = b - a # absolute deviation
    arr_poisson = (dev/sqrt(a))[a_sub] # compute normalized Poisson fluctuation
    N_poisson = sum(square(arr_poisson)) # total fluctuation
    p_poisson = 1 - stats.chi2.cdf(N_poisson, dof) # p-value

    # drawing samples
    cdf_a, cdf_b = linspace(0,1,N+1), linspace(0,1,N+1) # initializing CDF
    cdf_a[1:], cdf_b[1:] = a.cumsum()/Na, b.cumsum()/Nb # computing CDF
    a_samp = interp(rand(int(Na)), cdf_a, arange(N+1))
    b_samp = interp(rand(int(Nb)), cdf_b, arange(N+1))

    try:
        N_ad = stats.anderson_ksamp((a_samp, b_samp)) # compute AD 2-sample test score
    except OverflowError:
        N_ad = (NaN,NaN,0.0) # assign NaNs if this fails

    if dist_test_err is None:
        return [N_poisson, arr_poisson, a_sub, p_poisson], N_ad
    else:
        arr_err = (dev/dist_test_err)[a_sub] # standardized fluctuation (sqrt(N)->err)
        N_err = sum(square(arr_err)) # total fluctuation
        p_err = 1 - stats.chi2.cdf(N_err, dof) # p-value
        return [N_poisson, arr_poisson, a_sub, p_poisson], N_ad, [N_err, arr_err, a_sub, p_err]



########## PDF FUNCTIONS ##########


def pdfs_resample(target_grid, target_pdfs, new_grid):
    """
    Resample input PDFs from a given grid onto a new grid.

    Keyword arguments:
    target_grid -- original grid
    target_pdfs -- original collection of PDFs
    new_grid -- new grid

    Outputs:
    new_PDFs -- resampled PDFs
    """
    
    sys.stdout.write("Resampling PDFs...")
    
    Nobj, Npoints = len(target_pdfs), len(new_grid) # size of inputs
    new_pdfs = empty((Nobj, Npoints), dtype='float32') # initialize array
    
    for i in xrange(Nobj):
        if i%5000 == 0: # print status every 5000 objects
            sys.stdout.write(str(i)+' ')
            sys.stdout.flush()
        new_pdfs[i] = interp(new_grid, target_grid, target_pdfs[i]) # interpolate PDF
        new_pdfs[i] /= sum(new_pdfs[i]) # re-normalize
        
    sys.stdout.write("done!\n")

    return new_pdfs


def pdfs_summary_statistics(target_grid, target_pdfs):
    """
    Compute summary statistics from input PDFs.

    Keyword arguments:
    target_grid -- input grid
    target_pdfs -- input collection of PDFs
    conf_width -- redshift range used to establish the 'zConf' flag (see Carrasco Kind & Brunner 2013) 
    deg_spline -- order of spline fit 

    Outputs:
    pdf_mean -- mean (first moment)
    pdf_med -- median (50th percentile)
    pdf_mode -- peak (mode)
    pdf_mc -- Monte Carlo estimate (random)
    pdf_l68 -- lower 68th percentile of CDF
    pdf_h68 -- higher 68th percentile of CDP
    pdf_l95 -- lower 95th percentile of CDF
    pdf_h95 -- higher 95th percentile of CDF
    pdf_std -- standard deviation (sqrt of normalized second moment)
    """

    # initialize variables
    Ntest = len(target_pdfs) # number of input pdfs
    pdf_mean = zeros(Ntest, dtype='float32') # mean (first moment)
    pdf_std = zeros(Ntest, dtype='float32') # standard deviation (sqrt of normalized second moment)
    pdf_mode = zeros(Ntest, dtype='float32') # peak (mode)
    pdf_mc = zeros(Ntest, dtype='float32') # Monte Carlo (random)
    pdf_med = zeros(Ntest, dtype='float32') # median
    pdf_l95 = zeros(Ntest, dtype='float32') # lower 95% confidence interval
    pdf_l68 = zeros(Ntest, dtype='float32') # lower 68% confidence interval
    pdf_h68 = zeros(Ntest, dtype='float32') # upper 68% confidence interval
    pdf_h95 = zeros(Ntest, dtype='float32') # upper 95% confidence interval


    # confidence intervals
    i1 = 0.68 # interval 1
    i2 = 0.95 # interval 2
    m = 0.5 # median
    l2 = m-i2/2 # lower 2
    l1 = m-i1/2 # lower 1
    u1 = m+i1/2 # upper 1
    u2 = m+i2/2 # upper 2

    
    sys.stdout.write("Computing PDF quantities...")
    
    for i in xrange(Ntest):
        if i%5000 == 0: # print status every 5000 objects
            sys.stdout.write(str(i)+' ')
            sys.stdout.flush()
        
        # mean quantities
        pdf_mean[i] = dot(target_pdfs[i], target_grid)
        pdf_std[i] = sqrt( dot(target_pdfs[i], square(target_grid))-square(pdf_mean[i]) )
        
        # mode quantities
        pdf_mode[i] = target_grid[argmax(target_pdfs[i])]
        
        # cumulative distribution function
        cdf = target_pdfs[i].cumsum() # original CDF (normalized to 1)
        pdf_med[i], pdf_mc[i], pdf_h68[i], pdf_l68[i], pdf_h95[i], pdf_l95[i] = interp([m, rand(), u1, l1, u2, l2], cdf, target_grid) # median quantities
        
    sys.stdout.write("done!\n")

    return pdf_mean, pdf_med, pdf_mode, pdf_mc, pdf_l68, pdf_h68, pdf_l95, pdf_h95, pdf_std



########## KERNAL DENSITY ESTIMATION ##########


def gaussian(mu, std, x):
    """
    Compute (normalized) Gaussian kernal.

    Keyword arguments:
    mu -- mean (center)
    var -- standard deviation (width)
    x -- input grid

    Outputs:
    Normal(x | mu, std)
    """
    
    dif = x - mu # difference
    norm = 1. / sqrt(2*pi) / std # normalization
    
    return norm * exp(-0.5 * square(dif/std))


def pdf_kde(y, y_std, y_wt, x, dx, Ny, Nx, sig_thresh=5, wt_thresh=1e-3):
    """
    Compute smoothed PDF using kernel density estimation.

    Keyword arguments:
    y -- Gaussian kernel mean
    y_std -- Gaussian kernel standard deviation
    y_wt -- associated weight
    x -- PDF grid
    dx -- PDF spacing
    Ny -- number of objects
    Nx -- number of grid elements
    sig_thresh -- +/-sigma threshold for clipping kernels (default=5)
    wt_thresh -- wt/wt_max threshold for clipping observations (default=1e-3)

    Outputs:
    pdf -- probability distribution function (PDF) evaluated over x
    """

    # clipping kernel
    centers=((y-x[0]) / dx).astype(int) # discretized centers
    offsets = (sig_thresh * y_std / dx).astype(int) # discretized offsets
    uppers, lowers = centers+offsets, centers-offsets # upper/lower bounds
    uppers[uppers>Nx], lowers[lowers<0] = Nx, 0 # limiting to grid edges

    # initialize PDF
    pdf = zeros(Nx)

    # limit analysis to observations with statistically relevant weight
    sel_arr = y_wt > (wt_thresh*y_wt.max())

    # compute PDF
    for i in arange(Ny)[sel_arr]: # within selected observations
        pdf[lowers[i]:uppers[i]] += y_wt[i] * gaussian(y[i], y_std[i], x[lowers[i]:uppers[i]]) # stack weighted Gaussian kernel over array slic
    
    return pdf


def pdf_kde_dict(ydict, ywidth, y_pos, y_idx, y_wt, x, dx, Ny, Nx, wt_thresh=1e-3):
    """
    Compute smoothed PDF from point estimates using KDE utilizing a PRE-COMPUTED DICTIONARY.

    Keyword arguments:
    ydict -- dictionary of kernels
    ywidth -- associated widths of kernels
    y_pos -- discretized position of observed data
    y_idx -- corresponding index of kernel from dictionary
    y_wt -- associated weight
    x -- PDF grid
    dx -- PDF spacing
    Ny -- number of objects
    Nx -- number of grid elements
    wt_thresh -- wt/wt_max threshold for clipping observations (default=1e-3)

    Outputs:
    pdf -- probability distribution function (PDF) evaluated over x
    """

    # initialize PDF
    pdf = zeros(Nx) 

    # limit analysis to observations with statistically relevant weight
    sel_arr = y_wt > (wt_thresh*y_wt.max())

    # compute PDF
    for i in arange(Ny)[sel_arr]: # within selected observations
        idx = y_idx[i] # dictionary element
        yp = y_pos[i] # kernel center
        yw = ywidth[idx] # kernel width
        pdf[yp-yw:yp+yw+1] += y_wt[i] * ydict[idx] # stack weighted Gaussian kernel over array slice
    
    return pdf


####################################################################################################


########## WINBET ##########

### THIS HAS TO BE ENTIRELY RE-WRITTEN
class WINBET():
    """
    The Weighted Inference with Naive Bayes and Extra Trees (WINBET) class. 

    Functions: 
    train -- train the trees
    impute -- generates missing photometric predictions
    """

    
    def __init__(self, Ntrees=100, Nleaf=10):
        """
        Initializes the instance. 

        Keyword arguments:
        N_members -- number of members in ensemble
        See sklearn.neighbors.NearestNeighbors for a description of additional keyword arguments and their defaults.
        """

        # establish baseline model
        self.NTREES=Ntrees # number of trees
        self.NLEAF=Nleaf # minimum number of objects per leaf
        self.lf=tree.ExtraTreeRegressor(min_samples_leaf=self.NLEAF)

    def train(self, phot, var, masks, X, Xe, Xdict):
        """
        Train underlying trees using Naive Bayes and Extra Trees.

        Keyword arguments:
        phot -- measured fluxes
        var -- measured flux variances
        masks -- flux masks
        X -- transformed features
        Xe -- transformed feature errors
        Xdict -- feature dictionary
        """

        # initialize stuff
        self.NOBJ,self.NFILT=len(phot),len(phot[0]) # number of objects/filters
        self.censor_sel=arange(self.NOBJ)[(masks.sum(axis=1)<self.NFILT)] # indices of objects with censored (missing) data
        self.NCENSOR=len(self.censor_sel) # number of censored objects
        self.csel=[(masks[:,i]==False) for i in xrange(self.NFILT)] # mask slice in relevant dimension
        self.NFILL=[self.csel[i].sum() for i in xrange(self.NFILT)] # number of censored mags to be filled in

        self.phot=copy(phot)
        self.var=copy(var)
        self.masks=copy(masks)

        # fill in missing features with arbitrary values
        Xt=copy(X)
        Xet=copy(Xe)
        Xt[masks==False]=1.
        Xet[masks==False]=1.
    
        # discretize features along dictionary
        Xidx,Xeidx=Xdict.fit(Xt,Xet)

    
        # construct Naive Bayes priors
        X_pdf=array([pdf_kde_dict(Xdict.sig_dict,Xdict.sig_width,Xidx[:,i],Xeidx[:,i],masks[:,i].astype(int),Xdict.grid,Xdict.delta,Xdict.Ngrid) for i in xrange(self.NFILT)]) # feature PDF
        X_cdf=X_pdf.cumsum(axis=1)/X_pdf.sum(axis=1)[:,None] # compute CDF
        self.Xcdf=[unique(X_cdf[i]) for i in xrange(self.NFILT)] # select unique elements
        self.Xcdf_grid=[Xdict.grid[unique(X_cdf[i],return_index=True)[1]] for i in xrange(self.NFILT)] # select corresponding unique grid elements

    
        # initialize Extra Tree Regressor
        self.test_indices=[[] for i in xrange(self.NCENSOR)] # collection of neighbors for each object

        # gather neighbors
        for counter in xrange(self.NTREES):
            sys.stdout.write(str(counter)+' ')
            sys.stdout.flush()

            # generate new fluxes (training)
            X_filled=normal(Xt,Xet) # perturb features
            for i in xrange(self.NFILT):
                Xfill=interp(random.uniform(size=self.NFILL[i]),self.Xcdf[i],self.Xcdf_grid[i]) # draw from CDF
                X_filled[self.csel[i],i]=Xfill # impute censored fluxes

            # train tree
            self.lf.fit(X_filled,X_filled) # fit data
            idx=self.lf.apply(X_filled) # map training data leaf indices
            tree_idx=[[] for i in xrange(max(idx)+1)] # tree-structured object list
            for i in xrange(len(idx)):
                tree_idx[idx[i]].append(i) # add object to tree-indexed list

            # generate new fluxes (testing)
            X_filled=normal(Xt,Xet) # perturb features
            for i in xrange(self.NFILT):
                Xfill=interp(random.uniform(size=self.NFILL[i]),self.Xcdf[i],self.Xcdf_grid[i]) # draw from CDF
                X_filled[self.csel[i],i]=Xfill # impute censored fluxes

            # query objects with censored fluxes
            tidx=self.lf.apply(X_filled[self.censor_sel]) 
            for i in xrange(self.NCENSOR):
                self.test_indices[i].append(tree_idx[tidx[i]]) # add leaf neighbors to object-indexed list  


    def impute(self, phot, var, masks, impute_type='random', ll_func=loglikelihood):
        """
        Impute missing photometry.

        Keyword arguments:
        phot -- measured fluxes (same as train)
        var -- measured flux variances (same as train)
        masks -- flux masks (same as train)
        impute_type -- 'mean' or 'random'
        ll_func -- loglikelihood function (default: loglikelihood)

        Outputs:
        phot_impute -- filled photometric data
        var_impute -- filled variance data
        """

        pcensor=empty((self.NOBJ,self.NFILT))
        vcensor=empty((self.NOBJ,self.NFILT))

        # compute likelihood-weighted estimates for fluxes, errors
        for obj in xrange(self.NCENSOR):
            if obj%500==0: 
                sys.stdout.write(str(obj)+' ')
                sys.stdout.flush()

            idx=self.censor_sel[obj] # object index

            tidx=pandas.unique([i for j in self.test_indices[obj] for i in j]) # derive unique list of neighbors
            ptemp=phot[tidx] # phot subset
            vtemp=var[tidx] # var subset
            mtemp=masks[tidx] # mask subset

            # derive likelihoods
            ll,nbands=ll_func(phot[idx],var[idx],masks[idx],ptemp,vtemp,mtemp)

            # derive dimension-specific collections of weights and photometry
            mtemp=[mtemp[:,i]&(nbands>0) for i in xrange(self.NFILT)] # band-specific masks
            ptemp=[ptemp[:,i][mtemp[i]] for i in xrange(self.NFILT)] # band-specific phot
            vtemp=[vtemp[:,i][mtemp[i]] for i in xrange(self.NFILT)] # band-specific var
            lltemp=[ll[mtemp[i]] for i in xrange(self.NFILT)] # band-specific log-likelihoods
            wtemp=[exp(-0.5*(lltemp[i]-lltemp[i].min())) for i in xrange(self.NFILT)] # band-specific weights

            if impute_type=='mean':
                # compute expected photometry
                p1=array([average(ptemp[i],weights=wtemp[i]) for i in xrange(self.NFILT)]) # mean phot (first moment): E(X)
                p2=array([average(square(ptemp[i]),weights=wtemp[i]) for i in xrange(self.NFILT)]) # second moment: E(X^2)
                v1_i=p2-square(p1) # variance imputed phot: V(X)=E(X^2)-E^2(X)
                v1_m=array([average(vtemp[i],weights=wtemp[i]) for i in xrange(self.NFILT)]) # mean measured variance: E(Xe^2)
                v_eff=v1_i+v1_m # effective error: observed variance and mean variance in quadrature sqrt([E(Xe^2)+V(X)])
                pcensor[idx]=p1 # fill in photometry
                vcensor[idx]=v_eff # fill in photometry

            if impute_type=='random':
                # generate random sample
                choice_idx=[choice(tidx[mtemp[i]],p=wtemp[i]/wtemp[i].sum()) for i in xrange(self.NFILT)]
                pcensor[idx]=phot[choice_idx,xrange(self.NFILT)]
                vcensor[idx]=var[choice_idx,xrange(self.NFILT)]

        # fill in photometry
        phot_impute=copy(phot)
        var_impute=copy(var)
        phot_impute[masks==False]=pcensor[masks==False]
        var_impute[masks==False]=vcensor[masks==False]
    
        return phot_impute,var_impute


########### FRANKEN-Z ###############


class FRANKENZ():
    """
    The Flexible Regression over Associated Neighbors using Kernel dEsity estimatioN for Redshift (FRANKEN-Z) class. 

    Functions: 
    predict -- generates redshift PDF predictions
    """
    
    def __init__(self, N_members=100, n_neighbors=10, radius=1.0, algorithm='kd_tree', leaf_size=50,
                 p=2, metric='minkowski', metric_params=None, n_jobs=1):
        """
        Initializes the instance. 

        Keyword arguments:
        N_members -- number of members in ensemble
        See sklearn.neighbors.NearestNeighbors for a description of additional keyword arguments.
        """

        # establish baseline model
        self.knn = neighbors.NearestNeighbors(n_neighbors=n_neighbors, radius=radius, algorithm=algorithm,
                                              leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=1)
        self.NNEIGHBORS = n_neighbors # number of nearest neighbors (per member)
        self.metric = metric # chosen distance metric
        self.NBRUTEFORCE = leaf_size # leaf size (for brute force calculation)
        self.NMEMBERS = N_members # ensemble size (number of Monte Carlo draws)


    def predict(self, x_train, xe_train, xm_train, x_targ, xe_targ, xm_targ,
                feature_map=asinh_mag_map, ll_func=loglikelihood, impute_train=None, impute_targ=None):
        """
        Compute log-likelihoods over neighboring training objects.

        Keyword arguments:
        x_[train/targ] -- features (training/testing)
        xe_[train/targ] -- feature errors (training/testing)
        xm_[train/targ] -- feature masks (training/testing)        
        feature_map -- feature transformation map (default=asinh_mag_map)
        ll_func -- log-likelihood function (default=loglikelihood)
        impute_train -- 
        impute_[train/targ] -- instances used to impute missing quantities (default=None)

        Outputs:
        model_objects -- unique matched object indices
        model_Nobj -- number of unique matched objects
        model_ll -- -2ln(likelihood)
        model_Ndim -- number of overlapping observations used to compute log-likelihoods
        """

        Ntrain, Ntarg = len(x_train), len(x_targ) # size of training/testing sets
        Npred = self.NNEIGHBORS * self.NMEMBERS # number of non-unique predictions
        Ndim = x_train.shape[1] # number of dimensions

        model_indices = empty((self.NMEMBERS, Ntarg, self.NNEIGHBORS), dtype='int') # NON-UNIQUE collection of training object indices selected for each target object

        model_Nobj = empty(Ntarg, dtype='int') # number of unique training objects selected for each target object
        model_objects = empty((Ntarg, Npred), dtype='int') # UNIQUE collection of training object indices selected for each target object
        model_ll = empty((Ntarg, Npred), dtype='float32') # log-likelihood
        model_Ndim = empty((Ntarg, Npred), dtype='uint8') # number of dimensions used in fit

        # neighbor selection step
        for i in xrange(self.NMEMBERS): # for each member of the ensemble
            sys.stdout.write('('+str(i)+') ') # print progress
            sys.stdout.flush()

            # impute missing training values
            if impute_train is not None:
                x_train_t, xe_train_t = impute_train.impute(x_train, xe_train, xm_train, impute_type='random') # impute values
                x_train_t = normal(x_train_t, xe_train_t) # jitter
            else:
                x_train_t, xe_train_t = normal(x_train, xe_train).astype('float32'), xe_train # jitter

            # impute missing target values
            if impute_targ is not None:
                x_targ_t, xe_targ_t = impute_targ.impute(x_targ, xe_targ, xm_targ, impute_type='random')
                x_targ_t = normal(x_targ_t, xe_targ_t)
            else:
                x_targ_t, xe_targ_t = normal(x_targ, xe_targ).astype('float32'), xe_targ # perturb fluxes

            # transform features
            X_train_t, Xe_train_t = feature_map(x_train_t, xe_train_t) # training data
            X_targ_t, Xe_targ_t = feature_map(x_targ_t, xe_targ_t) # testing data

            # find neighbors
            knn = base.clone(self.knn).fit(X_train_t) # train k-d tree
            model_indices[i] = knn.kneighbors(X_targ_t, return_distance=False) # query k-d tree
            

        # log-likelihood step
        for i in xrange(Ntarg): # for each target object
            midx_unique = pandas.unique(model_indices[:,i,:].flatten()) # select unique indices
            Nidx = len(midx_unique) # number of unique indices
            model_Nobj[i] = Nidx
            
            model_objects[i][:Nidx] = midx_unique # assign unique indices
            model_objects[i][Nidx:] = -99 # right-pad with defaut values

            # compute log-likelihoods
            model_ll[i][:Nidx], model_Ndim[i][:Nidx] = ll_func(x_targ[i], xe_targ[i], xm_targ[i],
                                                               x_train[midx_unique], xe_train[midx_unique], xm_train[midx_unique])
            model_ll[i][Nidx:], model_Ndim[i][Nidx:] = -99., -99.

            if i%5000==0: # update status every 5000 objects
                sys.stdout.write(str(i)+' ')
                sys.stdout.flush()
                gc.collect() # garbage collect
    
        sys.stdout.write('done!\n')

        return model_objects, model_Nobj, model_ll, model_Ndim





    



        
        


########## INPUT/OUTPUT OPERATIONS ##########


class ReadParams():
    """
    Read in configuration files and initialize parameters. [Code based on Gabriel Brammer's threedhst.eazyPy module.]
    """

    def __init__(self, config_file):
        """
        Process configuration file.
        """

        self.filename = config_file # filename

        # read in file
        f = open(config_file, 'r')
        self.lines = f.readlines()
        f.close()

        # process file
        self._process_params()

        # process additional configuration files
        for param in self.param_names:
            if 'CONFIG_' in param:
                fname = self.params[param] # grab filename
                exec("self." + param + "=ReadParams(self.params['HOME']+fname)") # assign ReadParams output to associated variable name


    def _process_params(self):
        """
        Process input parameters and add them to the class dictionary.
        """

        params = {} # parameters
        formats = {} # format of parameters
        self.param_names = [] # parameter names

        # extract parameters
        for line in self.lines:
            if (line.startswith('#') | line.startswith(' ')) is False:

                # split line
                lsplit = line.split()

                # assign name and parameter
                if len(lsplit) >= 2:
                    lsplit[0] = lsplit[0][:-1]
                    params[lsplit[0]] = lsplit[1]
                    self.param_names.append(lsplit[0])

                    # (re)assign formats
                    try:
                        flt = float(lsplit[1])
                        formats[lsplit[0]] = 'f'
                        params[lsplit[0]] = flt
                    except:
                        formats[lsplit[0]] = 's'

                    if params[lsplit[0]] == 'None':
                        params[lsplit[0]] = None
                        formats[lsplit[0]] = 'n'

        self.params = params
        self.formats = formats



class ReadFilters():
    """
    Read in filter files.
    """

    def __init__(self, filter_list, path='', Npoints=5e4):
        """
        Keyword arguments:
        filter_dir -- directory where filter files are stored
        path -- home path
        Npoints -- number of points used to interpolate filter transmission curves
        """

        
        f = open(filter_list) # open file list
        
        self.filters = []
        self.filenames = []
        
        for line in f:
            lsplit = line.split()
            self.filters.append(lsplit[0]) # filter name
            self.filenames.append(lsplit[1]) # file name
            
        f.close()

        self.NFILTER = len(self.filters) # number of filters
        
        self.fw = [0.]*self.NFILTER # filter wavelengths
        self.ft = [0.]*self.NFILTER # filter transmissions

        for i in xrange(self.NFILTER):
            self.fw[i], self.ft[i] = swapaxes(loadtxt(path+self.filenames[i]), 0, 1) # load file and swap dimensions

        self.lambda_eff=zeros(self.NFILTER) # effective wavelengths

        for i in xrange(self.NFILTER):
            nuMax=0.999*c/(min(self.fw[i])*1e-10) # max frequency
            nuMin=1.001*c/(max(self.fw[i])*1e-10) # min frequency
            nuInc=(nuMax-nuMin)/Npoints # increment (linear)
            nu=arange(nuMin,nuMax+nuInc,nuInc) # frequency array
            lnu=log(nu) # log(frequency)
            wave=c/nu # convert to wavelength
            lwave=log(wave) # log(wavelength)

            temp = interp(1e10*wave, self.fw[i],self.ft[i]) # interpolate filter transmission
            top = trapz(temp*lwave,lnu) # numerator
            bottom = trapz(temp,lnu) # denominator
            self.lambda_eff[i]=exp(top/bottom)*1e10 # compute effective wavelength [A]



########## DICTIONARIES ##########


class RedshiftDict():
    """
    Set up redshift grids and kernels used for computations.
    """

    
    def __init__(self, rparams, sigma_trunc=5.0):
        """
        Set up redshift grids/kernels.

        Keyword arguments:
        rparams -- redshift configuration parameters (see class::ReadParams)
        sigma_trunc -- number of standard deviations used before truncating the kernels (default=5)
        """

        self.Ndict = int(rparams['N_DICT']) # number of dictionary elements

        # discrete kernel parameters
        self.lze_grid = linspace(rparams['DLZ'], rparams['DLZ_MAX'], self.Ndict) # Gaussian dictionary parameter grid
        self.dlze = self.lze_grid[1] - self.lze_grid[0] # kernel spacing

        # high-res ln(1+z) grid
        self.res = int(rparams['RES']) # resolution
        self.dlz_highres = rparams['DLZ'] / rparams['RES'] # high-res spacing
        self.Npad = int(sigma_trunc*rparams['DLZ_MAX'] / self.dlz_highres) # padding on ends of grid (for sliding addition)
        self.lzgrid_highres = arange(log(1+rparams['ZMIN']), log(1+rparams['ZMAX']) + self.dlz_highres, self.dlz_highres) # high-res grid

        # left pad
        lpad = arange(log(1+rparams['ZMIN']) - self.dlz_highres*self.Npad, log(1+rparams['ZMIN']), self.dlz_highres)
        self.lzgrid_highres = append(lpad, self.lzgrid_highres)

        # right pad
        rpad = arange(log(1+rparams['ZMAX']) + self.dlz_highres, log(1+rparams['ZMAX']) + self.dlz_highres*(self.Npad+1), self.dlz_highres)
        self.lzgrid_highres = append(self.lzgrid_highres, rpad)
        
        self.Nz_highres = len(self.lzgrid_highres) # size of high-res grid

        # effective bounds of high-res grid
        self.zmin_idx_highres = argmin( abs(self.lzgrid_highres - log(1+rparams['ZMIN'])) ) # minimum
        self.zmax_idx_highres = argmin( abs(self.lzgrid_highres - log(1+rparams['ZMAX'])) ) # maximum
        self.zmax_idx_highres = self.zmin_idx_highres + int( ceil((self.zmax_idx_highres - self.zmin_idx_highres) / rparams['RES']) * rparams['RES'] ) # adding left-pad and offset

        # lower-res log(1+z) grid
        self.dlz = rparams['DLZ'] # dln(1+z)
        self.lzgrid = arange(self.lzgrid_highres[self.zmin_idx_highres], self.lzgrid_highres[self.zmax_idx_highres], self.dlz) # ln(1+z) grid
        self.Nz = len(self.lzgrid) # number of elements

        # corresponding redshift grids
        self.zgrid_highres = exp(self.lzgrid_highres[self.zmin_idx_highres:self.zmax_idx_highres]) - 1 # high-res z grid
        self.zgrid = exp(self.lzgrid) - 1 # low-res z grid

        # create dictionary
        self.lze_width = ceil(self.lze_grid*sigma_trunc / self.dlz_highres).astype('int') # width of kernel
        self.lze_dict = [gaussian(mu=self.lzgrid_highres[self.Nz_highres/2], std=self.lze_grid[i],
                                  x=self.lzgrid_highres[self.Nz_highres/2-self.lze_width[i]:self.Nz_highres/2+self.lze_width[i]+1])
                         for i in xrange(self.Ndict)] # dictionary
        
        # output redshift grid
        self.zgrid_out = arange(rparams['ZMIN_OUT'], rparams['ZMAX_OUT']+rparams['DZ_OUT'], rparams['DZ_OUT']) # output z grid
        self.dz_out = rparams['DZ_OUT'] # output dz
        self.dz_out_highres = rparams['DZ_OUT'] / rparams['RES_OUT'] # output high-resolution dz
        self.Nz_out = len(self.zgrid_out) # number of elements
        

    def fit(self, lz, lze):
        """
        Map Gaussian ln(1+z) PDFs onto the ln(1+z) dictionary.

        Keyword arguments:
        lz -- ln(1+z) means
        lze -- ln(1+z) standard deviation (i.e. dz/(1+z))

        Outputs:
        lz_idx -- corresponding dictionary grid indices (high-resolution)
        lze_idx -- corresponding dictionary kernel indices (high-resolution)
        """
        
        lz_idx = ( (lz-self.lzgrid_highres[0]) / self.dlz_highres ).round().astype('int') # discretize ln(1+z)
        lze_idx = ( (lze-self.lze_grid[0]) / self.dlze ).round().astype('int') # discretize dz/(1+z)
        lze_idx[lze_idx >= len(self.lze_grid)] = len(self.lze_grid) - 1 # impose error ceiling
        lze_idx[lze_idx < 0] = 0. # impose error floor
        
        return lz_idx, lze_idx


        
class PDFDict():
    """
    Set up underlying grids and kernels used to compute PDFs for ancillary parameters.
    """

    
    def __init__(self, pparams, sigma_trunc=5.):
        """
        Keyword arguments:
        pparams -- configuration parameters for the PDF file (see class:ReadParams)
        """

        self.Ndict = int(pparams['N_DICT']) # number of dictionary elements

        # initialize grid
        self.delta = pparams['DELTA'] # grid spacing
        self.min = pparams['MIN'] # grid lower bound
        self.max = pparams['MAX'] # grid upper bound
        self.grid = arange(self.min, self.max+self.delta/2, self.delta) # grid
        self.Ngrid = len(self.grid) # number of elements
        
        # create dictionary
        self.sig_grid = linspace(pparams['SIG_MIN'], pparams['SIG_MAX'], self.Ndict) # Gaussian dictionary parameter grid
        self.dsig = self.sig_grid[1] - self.sig_grid[0] # kernel spacing
        self.sig_width = ceil(self.sig_grid*sigma_trunc / self.delta).astype('int') # width of kernel
        self.sig_dict = [gaussian(mu=self.grid[self.Ngrid/2], std=self.sig_grid[i],
                                  x=self.grid[self.Ngrid/2-self.sig_width[i]:self.Ngrid/2+self.sig_width[i]+1])
                         for i in xrange(self.Ndict)] # dictionary


    def fit(self, X, Xe):
        """
        Map Gaussian PDFs onto the dictionary.

        Keyword arguments:
        X -- target features (mean)
        Xe -- target feature errors (std dev)

        Outputs:
        X_idx -- corresponding dictionary grid indices
        Xe_idx -- corresponding dictionary kernel indices
        """
        
        X_idx = ((X-self.grid[0]) / self.delta).round().astype('int')
        Xe_idx = ((Xe - self.sig_grid[0]) / self.dsig).round().astype('int')
        Xe_idx[Xe_idx >= len(self.sig_grid)] = len(self.sig_grid) - 1 # impose error ceiling
        Xe_idx[Xe_idx < 0] = 0. # impose error floor

        return X_idx,Xe_idx











################ PLOTTING ################

###### MANY OF THESE STILL ARE OLD

def plot_densities(train_pdf, out_pdf, x, out_pdf_draws=None, xbounds=[0,6], var_names=['Redshift','PDF'], sample_names=['True','Predicted'], colors=['black','red']):
    """
    Plot comparison between two number densities.

    Keyword arguments:
    train_pdf -- original PDF
    out_pdf -- comparison PDF
    x -- input grid
    out_pdf_draws -- samples from the PDF
    xbounds -- plotting range (default=0-6)
    var_names -- variable names (default='Redshift'/'PDF')
    sample_names -- names of samples (default='True'/'Predicted')
    colors -- colors for respective samples (default='black'/'red')

    Outputs:
    Identical to func::compute_density_score.
    """

    # compute errors from draws (if exists)
    if out_pdf_draws is not None:
        # compute density scores (Poisson, Anderson-Darling, and Error-weighted statistics)
        out_pdf_err = std(out_pdf_draws, axis=0) # naive sigma
        [N_p, arr_p, p_sel, prob_p], [ad_s, ad_cv, ad_p], [N_e, arr_e, e_sel, prob_e] = compute_density_score(train_pdf, out_pdf, out_pdf_err)
        if ad_p>1: ad_p=0. # assign p-value of zero if something went wrong

    else:
        # compute density scores (Poisson, Anderson-Darling, and Error-weighted statistics)
        [N_p, arr_p, p_sel, prob_p], [ad_s, ad_cv, ad_p] = compute_density_score(train_pdf, out_pdf)
        if ad_p>1: ad_p=0. # assign p-value of zero if something went wrong
    
    # initializing figure
    gs=gridspec.GridSpec(2, 1, height_ratios=[4,1])

    # plot number density
    subplot(gs[0])
    plot(x, train_pdf, color=colors[0], lw=3, label=sample_names[0])
    if out_pdf_draws is not None:
        for draw in out_pdf_draws:
            plot(x, draw, color=colors[1], lw=0.2, alpha=0.3)
    plot(x, out_pdf, color=colors[1], lw=3, label=sample_names[1])
    fill_between(x, train_pdf, out_pdf, color='yellow')
    xlim(xbounds)
    ylims = [0, max([max(train_pdf),max(out_pdf)])*1.1]
    ylim(ylims)
    legend(fontsize=24)
    xlabel(var_names[0])
    ylabel(var_names[1])
    text( xbounds[0] + (xbounds[1]-xbounds[0])*0.62, ylims[0] + (ylims[1]-ylims[0])*0.5, 'Pois$(S/n,p)$=('+str(round(N_p/sum(p_sel),2))+','+str(round(prob_p,2))+')' )
    text( xbounds[0] + (xbounds[1]-xbounds[0])*0.62, ylims[0] + (ylims[1]-ylims[0])*0.35, 'AD$(S,p)$=('+str(round(ad_s,2))+','+str(round(ad_p,2))+')' )
    if out_pdf_draws is not None:
        text( xbounds[0] + (xbounds[1]-xbounds[0])*0.62, ylims[0] + (ylims[1]-ylims[0])*0.2, 'Err$(S/n,p)$=('+str(round(N_e/sum(e_sel),2))+','+str(round(prob_e,2))+')' )
    tight_layout()

    # plot running Poisson/error fluctuation
    
    subplot(gs[1])
    xlim(xbounds)
    xlabel(var_names[0])
    ylabel('$\Delta \sigma$')
    plot(x, zeros(len(x)), 'k--', lw=2)
    fill_between(x[p_sel], arr_p, color='yellow', alpha=0.7)
    plot(x[p_sel], arr_p, lw=2, color=colors[0])
    ymin, ymax = round(min(arr_p),3), round(max(arr_p),3)
    yticks([ymin,ymax], fontsize=24)

    if out_pdf_draws is not None:
        fill_between(x[e_sel], arr_e, color='orange', alpha=0.7)
        plot(x[e_sel], arr_e, lw=2, color=colors[1])
        ymin, ymax = min(round(min(arr_e),3), ymin), max(round(max(arr_e),3), ymax)
        yticks([ymin,ymax], fontsize=24)

    tight_layout()

    if out_pdf_draws is not None:
        return [N_p, arr_p, p_sel, prob_p], [ad_s, ad_cv, ad_p], [N_e, arr_e, z_sel, prob_e]
    else:
        return [N_p, arr_p, p_sel, prob_p], [ad_s, ad_cv, ad_p]


def plot_points(y, yp, markersize=1.5, limits=[0,6], binwidth=0.05, thresh=10, cat_thresh=0.15, selection=None, weights=None):
    """
    Plot results from redshift POINT ESTIMATES. To illustrate density scales, 2-D density histograms are used for the majority of the data, while outlying points are plotted individually.

    Keyword arguments:
    y -- input values
    yp -- predicted values
    markersize -- size of outlying points
    limits -- scale of x,y axes
    binwidth -- width of 2-D histogram bins
    thresh -- threshold before switching from histogram to outlying points
    cat_thresh -- threshold used to categorize catastrophic failures
    selection -- selection array for plotting a subset of objects

    Outputs:
    Identical to func::compute_score.
    """

    cmap = get_cmap('rainbow')
    cmap.set_bad('white')

    success_sel = isfinite(yp) & (yp>0)

    if weights is not None:
        weights = weights
    else:
        weights = ones(len(y))
    
    if selection is not None:
        sel = success_sel & selection & (weights>0.)
    else:
        sel = success_sel & (weights>0.)

    score = compute_score(y[sel], yp[sel], weights=weights[sel])

    # declare binning parameters
    xyrange=[[0,10], [0,10]]
    bins=[arange(xyrange[0][0], xyrange[0][1]+binwidth, binwidth), arange(xyrange[1][0], xyrange[1][1]+binwidth, binwidth)]
    
    # bin data
    xdat, ydat = y[sel], yp[sel]
    hh, locx, locy = histogram2d(xdat, ydat, range=xyrange, bins=bins, weights=weights[sel])
    posx = digitize(xdat, locx)
    posy = digitize(ydat, locy)

    # select points within the histogram
    hhsub = hh[posx-1, posy-1] # values of the histogram where the points are
    xdat1 = xdat[(hhsub<thresh)] # low density points (x)
    ydat1 = ydat[(hhsub<thresh)] # low density points (y)
    hh[hh<thresh] = NaN # fill the areas with low density by NaNs

    # plot results
    plot(xdat1, ydat1, '.', color='black', markersize=markersize) # outliers/low-density regions
    imshow(flipud(hh.T), cmap = cmap, extent=array(xyrange).flatten(), interpolation='none', norm=matplotlib.colors.LogNorm()) # high-density regions

    # establishing the colorbar
    cbar = colorbar()
    cticks = arange( ceil(log10(nanmin(hh.flatten())) / 0.25) * 0.25, int(log10(nanmax(hh.flatten())) / 0.25) * 0.25 + 1e-6, 0.25)
    cticklabels = ['$10^{'+str(round(i,2))+'}$' for i in cticks]
    cbar.set_ticks(10**cticks)
    cbar.set_ticklabels(cticklabels)

    # plotting 1:1 line+bounds
    plot(array([0,100]), array([0,100]), 'k--', lw=3)
    plot(array([0,100]), array([0,100]) * (1+cat_thresh) + cat_thresh, 'k-.', lw=2)
    plot(array([0,100]), array([0,100]) * (1-cat_thresh) - cat_thresh, 'k-.', lw=2)

    # statistics
    Nobj = sum(weights[sel])
    text(1.2*(limits[1]/5.0), 4.7*(limits[1]/5.0), "$N$: "+str(int(Nobj))+" ("+str(round(Nobj*1.0/sum(weights[success_sel]),3))+")", fontsize=18, color='black')
    text(1.2*(limits[1]/5.0), 4.5*(limits[1]/5.0), "$\Delta z^\prime$ (mean): "+str(round(score[0][2],4)*100)+"%", fontsize=18, color='black')
    text(1.2*(limits[1]/5.0), 4.3*(limits[1]/5.0), "$\Delta z^\prime$ (med): "+str(round(score[1][2],4)*100)+"%", fontsize=18, color='black')
    text(1.2*(limits[1]/5.0), 4.1*(limits[1]/5.0), "$\sigma_{\Delta z^\prime}$ (MAD): "+str(round(score[1][3],4)*100)+"%", fontsize=18, color='black')
    text(1.2*(limits[1]/5.0), 3.9*(limits[1]/5.0), "$f_{cat}$: "+str(round(score[2],4)*100)+"%", fontsize=18, color='black')

    # miscallaneous
    xlabel('Input')
    ylabel('Output')
    xlim(limits)
    ylim(limits)

    return score


def plot_zpdfstack(zpdf, zgrid, lz_idx, lze_idx, rdict, sel=None, weights=None, limits=[0,6,1.0], pdf_thresh=1e-1, plot_thresh=50., boxcar=1):
    """
    Plot 2-D P(z) vs redshift.

    Keyword arguments:
    zpdf -- redshift PDFs
    zgrid -- redshift grid PDFs are evaluated on
    cell_lz_idx -- log(1+z) discretized indices
    cell_lze_idx -- log(1+z) dictionary indoces
    rdict -- redshift dictionary
    sel -- selection array
    limits -- [low,high,delta] parameters for plotting
    pdf_thresh -- input to func::pdf_kde_wt_dict
    plot_thresh -- minimum threshold for plotting stacked PDFs
    boxcar -- width of boxcar for smoothing running mean, median, etc.

    Outputs:
    2d_stack,np,zpoints
    2d_stack -- 2-D stacked PDF
    np -- number density computed along input redshift axis
    zpoints -- point estimates from func::pdfs_summary_statistics computed along input redshift axis
    """

    Np,Nz=len(rdict.lzgrid_highres),len(zgrid)
    Nobj=len(zpdf)
    temp_stack=zeros((Np,Nz)) # 2-D P(z) grid

    if sel is None:
        sel=ones(Nobj,dtype='bool')

    if weights is None:
        weights=ones(Nobj,dtype='float32')

    # compute stack
    count=0
    for i in arange(Nobj)[sel]:
        if count%5000==0: sys.stdout.write(str(count)+' ')
        count+=1
        tzpdf=zpdf[i] # redshift pdf
        tsel=tzpdf>max(tzpdf)*pdf_thresh # probability threshold cut
        x_idx,x_cent=lze_idx[i],lz_idx[i]
        x_bound=rdict.lze_width[x_idx] # dictionary entry, location, and width
        tstack=rdict.lze_dict[x_idx][:,None]*tzpdf[tsel] # 2-D pdf
        temp_stack[x_cent-x_bound:x_cent+x_bound+1,tsel]+=tstack*weights[i] # stack 2-D pdf

    zpoints=pdfs_summary_statistics(zgrid,temp_stack/temp_stack.sum(axis=1)[:,None]) # compute summary statistics
    for i in zpoints:
        i[i==zgrid[0]]=NaN

    # converting from log to linear space
    temp_stack=temp_stack[rdict.zmin_idx_highres:rdict.zmax_idx_highres:int(rdict.res)]/(1+rdict.zgrid)[:,None] # reducing resolution
    prob=interp(zgrid,rdict.zgrid,temp_stack.sum(axis=1)) # running pdf
    temp_stack=swapaxes(pdfs_resample(rdict.zgrid,swapaxes(temp_stack,0,1),zgrid),0,1) # resampling to linear redshift grid
    temp_stack*=prob[None,:] # re-normalizing
    temp_stack=ma.array(temp_stack,mask=temp_stack<plot_thresh) # mask array

    # plot results
    zgrid_highres=exp(rdict.lzgrid_highres)-1
    imshow(swapaxes(temp_stack,0,1),origin='lower',aspect='auto',norm=matplotlib.colors.LogNorm(vmin=None, vmax=None),extent=(zgrid[0],zgrid[-1],zgrid[0],zgrid[-1]))
    colorbar(label='PDF')
    plot(array([0,100]),array([0,100]),'k--',lw=3) # 1:1 relation
    plot(array([0,100]),array([0,100])*1.15+0.15,'k-.',lw=2) # +15% bound
    plot(array([0,100]),array([0,100])*0.85-0.15,'k-.',lw=2) # -15% bound 
    #plot(convolve(zgrid_highres[isnan(zpoints[0])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[0][isnan(zpoints[0])==False],ones(boxcar)/float(boxcar),'valid'),color='black',lw=2) # mean
    plot(convolve(zgrid_highres[isnan(zpoints[1])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[1][isnan(zpoints[1])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',lw=2) # median
    #plot(convolve(zgrid_highres[isnan(zpoints[3])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[3][isnan(zpoints[3])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',linestyle='--',lw=2) # lower 68% CI
    #plot(convolve(zgrid_highres[isnan(zpoints[4])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[4][isnan(zpoints[4])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',linestyle='--',lw=2) # upper 68% CI
    #plot(convolve(zgrid_highres[isnan(zpoints[5])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[5][isnan(zpoints[5])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',linestyle='-.',lw=2) # lower 95% CI
    #plot(convolve(zgrid_highres[isnan(zpoints[6])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[6][isnan(zpoints[6])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',linestyle='-.',lw=2) # upper 95% CI
    xticks(arange(limits[0],limits[1]+limits[2],limits[2]))
    xlim([limits[0],limits[1]])
    yticks(arange(limits[0],limits[1]+limits[2],limits[2]))
    ylim([limits[0],limits[1]])
    xlabel('Redshift (input)')
    ylabel('Redshift (output)')
    tight_layout()

    return temp_stack,prob,zpoints


def plot_pdfstack(zpdf, zgrid, p_idx, pe_idx, pdict, pparams, pname, sel=None, weights=None, yparams=[0,6,1.0], pdf_thresh=1e-1, plot_thresh=50., boxcar=1):
    """
    Plot 2-D P(z) vs input parameter dictionary.

    Keyword arguments:
    zpdf -- redshift PDFs
    zgrid -- redshift grid PDFs are evaluated on
    p_idx -- parameter discretized indices
    pe_idx -- parameter dictionary indices
    pdict -- parameter dictionary
    pparams -- as limits (see func::plot_zpdfstack), but for the parameter of interest
    pname -- parameter name
    See func::plot_zpdfstack for additional inputs.

    Outputs:
    2d_stack,np,zpoints
    2d_stack -- 2-D stacked PDF
    np -- number density computed along input parameter axis
    zpoints -- point estimates from func::pdfs_summary_statistics computed along input parameter axis
    """

    Np,Nz=len(pdict.grid),len(zgrid)
    Nobj=len(zpdf)
    temp_stack=zeros((Np,Nz)) # 2-D P(z) grid

    if sel is None:
        sel=ones(Nobj,dtype='bool')

    if weights is None:
        weights=ones(Nobj,dtype='float32')

    # compute stack
    count=0
    for i in arange(Nobj)[sel]:
        if count%5000==0: sys.stdout.write(str(count)+' ')
        count+=1
        tzpdf=zpdf[i] # redshift pdf
        tsel=tzpdf>max(tzpdf)*pdf_thresh # probability threshold cut
        x_idx,x_cent=pe_idx[i],p_idx[i]
        x_bound=pdict.sig_width[x_idx] # dictionary entry, location, and width
        tstack=pdict.sig_dict[x_idx][:,None]*tzpdf[tsel] # 2-D pdf
        temp_stack[x_cent-x_bound:x_cent+x_bound+1,tsel]+=tstack # stack 2-D pdf

    # truncate array
    ylow,yhigh=argmin(abs(pdict.grid-pparams[0])),argmin(abs(pdict.grid-pparams[1]))
    temp_stack=temp_stack[ylow:yhigh+1]
    zpoints=pdfs_summary_statistics(zgrid,temp_stack/temp_stack.sum(axis=1)[:,None]) # compute summary statistics
    for i in zpoints:
        i[i==zgrid[0]]=NaN

    prob=trapz(temp_stack,zgrid) # running pdf
    temp_stack=ma.array(temp_stack,mask=temp_stack<plot_thresh) # mask array

    # plot results
    pgrid=pdict.grid[ylow:yhigh+1]
    imshow(swapaxes(temp_stack,0,1),origin='lower',aspect='auto',norm=matplotlib.colors.LogNorm(vmin=None, vmax=None),extent=(pparams[0],pparams[1],zgrid[0],zgrid[-1]))
    colorbar(label='PDF')
    #plot(convolve(pgrid[isnan(zpoints[0])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[0][isnan(zpoints[0])==False],ones(boxcar)/float(boxcar),'valid'),color='black',lw=2) # mean
    plot(convolve(pgrid[isnan(zpoints[1])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[1][isnan(zpoints[1])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',lw=2) # median
    #plot(convolve(pgrid[isnan(zpoints[3])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[3][isnan(zpoints[3])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',linestyle='--',lw=2) # lower 68% CI
    #plot(convolve(pgrid[isnan(zpoints[4])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[4][isnan(zpoints[4])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',linestyle='--',lw=2) # upper 68% CI
    #plot(convolve(pgrid[isnan(zpoints[5])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[5][isnan(zpoints[5])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',linestyle='-.',lw=2) # lower 95% CI
    #plot(convolve(pgrid[isnan(zpoints[6])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[6][isnan(zpoints[6])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',linestyle='-.',lw=2) # upper 95% CI
    xticks(arange(pparams[0],pparams[1]+pparams[2],pparams[2]))
    xlim([pparams[0],pparams[1]])
    yticks(arange(yparams[0],yparams[1]+yparams[2],yparams[2]))
    ylim([yparams[0],yparams[1]])
    xlabel(pname)
    ylabel('Redshift (output)')
    tight_layout()

    return temp_stack,prob,zpoints


def plot_dpdfstack(zpdf, zgrid, z, p_idx, pe_idx, pdict, pparams, pname, sel=None, weights=None, yparams=[-0.5,0.5,0.1], ybins=[-1.0,1.0,1e-2], pdf_thresh=1e-1, plot_thresh=50., boxcar=1):
    """
    Plot 2-D [P(z)-z]/(1+z) (redshift dispersion) vs input parameter dictionary.

    Keyword arguments:
    zpdf -- redshift PDFs
    zgrid -- redshift grid PDFs are evaluated on
    p_idx -- parameter discretized indices
    pe_idx -- parameter dictionary indices
    pdict -- parameter dictionary
    pparams -- as limits (see func::plot_zpdfstack), but for the parameter of interest
    pname -- parameter name
    ybins -- binning used when computing redshift dispersion
    See func::plot_zpdfstack for additional inputs.

    Outputs:
    2d_stack,np,zpoints
    2d_stack -- 2-D stacked PDF
    np -- number density computed along input parameter axis
    zpoints -- point estimates from func::pdfs_summary_statistics computed along input parameter axis
    """

    # error distribution (PDF)
    zdisp_bins=arange(ybins[0],ybins[1],ybins[2])
    zdisp_grid=(zdisp_bins[1:]+zdisp_bins[:-1])/2.0

    Np,Ndisp=len(pdict.grid),len(zdisp_bins)-1
    Nobj=len(zpdf)
    temp_stack=zeros((Np,Ndisp)) # 2-D P(z) grid

    if sel is None:
        sel=ones(Nobj,dtype='bool')

    if weights is None:
        weights=ones(Nobj,dtype='float32')

    count=0
    for i in arange(Nobj)[sel]:
        if count%5000==0: sys.stdout.write(str(count)+' ')
        count+=1
        tzpdf=zpdf[i]
        tsel=tzpdf>max(tzpdf)*pdf_thresh # probability threshold cut
        x_idx,x_cent=pe_idx[i],p_idx[i] # dictionary entry and location
        x_bound=pdict.sig_width[x_idx] # dictionary width
        pstack=histogram((zgrid[tsel]-z[i])/(1+z[i]),zdisp_bins,weights=tzpdf[tsel])[0] # d(pdf) stack
        psel=pstack>max(pstack)*pdf_thresh
        tstack=pdict.sig_dict[x_idx][:,None]*pstack[psel]
        temp_stack[x_cent-x_bound:x_cent+x_bound+1,psel]+=tstack*weights[i]

    # truncate array
    ylow,yhigh=argmin(abs(pdict.grid-pparams[0])),argmin(abs(pdict.grid-pparams[1]))
    temp_stack=temp_stack[ylow:yhigh+1]
    zpoints=pdfs_summary_statistics(zdisp_grid,temp_stack/temp_stack.sum(axis=1)[:,None]) # compute summary statistics
    for i in zpoints:
        i[i==zdisp_grid[0]]=NaN

    prob=trapz(temp_stack,zdisp_grid) # running pdf
    temp_stack=ma.array(temp_stack,mask=temp_stack<plot_thresh) # mask array

    # plot results
    pgrid=pdict.grid[ylow:yhigh+1]
    imshow(swapaxes(temp_stack,0,1),origin='lower',aspect='auto',norm=matplotlib.colors.LogNorm(vmin=None, vmax=None),extent=(pparams[0],pparams[1],zdisp_grid[0],zdisp_grid[-1]))
    colorbar(label='PDF')
    plot(array([-100,100]),array([0,0]),'k--',lw=3)
    plot(array([-100,100]),[0.15,0.15],'k-.',lw=2)
    plot(array([-100,100]),[-0.15,-0.15],'k-.',lw=2)
    #plot(convolve(pgrid[isnan(zpoints[0])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[0][isnan(zpoints[0])==False],ones(boxcar)/float(boxcar),'valid'),color='black',lw=2) # mean
    plot(convolve(pgrid[isnan(zpoints[1])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[1][isnan(zpoints[1])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',lw=2) # median
    #plot(convolve(pgrid[isnan(zpoints[3])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[3][isnan(zpoints[3])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',linestyle='--',lw=2)
    #plot(convolve(pgrid[isnan(zpoints[4])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[4][isnan(zpoints[4])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',linestyle='--',lw=2)
    #plot(convolve(pgrid[isnan(zpoints[5])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[5][isnan(zpoints[5])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',linestyle='-.',lw=2)
    #plot(convolve(pgrid[isnan(zpoints[6])==False],ones(boxcar)/float(boxcar),'valid'),convolve(zpoints[6][isnan(zpoints[6])==False],ones(boxcar)/float(boxcar),'valid'),color='darkviolet',linestyle='-.',lw=2)    
    xticks(arange(pparams[0],pparams[1]+pparams[2],pparams[2]))
    xlim([pparams[0],pparams[1]])
    yticks(arange(yparams[0],yparams[1]+yparams[2],yparams[2]))
    ylim([yparams[0],yparams[1]])
    xlabel(pname)
    ylabel('$\Delta z/(1+z)$ (output)')
    tight_layout()

    return temp_stack,prob,zpoints


