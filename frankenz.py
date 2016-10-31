###### FRANKEN-Z FUNCTIONS ##########
# Current version (v0) written by Josh Speagle (Harvard University; jspeagle@cfa.harvard.edu)
# Released under MIT License.
# Please see Speagle et al. (2017) arxiv:XXXX.XXXXX for more details.




########## MODULES ##########

# general environment
import numpy as np
import matplotlib
import scipy
from numpy import *
from numpy.random import *
from matplotlib import *
from matplotlib.pyplot import *
from scipy import *

# general functions
import pandas # uniqueness checks
import sys # print statements
from astropy.io import fits # I/O on fits
from sklearn.externals import joblib # I/O on ML models
import os # used to check for files
from scipy import interpolate # interpolation

# pre-processing and cross-validation
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

# machine learning
from sklearn import tree # decision trees
from sklearn import neighbors # nearest neighbors
from sklearn import base # additional methods

# statistics
from scipy import stats
from scipy import random

# additional memory management
import gc

# confidence intervals
SIG1=68.2689492/100.
SIG2=95.4499736/100.
SIG3=99.7300204/100.


# pre-defined constants
l2pi=log(2*pi)


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










########## MODEL FITTING ##########


def loglikelihood(data, data_var, data_mask, models, models_var, models_mask):
    """
    Compute -2lnL W/ FIXED SCALING using a set of models W/ ERRORS.

    Keyword arguments:
    data -- input fluxes
    data_var -- input variances
    data_mask -- mask for missing input data
    models -- collection of comparison models
    models_var -- model variances
    models_mask -- mask for missing model data

    Outputs:
    chi2_mod -- -2lnL for each model
    Nbands -- total number of bands used in likelihood calculation
    """

    tot_var=data_var+models_var # combined variance
    tot_mask=data_mask*models_mask # combined binary mask
    Nbands=tot_mask.sum(axis=1) # number of bands

    # compute ln(likelihood)    
    resid=data-models # residuals
    chi2=(tot_mask*resid*resid/tot_var).sum(axis=1) # compute standard chi2
    chi2_mod=chi2-Nbands # normalize by E[chi2(N)]
    
    return chi2_mod, Nbands



def loglikelihood_s(data, data_var, data_mask, models, models_var, models_mask):
    """
    Compute -2lnL W/ FREE SCALING using a set of models W/O ERRORS.

    Keyword arguments:
    data -- input fluxes
    data_var -- input variances
    data_mask -- mask for missing input data
    models -- collection of comparison models
    models_var -- model variances (unused)
    models_mask -- mask for missing model data

    Outputs:
    chi2_mod -- -2lnL for each model
    chi2_s -- maximum-likelihood model scalefactor
    chi2_a -- maximum-likelihood model 'shapefactor' (i.e. quadratic term)
    Nbands -- total number of bands used in likelihood calculation
    """

    tot_mask=data_mask*models_mask # combined binary mask
    Nbands=tot_mask.sum(axis=1) # number of bands
    
    # derive scalefactors between data and models
    inter_vals=(tot_mask*models*data[None,:]/data_var[None,:]).sum(axis=1) # interaction term
    shape_vals=(tot_mask*models*models/data_var[None,:]).sum(axis=1) # model-dependent term (i.e. quadratic 'steepness' of chi2)
    scale_vals=inter_vals/shape_vals # maximum-likelihood scalefactors

    # compute ln(likelihood)
    resid=data-scale_vals[:,None]*models # compute scaled residuals
    
    chi2=(tot_mask*resid*resid/data_var[None,:]).sum(axis=1) # compute chi2
    chi2_mod=chi2-(Nbands-1) # normalize by E[chi2(N-1)]
    
    return chi2_mod, Nbands






########## FEATURE MAPS ##########


def asinh_mag_map(phot, err, skynoise, zeropoint=None):
    """
    Map input fluxes/errors to asinh magnitudes (i.e. "Luptitude"; Lupton et al. 1999).

    Keyword arguments:
    phot -- fluxes
    err -- flux errors
    skynoise -- background sky noise (i.e. softening parameter)
    zeropoint -- flux zero-points (default: 1.)
    
    Outputs:
    asinh_mag -- asinh magnitudes
    asinh_mag_err -- associated transformed errors
    """

    # initialize flux zero-points
    Nbands=(phot.shape)[-1] # total number of bands
    if zeropoint is None:
        zeropoint=ones(Nbands)

    # compute Luptitudes 
    mag_asinh=-2.5/log(10)*(arcsinh(phot/(2*skynoise))+log(skynoise/zeropoint))
    mag_asinh_err=sqrt(square(2.5*log10(e)*err)/(square(2*skynoise)+square(phot)))

    return mag_asinh,mag_asinh_err


def inv_asinh_mag_map(mag, magerr, skynoise, zeropoint=None):
    """
    Map input asinh magnitudes to fluxes.

    Keyword arguments:
    mag -- asinh magnitudes
    magerr -- asinh magnitude errors
    zeropoint -- flux zero-points (default: 1.)

    Outputs: 
    phot -- fluxes
    err -- associated flux errors
    """

    # initialize flux zero-points
    Nbands=(mag.shape)[-1] # total number of bands
    if zeropoint is None:
        zeropoint=ones(Nbands)

    # compute fluxes
    phot=sinh(-log(10)/2.5*mag-log(skynoise/zeropoint))*(2*skynoise)
    err=sqrt(square(magerr)*(square(2*skynoise)+square(phot)))/(2.5*log10(e))

    return phot,err

def asinh_magcolor_map(phot, err, zeropoint=None, skynoise=None):
    """
    Map input fluxes/errors to asinh magnitudes (i.e. "Luptitude"; Lupton et al. 1999) AND colors.

    Keyword arguments:
    phot -- fluxes
    err -- flux errors
    zeropoint -- flux zero-points (default: 1.)
    skynoise -- background sky noise (default: median(err))
    
    Outputs:
    asinh_magcolor -- asinh magnitudes
    asinh_magcolor_err -- associated transformed errors
    """
    
    # initialize flux zero-points and sky noise levels
    Nbands=(phot.shape)[-1] # total number of bands
    if zeropoint is None:
        zeropoint=ones(Nbands)
    if skynoise is None:
        skynoise=median(err,axis=0)

    # compute Luptitudes 
    mag_asinh=-2.5/log(10)*(arcsinh(phot/(2*skynoise))+log(skynoise/zeropoint))
    mag_asinh_err=sqrt(square(2.5*log10(e)*err)/(square(2*skynoise)+square(phot)))

    # compute colors
    color_asinh=mag_asinh[:,:-1]-mag_asinh[:,1:]
    color_asinh_err=sqrt(square(mag_asinh_err[:,:-1])+square(mag_asinh_err[:,1:]))

    return c_[mag_asinh,color_asinh], c_[mag_asinh_err,color_asinh_err]








########## QUANTIFYING PERFORMANCE #############


def compute_score(y, pred, weights=None, eta=0.15):
    """
    Compute associated quality scores between two sets of matched redshift predictions.

    Keyword arguments:
    y -- input set of targets
    pred -- matching set of predictions
    weights -- weights for computing weighted statistics (default: None)
    eta -- threshold used for computing catastrophic outliers (|pred-y|/1+y > eta) (default: 0.15)
    
    Outputs:
    mean scores, median scores, f_cat
    mean scores -- mean(deltaz), std(deltaz), mean(deltaz_p), std(deltaz_p), where deltaz_p=deltaz/(1+y)
    median scores -- median(deltaz), MAD(deltaz; 1-sigma), median(deltaz_p), MAD(deltaz_p; 1-sigma)
    f_cat -- catastrophic outlier fraction
    """

    # initialize weights
    if weights is not None:
        weights=weights
    else:
        weights=ones(len(y))

    Nobj=sum(weights) # effective number of objects
    Ny=len(y) # total number of objects
    
    sig1=int(SIG1*Nobj) # defining 1-sigma boundary
    deltaz,deltaz_p=(pred-y),(pred-y)/(1+y) # defining offsets and (1+z)-normalized offsets

    # mean stats
    mean_dz,mean_dzp=average(deltaz,weights=weights),average(deltaz_p,weights=weights) # mean
    std_dz,std_dzp=sqrt(average((deltaz-mean_dz)**2,weights=weights)),sqrt(average((deltaz_p-mean_dzp)**2,weights=weights)) # scatter
    mean_stat=array([mean_dz,std_dz,mean_dzp,std_dzp]) # bundle scores
    
    # median stats
    s1,s2=argsort(deltaz),argsort(deltaz_p) # sorted list of deviations (low->high)
    cdf1,cdf2=cumsum(weights[s1]),cumsum(weights[s2]) # CDF of sorted weights
    med_dz,med_dzp=deltaz[s1][argmin(abs(cdf1-0.5*Nobj))],deltaz[s2][argmin(abs(cdf2-0.5*Nobj))] # median
    mad1,mad2=abs(deltaz-med_dz),abs(deltaz_p-med_dzp) # median absolute deviations (MADs)
    s1,s2=argsort(mad1),argsort(mad2) # sorted list of MADs
    cdf1,cdf2=cumsum(weights[s1]),cumsum(weights[s2]) # CDF of sorted weights
    mad_dz,mad_dzp=mad1[s1][argmin(abs(cdf1-SIG1*Nobj))],mad2[s2][argmin(abs(cdf2-SIG1*Nobj))] # scatter
    med_stat=array([med_dz,mad_dz,med_dzp,mad_dzp]) # bundle scores

    # compute R2 stats
    #r2_stat=1-sum(weights*deltaz**2)/sum(weights*(y-average(y,weights=weights))**2) # R^2 (correlation coefficient)

    # compute pearsonr stats
    #temp=[]
    #wtemp=weights/sum(weights) # normalize weights to 1
    #for i in xrange(10): # 10 Monte Carlo trials
    #    idx=choice(Ny,p=wtemp,size=int(Nobj)) # sample objects proportional to weights
    #    temp.append(stats.pearsonr(y[idx],pred[idx])) # compute Pearson R
    #pearsonr_coeff=average(temp,axis=0) # average Pearson R
    
    # catastrophic outliers
    eta_cat=sum(weights[abs(deltaz_p)>0.15])*1.0/Nobj
    
    return mean_stat, med_stat, eta_cat



def compute_density_score(dist_base, dist_test, dist_test_err=None):
    """
    Compute quality scores for two sets of matched densities.

    Keyword arguments:
    dist_base -- the baseline distribution to be compared to
    dist_test -- the new distribution to be tested

    Outputs:
    [N_poisson statistic, arr_poisson, sel_poisson], AD_statistic
    N_poisson -- the total Poisson deviation between the two datasets (i.e. chi2)
    arr_poisson -- standard deviations
    sel_poisson -- all non-zero terms used in the sum
    p_poisson -- p-value
    AD_statistic -- the Anderson-Darling 2-sample test statistics
    """

    a,b=dist_base.flatten(),dist_test.flatten()
    if dist_test_err is not None: be=dist_test_err.flatten() # flatten arrays to 1-D
    N,Na,Nb=len(a),a.sum(),b.sum()
    a_sub=a>0 # find all positive terms
    dof=sum(a_sub)
    dev=b-a    
    arr_poisson=(dev/sqrt(a))[a_sub] # compute normalized Poisson fluctuation
    N_poisson=sum(square(arr_poisson)) # total fluctuation
    p_poisson=1-stats.chi2.cdf(N_poisson,dof) # p-value

    a_samp=choice(N,size=int(Na),p=a/Na)
    b_samp=choice(N,size=int(Nb),p=b/Nb)

    try:
        N_ad=stats.anderson_ksamp((a_samp,b_samp)) # compute AD 2-sample test score
    except OverflowError:
        N_ad=(NaN,NaN,0.0)

    if dist_test_err is not None:
        arr_err=(dev/dist_test_err)[a_sub] # standardized fluctuation
        N_err=sum(square(arr_err)) # total fluctuation
        p_err=1-stats.chi2.cdf(N_err,dof) # p-value
        return [N_poisson, arr_poisson, a_sub, p_poisson], N_ad, [N_err, arr_err, a_sub, p_err]
    else:
        return [N_poisson, arr_poisson, a_sub, p_poisson], N_ad

    









########### PDF FUNCTIONS ###############


def pdfs_resample(target_grid, target_pdfs, new_grid):
    """
    Resample input PDFs from a given grid onto a new grid.

    Keyword arguments:
    target_grid -- original grid
    target_pdfs -- original collection of PDFs
    new_grid -- new grid

    Outputs:
    resampled PDFs
    """
    
    sys.stdout.write("Resampling PDFs...")
    
    Nobj,Npoints=len(target_pdfs),len(new_grid) # grab size of inputs
    new_pdfs=zeros((Nobj,Npoints)) # create new array
    for i in xrange(Nobj):
        if i%5000==0: sys.stdout.write(str(i)+" ")
        new_pdfs[i]=interp(new_grid,target_grid,target_pdfs[i]) # interpolate PDF
        new_pdfs[i]/=sum(new_pdfs[i]) # re-normalize
        
    sys.stdout.write("done!\n")

    return new_pdfs



def pdfs_summary_statistics(target_grid, target_pdfs, conf_width=0.03, deg_spline='linear'):
    """
    Compute a range of summary statistics from the input PDFs.

    Keyword arguments:
    target_grid -- input grid
    target_pdfs -- input collection of PDFs
    conf_width -- redshift range used to establish the 'zConf' flag (see Carrasco Kind & Brunner 2013) 
    deg_spline -- order of spline fit 

    Outputs:
    pdf_mean, pdf_med, pdf_mode, pdf_l68, pdf_h68, pdf_l95, pdf_h95, pdf_std, pdf_conf
    pdf_mean -- mean (first moment)
    pdf_med -- median (50th percentile)
    pdf_mode -- peak (mode)
    pdf_l68 -- lower 68th percentile of CDF
    pdf_h68 -- higher 68th percentile of CDP
    pdf_l95 -- lower 95th percentile of CDF
    pdf_h95 -- higher 95th percentile of CDF
    pdf_std -- standard deviation (sqrt of normalized second moment)
    pdf_conf -- zConf flag (see Carrasco Kind & Brunner 2013)
    """

    # initialize variables
    Ntest=len(target_pdfs) # number of input pdfs
    
    pdf_mean=zeros(Ntest,dtype='float32') # mean (first moment)
    pdf_std=zeros(Ntest,dtype='float32') # standard deviation (sqrt of normalized second moment)
    pdf_mode=zeros(Ntest,dtype='float32') # peak (mode)
    pdf_med=zeros(Ntest,dtype='float32') # median
    pdf_l95=zeros(Ntest,dtype='float32') # lower 95% confidence interval
    pdf_l68=zeros(Ntest,dtype='float32') # lower 68% confidence interval
    pdf_h68=zeros(Ntest,dtype='float32') # upper 68% confidence interval
    pdf_h95=zeros(Ntest,dtype='float32') # upper 95% confidence interval
    pdf_conf=zeros(Ntest,dtype='float32') # zConf flag


    # confidence intervals
    i1=0.68 # interval 1
    i2=0.95 # interval 2
    m=0.5 # median
    l2=m-i2/2 # lower 2
    l1=m-i1/2 # lower 1
    u1=m+i1/2 # upper 1
    u2=m+i2/2 # upper 2

    
    sys.stdout.write("Computing PDF quantities...")
    
    for i in xrange(Ntest):
        if i%5000==0: sys.stdout.write(str(i)+" ")
        
        # mean quantities
        pdf_mean[i]=dot(target_pdfs[i],target_grid)
        pdf_std[i]=sqrt(dot(target_pdfs[i],square(target_grid))-square(pdf_mean[i]))
        
        # mode quantities
        pdf_mode[i]=target_grid[argmax(target_pdfs[i])]
        
        # cumulative distribution function
        cdf=cumsum(target_pdfs[i]) # original CDF (normalized to 1)
        pdf_med[i],pdf_h68[i],pdf_l68[i],pdf_h95[i],pdf_l95[i]=interp([m,u1,l1,u2,l2],cdf,target_grid) # median quantities
        
        # confidence flag
        conf_range=conf_width*(1+pdf_med[i]) # redshift integration range
        conf_high,conf_low=interp([pdf_med[i]+conf_range,pdf_med[i]-conf_range],target_grid,cdf) # high/low CDF values
        pdf_conf[i]=conf_high-conf_low # zConf
        
    sys.stdout.write("done!\n")

    return pdf_mean, pdf_med, pdf_mode, pdf_l68, pdf_h68, pdf_l95, pdf_h95, pdf_std, pdf_conf











############# KERNAL DENSITY ESTIMATION ################


def gaussian(mu, var, x):
    """
    Compute (normalized) Gaussian kernal.

    Keyword arguments:
    mu -- mean (center)
    var -- variance (width)
    x -- input grid

    Outputs:
    N(x|mu,var)
    """
    
    dif=(x-mu) # difference
    norm=1./sqrt(2*pi*var) # normalization
    
    return norm*exp(-0.5*dif*dif/var)



def pdf_kde(y, y_var, y_wt, pdf_grid, delta_grid, Ngrid, wt_thresh=1e-3):
    """
    Compute smoothed PDF from point estimates using kernel density estimation (KDE) from a set of weighted predictions.

    Keyword arguments:
    y -- observed data
    y_var -- variance of observed data
    y_wt -- weight of observed data
    pdf_grid -- underlying grid used to compute the probability distribution function (PDF)
    delta_grid -- spacing of the grid
    Ngrid -- number of elements in the grid
    wt_thresh -- wt/wt_max threshold used to clip observations (default: 1e-3)

    Outputs:
    Probability distribution function (PDF) evaluated over pdf_grid
    """

    # clipping kernel to (gridded) +/-5 sigma
    centers=((y-pdf_grid[0])/delta_grid).astype(int) # gridded centers
    offsets=(5*sqrt(y_var)/delta_grid).astype(int) # gridded 5-sigma offsets
    uppers,lowers=centers+offsets,centers-offsets # upper/lower bounds
    uppers[uppers>Ngrid],lowers[lowers<0]=Ngrid,0 # limiting to grid edges

    # initialize PDF
    pdf=zeros(Ngrid)

    # limit analysis to observations with statistically relevant weight
    sel_arr=y_wt>(wt_thresh*max(y_wt))

    # compute PDF
    for i in arange(len(y_wt))[sel_arr]:
        pdf[lowers[i]:uppers[i]]+=y_wt[i]*gaussian(y[i],y_var[i],pdf_grid[lowers[i]:uppers[i]]) # stack (weighted) Gaussian kernels on array segments
    
    return pdf



# Compute smoothed PDF using kernel density estimation (KDE) from a set of WEIGHTED predictions
def pdf_kde_dict(ydict, ywidth, y_pos, y_idx, y_wt, pdf_grid, delta_grid, Ngrid, wt_thresh=1e-3):
    """
    Compute smoothed PDF from point estimates using kernel density estimation (KDE) from a set of weighted predictions using a PRE-COMPUTED DICTIONARY.

    Keyword arguments:
    ydict -- dictionary of kernels
    ywidth -- associated widths of kernels
    y_pos -- discretized position of observed data
    y_idx -- corresponding index of kernel drawn from dictionary
    y_wt -- weight of observed data
    pdf_grid -- underlying grid used to compute the probability distribution function (PDF)
    delta_grid -- spacing of the grid
    Ngrid -- number of elements in the grid
    wt_thresh -- wt/wt_max threshold used to clip observations (default: 1e-3)

    Outputs:
    Probability distribution function (PDF) evaluated over pdf_grid
    """

    # initialize PDF
    pdf=zeros(Ngrid) 

    # limit analysis to observations with statistically relevant weight
    sel_arr=y_wt>(wt_thresh*max(y_wt))

    # compute PDF by stacking kernels
    for i in arange(len(y_idx))[sel_arr]: # run over selected observations
        idx,yp=y_idx[i],y_pos[i] # dictionary element, kernel center
        yw=ywidth[idx] # kernel width
        pdf[yp-yw:yp+yw+1]+=y_wt[i]*ydict[idx] # stack weighted kernel from dictionary on array slice
    
    return pdf







    


########### WINBET ###############


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
            if obj%500==0: sys.stdout.write(str(obj)+' ')

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
    The Full Regression over Associated Neighbors using Kernel dEsity estimatioN for Redshifts (FRANKEN-Z) class. 

    Functions: 
    predict -- generates redshift PDF predictions
    """
    
    def __init__(self, N_members=100, n_neighbors=10, radius=1.0, algorithm='kd_tree', leaf_size=50, p=2, metric='minkowski', metric_params=None, n_jobs=1):
        """
        Initializes the instance. 

        Keyword arguments:
        N_members -- number of members in ensemble
        See sklearn.neighbors.NearestNeighbors for a description of additional keyword arguments and their defaults.
        """

        # establish baseline model
        self.knn=neighbors.NearestNeighbors(n_neighbors=n_neighbors,radius=radius,algorithm=algorithm,leaf_size=leaf_size,p=p,metric=metric,metric_params=metric_params,n_jobs=1)
        self.NNEIGHBORS=n_neighbors # number of nearest neighbors
        self.metric=metric # chosen distance metric
        self.NBRUTEFORCE=50 # leaf size (for brute force calculation)
        self.NMEMBERS=N_members # ensemble size



    def predict(self, phot, err, masks, phot_test, err_test, masks_test, f_func=asinh_mag_map, ll_func=loglikelihood, impute_train=None, impute_test=None, subsample=None):
        """
        Generate Full Regression over Associated Neighbors with Kernel dENsity redshift (FRANKEN-Z) PDFs.
        Objects in the training set to be fit (i.e. regressed over) are selected from kd-trees based on a set of transformed FEATURES.
        Log-likelihoods are computed directly from the corresponding FLUXES using the input log-likelihood function.
        Errors on the TRAINING AND TESTING data are incorporated using Monte Carlo methods.

        Keyword arguments:
        phot[_test] -- fluxes (training/testing)
        err[_test] -- flux errors (training/testing)
        masks[_test] -- flux masks (training/testing)        
        f_func -- feature function (default: asinh_magnitude).
        ll_func -- log-likelihood function (default: loglikelihood)
        impute[_train/test] -- WINBET instances used for imputing missing fluxes (default: None)
        subsample -- number of dimensions to subsample (default: None)

        Outputs:
        model_objects -- unique matched object indices
        model_Nobj -- number of unique matched objects
        model_ll -- log-likelihoods
        model_Nbands -- number of bands used to compute log-likelihoods
        """

        # initialize stuff
        Ntrain,Ntest=len(phot),len(phot_test) # size of training/testing sets
        Npred=self.NNEIGHBORS*self.NMEMBERS # number of non-unique predictions
        Nf=len(phot[0]) # number of filters
        model_objects=empty((Ntest,Npred),dtype='int') # UNIQUE collection of training object indices selected for each test object
        model_Nobj=empty(Ntest,dtype='int') # number of unique training objects selected for each test object
        model_indices=empty((self.NMEMBERS,Ntest,self.NNEIGHBORS),dtype='int') # NON-UNIQUE collection of training object indices selected for each test object
        model_ll=empty((Ntest,Npred),dtype='float32') # log-likelihood
        model_Nbands=empty((Ntest,Npred),dtype='uint8') # number of bands used in fit

        var,var_test=square(err),square(err_test)
        skynoise=median(err,axis=0)

        if subsample is None:
            subsample=Nf

        # find nearest neighbors
        for i in xrange(self.NMEMBERS):
            sys.stdout.write(str(i)+' ')

            xdim=choice(Nf,size=subsample,replace=False)

            # train kd-trees
            if impute_train is not None:
                phot_t,var_t=impute_train.impute(phot,var,masks,impute_type='random')
                phot_t=normal(phot_t,sqrt(var_t))
            else:
                phot_t=normal(phot,err).astype('float32') # perturb fluxes
            X_t=f_func(phot_t,err,skynoise)[0] # map to feature space
            knn=base.clone(self.knn).fit(X_t[:,xdim]) # train kd-tree

            # query kd-trees
            if impute_test is not None:
                phot_test_t,var_test_t=impute_test.impute(phot_test,var_test,masks_test,impute_type='random')
                phot_test_t=normal(phot_test_t,sqrt(var_test_t))
            else:
                phot_test_t=normal(phot_test,err_test).astype('float32') # perturb fluxes
            X_test_t=f_func(phot_test_t,err_test,skynoise)[0] # map to feature space
            model_indices[i]=knn.kneighbors(X_test_t[:,xdim],return_distance=False) # find neighbors

        # select/compute log-likelihoods to unique subset of neighbors
        for i in xrange(Ntest):
            midx_unique=pandas.unique(model_indices[:,i,:].flatten()) # unique indices
            Nidx=len(midx_unique) # number of unique indices
            model_objects[i][:Nidx]=midx_unique
            model_Nobj[i]=Nidx

            # compute log-likelihoods and Nbands
            model_ll[i][:Nidx],model_Nbands[i][:Nidx]=ll_func(phot_test[i],var_test[i],masks_test[i],phot[midx_unique],var[midx_unique],masks[midx_unique])

            if i%5000==0: 
                sys.stdout.write(str(i)+' ') # counter
                gc.collect() # garbage collect
    
        sys.stdout.write('done!\n')

        return model_objects,model_Nobj,model_ll,model_Nbands





    



        
        


########### INPUT/OUTPUT OPERATIONS ###############


class ReadParams():
    """
    Read in configuration files and initialize parameters. [Code based on Gabriel Brammer's threedhst.eazyPy module.]
    """

    def __init__(self, config_file):

        self.filename = config_file # filename

        # read in file
        f=open(config_file,'r')
        self.lines=f.readlines()
        f.close()

        # process file
        self._process_params()

        # process additional configuration files
        for param in self.param_names:
            if 'CONFIG_' in param:
                fname=self.params[param] # grab filename
                exec("self."+param+"=ReadParams(self.params['HOME']+fname)") # assign ReadParams output to associated variable name


    def _process_params(self):
        """
        Process input parameters and add them to the class dictionary.
        """

        params={} # parameters
        formats={} # format of parameters
        self.param_names=[] # parameter names

        # extract parameters
        for line in self.lines:
            if (line.startswith('#') | line.startswith(' ')) is False:

                # split line
                lsplit=line.split()

                # assign name and parameter
                if len(lsplit)>=2:
                    lsplit[0]=lsplit[0][:-1]
                    params[lsplit[0]]=lsplit[1]
                    self.param_names.append(lsplit[0])

                    # (re)assign formats
                    try:
                        flt=float(lsplit[1])
                        formats[lsplit[0]]='f'
                        params[lsplit[0]]=flt
                    except:
                        formats[lsplit[0]]='s'

                    if params[lsplit[0]] == 'None':
                        params[lsplit[0]] = None
                        formats[lsplit[0]] = 'n'

        self.params=params
        self.formats=formats



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

        c=299792458.0 # speed of light in m/s
        
        f=open(filter_list)
        self.filters=[]
        self.filenames=[]
        for line in f:
            lsplit = line.split()
            self.filters.append(lsplit[0])
            self.filenames.append(lsplit[1])
        f.close()

        self.NFILTER=len(self.filters)
        
        self.fw=[0.]*self.NFILTER
        self.ft=[0.]*self.NFILTER

        for i in xrange(self.NFILTER):
            self.fw[i],self.ft[i]=swapaxes(loadtxt(path+self.filenames[i]),0,1)

        self.lambda_eff=zeros(self.NFILTER)

        for i in xrange(self.NFILTER):
            nuMax=0.999*c/(min(self.fw[i])*1e-10) # max frequency
            nuMin=1.001*c/(max(self.fw[i])*1e-10) # min frequency
            nuInc=(nuMax-nuMin)/Npoints # increment (linear)
            nu=arange(nuMin,nuMax+nuInc,nuInc) # frequency array
            lnu=log(nu)
            wave=c/nu # convert to lambda
            lwave=log(wave)

            func=interpolate.interp1d(self.fw[i],self.ft[i],kind='linear') # spline filter
            temp=func(1e10*wave) # transmission (in Angstroms)

            top=trapz(temp*lwave,lnu)
            bottom=trapz(temp,lnu)
            self.lambda_eff[i]=exp(top/bottom)*1e10 # effective wavelength of filter












        
    
        

########### DICTIONARIES ###############


class RedshiftDict():
    """
    Set up redshift grids and kernels used for computations.
    """

    
    def __init__(self, rparams, sigma_trunc=5.0):
        """
        Set up redshift grids/kernels.

        Keyword arguments:
        rparams -- redshift configuration parameters (see class::ReadParams)
        sigma_trunc -- number of standard deviations used before truncating the kernels (default: 5)
        """

        # discrete kernel parameters
        self.lze_grid=linspace(rparams['DLZ'],rparams['DLZ_MAX'],int(rparams['N_DICT'])) # Gaussian dictionary parameter grid
        self.dlze=self.lze_grid[1]-self.lze_grid[0] # kernel spacing

        # high-res log(1+z) grid
        self.res=rparams['RES']
        self.dlz_highres=rparams['DLZ']/rparams['RES'] # high-res spacing
        self.Npad=int(sigma_trunc*rparams['DLZ_MAX']/self.dlz_highres) # padding on ends of grid (for sliding addition)
        self.lzgrid_highres=arange(log(1+rparams['ZMIN']),log(1+rparams['ZMAX'])+self.dlz_highres,self.dlz_highres) # high-res grid
        self.lzgrid_highres=append(arange(log(1+rparams['ZMIN'])-self.dlz_highres*self.Npad,log(1+rparams['ZMIN']),self.dlz_highres),self.lzgrid_highres) # left-pad
        self.lzgrid_highres=append(self.lzgrid_highres,arange(log(1+rparams['ZMAX'])+self.dlz_highres,log(1+rparams['ZMAX'])+self.dlz_highres*(self.Npad+1),self.dlz_highres)) # right-pad
        self.Nz_highres=len(self.lzgrid_highres) # size of grid

        # effective bounds of high-res grid
        self.zmin_idx_highres=argmin(abs(self.lzgrid_highres-log(1+rparams['ZMIN']))) # minimum
        self.zmax_idx_highres=argmin(abs(self.lzgrid_highres-log(1+rparams['ZMAX']))) # maximum
        self.zmax_idx_highres=self.zmin_idx_highres+int(ceil((self.zmax_idx_highres-self.zmin_idx_highres)/rparams['RES'])*rparams['RES']) # adding left-pad and offset

        # lower-res log(1+z) grid
        self.dlz=rparams['DLZ'] # dlog(1+z)
        self.lzgrid=arange(self.lzgrid_highres[self.zmin_idx_highres],self.lzgrid_highres[self.zmax_idx_highres],self.dlz) # log(1+z) grid
        self.Nz=len(self.lzgrid) # number of elements

        # corresponding redshift grids
        self.zgrid=exp(self.lzgrid)-1 # low-res z grid
        self.znorm=(self.zgrid[1]-self.zgrid[0])*self.zgrid+(self.zgrid[1]-self.zgrid[0]) # normalizations (for conversions from log(1+z) to z)
        self.znorm/=self.znorm[0]

        # create dictionary
        self.lze_width=ceil(self.lze_grid*sigma_trunc/self.dlz_highres).astype('int') # width of kernel
        self.lze_dict=[gaussian(self.lzgrid_highres[self.Nz_highres/2],square(self.lze_grid[i]),self.lzgrid_highres[self.Nz_highres/2-self.lze_width[i]:self.Nz_highres/2+self.lze_width[i]+1])
                       for i in xrange(int(rparams['N_DICT']))] # dictionary
        self.Ndict=len(self.lze_dict) # number of dictionary elements
        
        # output redshift grid
        self.zgrid_out=arange(rparams['ZMIN_OUT'],rparams['ZMAX_OUT']+rparams['DZ_OUT'],rparams['DZ_OUT']) # output z grid
        self.dz_out=rparams['DZ_OUT'] # output dz
        self.dz_out_highres=rparams['DZ_OUT']/rparams['RES_OUT'] # output high-resolution dz
        self.Nz_out=len(self.zgrid_out) # number of elements
        

    def fit(self, lz, lze):
        """
        Map Gaussian redshift PDFs onto the log(1+z) dictionary.

        Keyword arguments:
        lz -- log(1+z) means
        lze -- log(1+z) errors (i.e. z/(1+z))

        Outputs:
        lz_idx -- corresponding dictionary grid indices (high-resolution)
        lze_idx -- corresponding dictionary kernel indices (high-resolution)
        """
        lz_idx=((lz-self.lzgrid_highres[0])/self.dlz_highres).round().astype('int')
        lze_idx=((lze-self.lze_grid[0])/self.dlze).round().astype('int')

        return lz_idx,lze_idx


        
class PDFDict():
    """
    Set up underlying grids and kernsl used to compute PDFs for ancillary parameters.
    """

    
    def __init__(self, pparams, sigma_trunc=5.0):
        """
        Keyword arguments:
        pparams -- configuration parameters for the PDF file (see class:ReadParams)
        """

        # initialize grid
        self.delta=pparams['DELTA'] # grid spacing
        self.min=pparams['MIN'] # grid lower bound
        self.max=pparams['MAX'] # grid upper bound
        self.grid=arange(self.min,self.max+self.delta/2,self.delta) # grid
        self.Ngrid=len(self.grid) # number of elements
        
        # create dictionary
        self.sig_grid=linspace(pparams['SIG_MIN'],pparams['SIG_MAX'],int(pparams['N_DICT'])) # Gaussian dictionary parameter grid
        self.dsig=self.sig_grid[1]-self.sig_grid[0] # kernel spacing
        self.sig_width=ceil(self.sig_grid*sigma_trunc/self.delta).astype('int') # width of kernel
        self.sig_dict=[gaussian(self.grid[self.Ngrid/2],square(self.sig_grid[i]),self.grid[self.Ngrid/2-self.sig_width[i]:self.Ngrid/2+self.sig_width[i]+1])
                       for i in xrange(int(pparams['N_DICT']))] # dictionary


    def fit(self, X, Xe):
        """
        Map Gaussian PDFs onto the dictionary.

        Keyword arguments:
        X -- target features
        Xe -- target feature errors

        Outputs:
        X_idx -- corresponding dictionary grid indices
        Xe_idx -- corresponding dictionary kernel indices
        """
        X_idx=((X-self.grid[0])/self.delta).round().astype('int')
        Xe_idx=((Xe-self.sig_grid[0])/self.dsig).round().astype('int')
        Xe_idx[Xe_idx>=len(self.sig_grid)]=len(self.sig_grid)-1 # impose error ceiling
        Xe_idx[Xe_idx<0]=0. # impose error floor

        return X_idx,Xe_idx











################ PLOTTING ################


def plot_nz(train_nz,out_nz,zgrid,deltaz,zrange=[0,6],out_nz_draws=None,sample_names=['True','Predicted'],colors=['black','red']):
    """
    Plot comparison between two N(z) (or two general number density) distributions.

    Keyword arguments:
    sample_names -- names for each sample
    train_nz -- original N(z)
    out_nz -- comparison N(z)
    zgrid -- grid N(z) is evaluated on
    deltaz -- dz spacing
    zrange -- plotting range

    Outputs:
    Identical to func::compute_density_score.
    """

    # compute errors from draws (if exists)
    if out_nz_draws is not None:
        # compute density scores
        out_nz_err=std(out_nz_draws,axis=0) # naive sigma
        [N_p,arr_p,z_sel,prob_p],N_ad,[N_e,arr_e,z_sel,prob_e]=compute_density_score(train_nz,out_nz,out_nz_err)
        ad_s,ad_p=N_ad[0],N_ad[2] # Anderson-Darling statistic and probability
        if ad_p>1: ad_p=0.

    else:
        # compute density scores
        [N_p,arr_p,z_sel,prob_p],N_ad=compute_density_score(train_nz,out_nz)
        ad_s,ad_p=N_ad[0],N_ad[2] # Anderson-Darling statistic and probability
        if ad_p>1: ad_p=0.
    
    # initializing figure
    gs=gridspec.GridSpec(2,1,height_ratios=[4,1])

    # plot N(z)
    subplot(gs[0])
    plot(zgrid,train_nz/deltaz,color=colors[0],lw=3,label=sample_names[0])
    if out_nz_draws is not None:
        for draw in out_nz_draws:
            plot(zgrid,draw/deltaz,color=colors[1],lw=0.2,alpha=0.3)
    plot(zgrid,out_nz/deltaz,color=colors[1],lw=3,label=sample_names[1])
    fill_between(zgrid,train_nz/deltaz,out_nz/deltaz,color='yellow')
    yscale('log',noposy='clip')
    xlim(zrange)
    ylims=[1,max([max(train_nz/deltaz),max(out_nz/deltaz)])*1.5]
    log_ylims=log10(ylims)
    ylim(ylims)
    legend(fontsize=24)
    xlabel('Redshift')
    ylabel('$dN/dz$')
    text(zrange[0]+(zrange[1]-zrange[0])*0.05,10**(log_ylims[0]+(log_ylims[1]-log_ylims[0])*0.18),'Pois$(S/n,p)$=('+str(round(N_p/sum(z_sel),2))+','+str(round(prob_p,2))+')')
    text(zrange[0]+(zrange[1]-zrange[0])*0.05,10**(log_ylims[0]+(log_ylims[1]-log_ylims[0])*0.09),'AD$(S,p)$=('+str(round(ad_s,2))+','+str(round(ad_p,2))+')')
    if out_nz_draws is not None:
        text(zrange[0]+(zrange[1]-zrange[0])*0.05,10**(log_ylims[0]+(log_ylims[1]-log_ylims[0])*0.27),'Error$(S/n,p)$=('+str(round(N_e/sum(z_sel),2))+','+str(round(prob_e,2))+')')
    tight_layout()

    # plot running Poisson/error fluctuation
    
    subplot(gs[1])
    xlim(zrange)
    xlabel('Redshift')
    ylabel('$\Delta \sigma$')
    plot(zgrid,zeros(len(zgrid)),'k--',lw=2)
    fill_between(zgrid[z_sel],arr_p,color='yellow',alpha=0.7)
    plot(zgrid[z_sel],arr_p,lw=2,color=colors[0])
    ymin,ymax=round(min(arr_p),3),round(max(arr_p),3)
    yticks([ymin,ymax],fontsize=24)

    if out_nz_draws is not None:
        fill_between(zgrid[z_sel],arr_e,color='orange',alpha=0.7)
        plot(zgrid[z_sel],arr_e,lw=2,color=colors[1])
        ymin,ymax=min(round(min(arr_e),3),ymin),max(round(max(arr_e),3),ymax)
        yticks([ymin,ymax],fontsize=24)

    tight_layout()

    if out_nz_draws is not None:
        return [N_p,arr_p,z_sel,prob_p], N_ad, [N_e,arr_e,z_sel,prob_e]
    else:
        return [N_p,arr_p,z_sel,prob_p], N_ad


def plot_zpoints(plot_title, y, yp, markersize=1.5, limits=[0,6], binwidth=0.05, thresh=10, selection=None, weights=None):
    """
    Plot results from redshift POINT ESTIMATES. To illustrate density scales, 2-D density histograms are used for the majority of the data, while outlying points are plotted individually.

    Keyword arguments:
    plot_title -- plot title
    y -- input values
    yp -- predicted values
    markersize -- size of outlying points
    limits -- scale of x,y axes
    binwidth -- width of 2-D histogram bins
    thresh -- threshold before switching from histogram to outlying points
    selection -- selection array for plotting a subset of objects

    Outputs:
    Identical to func::compute_score.
    """

    cmap=get_cmap('jet')
    cmap.set_bad('white')

    success_sel=isfinite(yp)&(yp>0)

    if weights is not None:
        weights=weights
    else:
        weights=ones(len(y))
    
    if selection is not None:
        sel=success_sel&selection&(weights>0.)
    else:
        sel=success_sel&(weights>0.)

    score=compute_score(y[sel],yp[sel],weights=weights[sel])

    # declare binning parameters
    xyrange=[[0,10],[0,10]]
    bins=[arange(xyrange[0][0],xyrange[0][1]+binwidth,binwidth),arange(xyrange[1][0],xyrange[1][1]+binwidth,binwidth)]
    
    # bin data
    xdat,ydat=y[sel],yp[sel]
    hh,locx,locy=histogram2d(xdat,ydat,range=xyrange,bins=bins,weights=weights[sel])
    posx=digitize(xdat,locx)
    posy=digitize(ydat,locy)

    #select points within the histogram
    hhsub=hh[posx-1,posy-1] # values of the histogram where the points are
    xdat1=xdat[(hhsub<thresh)] # low density points (x)
    ydat1=ydat[(hhsub<thresh)] # low density points (y)
    hh[hh<thresh]=NaN # fill the areas with low density by NaNs

    # plot results
    plot(xdat1, ydat1,'.',color='black',markersize=markersize) # outliers/low-density regions
    imshow(flipud(hh.T),cmap='jet',extent=array(xyrange).flatten(),interpolation='none',norm=matplotlib.colors.LogNorm()) # high-density regions

    # establishing the colorbar
    cbar=colorbar()
    cticks=arange(ceil(log10(nanmin(hh.flatten()))/0.25)*0.25,int(log10(nanmax(hh.flatten()))/0.25)*0.25+1e-6,0.25)
    cticklabels=['$10^{'+str(round(i,2))+'}$' for i in cticks]
    cbar.set_ticks(10**cticks)
    cbar.set_ticklabels(cticklabels)

    # plotting 1:1 line+bounds
    plot(array([0,100]),array([0,100]),'k--',lw=3)
    plot(array([0,100]),array([0,100])*1.15+0.15,'k-.',lw=2)
    plot(array([0,100]),array([0,100])*0.85-0.15,'k-.',lw=2)
    title(plot_title,y=1.02)

    # statistics
    Nobj=sum(weights[sel])
    text(1.2*(limits[1]/5.0),4.7*(limits[1]/5.0),"$N$: "+str(int(Nobj))+" ("+str(round(Nobj*1.0/sum(weights[success_sel]),3))+")",fontsize=18,color='black')
    text(1.2*(limits[1]/5.0),4.5*(limits[1]/5.0),"$\Delta z^\prime$ (mean): "+str(round(score[0][2],4)*100)+"%",fontsize=18,color='black')
    text(1.2*(limits[1]/5.0),4.3*(limits[1]/5.0),"$\Delta z^\prime$ (med): "+str(round(score[1][2],4)*100)+"%",fontsize=18,color='black')
    text(1.2*(limits[1]/5.0),4.1*(limits[1]/5.0),"$\sigma_{\Delta z^\prime}$ (MAD): "+str(round(score[1][3],4)*100)+"%",fontsize=18,color='black')
    text(1.2*(limits[1]/5.0),3.9*(limits[1]/5.0),"$f_{cat}$: "+str(round(score[2],4)*100)+"%",fontsize=18,color='black')

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
    temp_stack=temp_stack[rdict.zmin_idx_highres:rdict.zmax_idx_highres:int(rdict.res)]/rdict.znorm[:,None] # reducing resolution
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


