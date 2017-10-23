#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FRANKEN-Z package.

"""

from __future__ import (print_function, division)
import six
from six.moves import range
import numpy as np
import scipy
import pandas
import sys
import os
from scipy.spatial import KDTree
from scipy import stats
import gc

from .pdf import *

__all__ = [""]

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


####################################################################################################


########## WINBET ##########

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
        N_leaf -- minimum number of samples in a leaf
        """

        # establish baseline model
        self.NTREES = Ntrees # number of trees
        self.NLEAF = Nleaf # minimum number of objects per leaf
        self.lf = [tree.ExtraTreeRegressor(min_samples_leaf=self.NLEAF) for i in xrange(self.NTREES)]
        self.lf_idx = [[] for i in xrange(self.NTREES)]

    def train(self, x, xe, xm, feature_map=asinh_mag_map):
        """
        Train underlying Extra Trees using Naive Bayes for guesses.

        Keyword arguments:
        x -- features
        xe -- feature errors
        xm -- feature mask
        feature_map -- feature transformation map (default=asinh_mag_map)
        """

        # initialize stuff
        self.NOBJ, self.NDIM = x.shape # number of objects and feature dimensions
        censor_sel = arange(self.NOBJ)[xm.sum(axis=1) < self.NDIM] # indices of objects with censored (missing) data
        NCENSOR = len(censor_sel) # number of censored objects
        csel = [(xm[:,i]==False) for i in xrange(self.NDIM)] # mask slice in relevant dimension
        NFILL = [csel[i].sum() for i in xrange(self.NDIM)] # number of censored values to be filled in

        self.x, self.xe, self.xm = x.copy(), xe.copy(), xm.copy() # copy data
        self.feature_map = feature_map
    
        # gather neighbors
        for counter in xrange(self.NTREES):
            sys.stdout.write(str(counter)+' ')
            sys.stdout.flush()

            # generate new values
            x_t = normal(x, xe) # jitter values
            X_t, Xe_t = self.feature_map(x_t, xe) # transform features
            for i in xrange(self.NDIM):
                cdf_x, cdf_y = linspace(0, 1, self.NOBJ-NFILL[i]), sort(X_t[:,i][csel[i]==False]) # compute marginal CDF
                X_t[csel[i],i] = interp(rand(NFILL[i]), cdf_x, cdf_y) # fill in values by sampling from CDF

            # train tree
            self.lf[counter].fit(X_t, X_t) # fit data

            # map indices
            idx = self.lf[counter].apply(X_t) # map leaf indices
            self.lf_idx[counter] = [[] for i in xrange(max(idx)+1)] # tree-structured object list
            for i in xrange(len(idx)):
                self.lf_idx[counter][idx[i]].append(i) # add object to tree-indexed list


    def impute(self, y, ye, ym):
        """
        Impute missing photometry.

        Keyword arguments:
        y -- new input values
        ye -- new input errors
        ym -- new input masks

        Outputs:
        y_impute -- imputed value
        ye_impute -- imputed error
        [rand_idx] -- selected training object indices
        """

        # initialize stuff
        Ny = len(y) # number of objects 
        sel = arange(Ny)[ym.sum(axis=1) < self.NDIM] # indices of objects with censored (missing) data
        Nsel = len(sel) # number of censored objects
        y_csel = [(ym[:,i]==False) for i in xrange(self.NDIM)] # mask slice in relevant dimension
        Nfill = [y_csel[i].sum() for i in xrange(self.NDIM)] # number of censored values to be filled in

        model_idx = [[] for i in xrange(Nsel)] # model indices

        # gather neighbors
        for counter in xrange(self.NTREES):

            Y_t, Ye_t = self.feature_map(y, ye) # transform features
            for i in xrange(self.NDIM):
                cdf_x, cdf_y = linspace(0, 1, Ny-Nfill[i]), sort(Y_t[:,i][y_csel[i]==False]) # compute marginal CDF
                Y_t[y_csel[i],i] = interp(rand(Nfill[i]), cdf_x, cdf_y) # fill in values by sampling from CDF

            # query tree
            tidx = self.lf[counter].apply(Y_t[sel]) 
            for i in xrange(Nsel):
                for j in self.lf_idx[counter][tidx[i]]:
                    model_idx[i].append(j) # add leaf neighbors to object-indexed list
    
        # select random object
        rand_idx = empty(Nsel, dtype='int')
        for i in xrange(Nsel):
            rand_idx[i] = choice(model_idx[i])
        
        # jitter values
        y_impute, ye_impute = y.copy(), ye.copy()
        y_impute[ym==False] = self.x[rand_idx][ym[sel]==False]
        ye_impute[ym==False] = self.xe[rand_idx][ym[sel]==False]

        return y_impute, ye_impute



########### FRANKEN-Z ###############









################################################################################





########## PLOTTING UTILITIES ##########


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


def plot_zpoints(y, yp, markersize=1.5, limits=[0,6], binwidth=0.05, thresh=10, cat_thresh=0.15, selection=None, weights=None):
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


