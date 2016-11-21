#!/usr/bin/env python

##### SETUP #####

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
from scipy import stats # statistics
from scipy import special # special functions
from astropy.io import fits # reading fits files
import os,sys,argparse # OS operations
import gc # garbage cleanup

# FRANKEN-Z
import frankenz as fz


#############################
##### MAIN BODY OF CODE #####
#############################

def main(config_file, target_data, target_name, output_location, N_members=25, N_neighbors=10):


    ###################################
    ##### INITIALIZING PARAMETERS #####
    ###################################

    ##### CONFIG #####
    sys.stderr.write('Reading inputs...')

    # read master config file
    config=fz.ReadParams(config_file) 

    # import filters
    filt=fz.ReadFilters(config.params['FILTERS'],path=config.params['FILTER_PATH'])
    Nf=filt.NFILTER

    # initialize redshift dictionary
    rdict=fz.RedshiftDict(config.CONFIG_REDSHIFT.params)

    # add names so easier to reference later
    rdict.sig_dict=rdict.lze_dict
    rdict.sig_width=rdict.lze_width
    rdict.delta=rdict.dlz_highres
    rdict.grid=rdict.lzgrid_highres
    rdict.Ngrid=rdict.Nz_highres

    # initialize supplementary dictionaries
    mdict=fz.PDFDict(config.CONFIG_MAG.params) # magnitude
    cdict=fz.PDFDict(config.CONFIG_COLOR.params) # color
    
    sys.stderr.write('done!\n')




    ####################################
    ##### EXTRACTING TRAINING DATA #####
    ####################################

    sys.stderr.write('Reading training data...')

    # flag_class: 1=spec-z, 2=grism-z, 3=photo-z
    # flag_survey: SDSS=1, DEEP2=2, PRIMUS=3, VIPERS=4, VVDS=5, GAMA=6, WIGGLEZ=7, COSMOS=8, UDSZ=9, 3DHST=10, FMOS-COSMOS=11

    # load observed data
    hdul=fits.open('/media/joshspeagle/OSDisk/Users/Josh/Dropbox/HSC/HSC_photoz/catalogs/hsc_s16a_combined_specz_highq_clean_errsim_train_v1.fits') # training data location
    data=hdul[1].data
    Ntrain=len(data)

    # load dust corrections
    aphot=10**(-0.4*c_[data['a_g'],data['a_r'],data['a_i'],data['a_z'],data['a_y']])

    # load psf-matched aperture fluxes ("afterburner" fluxes)
    flux_afterburner=c_[data['gparent_flux_convolved_2_1'],data['rparent_flux_convolved_2_1'],data['iparent_flux_convolved_2_1'],
                        data['zparent_flux_convolved_2_1'],data['yparent_flux_convolved_2_1']]*aphot
    err_afterburner=c_[data['gflux_aperture15_err'],data['rflux_aperture15_err'],data['iflux_aperture15_err'],
                       data['zflux_aperture15_err'],data['yflux_aperture15_err']]*aphot
    mask_afterburner=(err_afterburner>0.)&isfinite(err_afterburner)
    mask_afterburner*=(flux_afterburner!=0.)&isfinite(flux_afterburner)

    # convert to Luptitudes
    flux_zeropoint=10**(-0.4*-23.9) # AB magnitude zeropoint
    skynoise=median(err_afterburner,axis=0) # "background" skynoise (used for consistent mappings)
    mag_afterburner,magerr_afterburner=fz.asinh_mag_map(flux_afterburner,err_afterburner,skynoise,zeropoint=flux_zeropoint) # Luptitude mapping
    
    # load redshifts
    z,ze,zt,zs=data['redshift'],data['redshift_err'],data['redshift_type'],data['redshift_source'] # z, z_err, z_type, z_survey
    Nzt=len(unique(zt)) # number of unique types
    Nzs=len(unique(zs)) # number of unique surveys
    ze[ze<0]=0. # set spec-z errors to be 0

    # discretize mappings
    lzidx,lzeidx=rdict.fit(log(1+z),ze/(1+z)) # discretize redshifts
    magidx,mageidx=mdict.fit(mag_afterburner,magerr_afterburner) # discretize magnitudes

    # clearing up memory
    del hdul[1].data,data
    for hdu in hdul:
        del hdu
    del hdul
    
    sys.stderr.write('done!\n')



    ###################################
    ##### EXTRACTING TESTING DATA #####
    ###################################

    sys.stderr.write('Reading target data...')

    # load observed data
    hdul=fits.open(target_data) # target data location
    data=hdul[1].data
    Ntrain=len(data)

    # load identifying information
    objid=data['object_id']
    tract,patch=data['tract'],data['patch']

    # load dust corrections
    aphot_test=10**(-0.4*c_[data['a_g'],data['a_r'],data['a_i'],data['a_z'],data['a_y']])

    # load psf-matched aperture fluxes ("afterburner" fluxes)
    flux_afterburner_test=c_[data['gparent_flux_convolved_2_1'],data['rparent_flux_convolved_2_1'],
                             data['iparent_flux_convolved_2_1'],data['zparent_flux_convolved_2_1'],
                             data['yparent_flux_convolved_2_1']]*aphot_test*1e29
    err_afterburner_test=c_[data['gflux_aperture15_err'],data['rflux_aperture15_err'],data['iflux_aperture15_err'],
                            data['zflux_aperture15_err'],data['yflux_aperture15_err']]*aphot_test*1e29
    mask_afterburner_test=(err_afterburner_test>0.)&isfinite(err_afterburner_test)
    mask_afterburner_test*=(flux_afterburner_test!=0.)&isfinite(flux_afterburner_test)

    # selecting objects with at least one band of data
    flux_sel=mask_afterburner_test.sum(axis=1)>0
    Nobs=flux_sel.sum() # number of objects

    # convert to Luptitudes
    mag_afterburner_test,magerr_afterburner_test=fz.asinh_mag_map(flux_afterburner_test,err_afterburner_test,
                                                                  skynoise,zeropoint=flux_zeropoint)
    magidx_test,mageidx_test=mdict.fit(mag_afterburner_test,magerr_afterburner_test) # discretize magnitudes

    # clearing up memory
    del hdul[1].data,data
    for hdu in hdul:
        del hdu
    del hdul
    
    sys.stderr.write('done!\n')



    #######################################
    ##### INITIALIZE WINBET INSTANCES #####
    #######################################

    # redefining fluxes
    p1,v1,m1=flux_afterburner,square(err_afterburner),mask_afterburner # training
    p1[m1==False],v1[m1==False]=1.,1. # filling in missing values with arbitrary values
    p2,v2,m2=flux_afterburner_test,square(err_afterburner_test),mask_afterburner_test # testing (observed)
    p2[m2==False],v2[m2==False]=1.,1. # filling in missing values with arbitrary values
    e1,e2=sqrt(v1+square(0.01*p1)),sqrt(v2+square(0.01*p2)) # add 1% error floor

    mag1,mage1=fz.asinh_mag_map(p1,e1,zeropoint=flux_zeropoint,skynoise=skynoise) # Luptitude mapping
    mag2,mage2=fz.asinh_mag_map(p2,e2,zeropoint=flux_zeropoint,skynoise=skynoise) # Luptitude mapping

    # clear memory
    del flux_afterburner,err_afterburner,mask_afterburner
    del flux_afterburner_test,err_afterburner_test,mask_afterburner_test
    
    # initializing WINBET instances
    sys.stderr.write('Initializing WINBET training data instance...')    
    winbet_train=fz.WINBET(Ntrees=N_members,Nleaf=N_neighbors)
    if (m1==False).sum()>0:
        winbet_train.train(p1,v1,m1,mag1,mage1,mdict)
    else:
        winbet_train=None
    sys.stderr.write('done!\n')

    sys.stderr.write('Initializing WINBET target data instance...')    
    winbet_test=fz.WINBET(Ntrees=N_members,Nleaf=N_neighbors)
    if (m2==False).sum()>0:
        winbet_test.train(p2[flux_sel],v2[flux_sel],m2[flux_sel],mag2[flux_sel],mage2[flux_sel],mdict)
    else:
        winbet_test=None
    sys.stderr.write('done!\n')



    ##########################################
    ##### GENERATE FRANKEN-Z PREDICTIONS #####
    ##########################################
    
    sys.stderr.write('Computing FRANKEN-Z fits...')    
    frankenz=fz.FRANKENZ(N_members=N_members,n_neighbors=N_neighbors) # initialize FRANKEN-Z instance
    model_obj,model_Nobj,model_ll,model_Nbands=frankenz.predict(p1,e1,m1,
                                                                p2[flux_sel],e2[flux_sel],m2[flux_sel],
                                                                impute_train=winbet_train,impute_test=winbet_test) # compute predictions
    #sys.stderr.write('done!\n')


    sys.stderr.write('Saving fits...')
    
    gc.collect() # clean up memory
    model_sel=arange(len(flux_sel))[flux_sel] # selected objects (index)
    Nm_max=N_neighbors*N_members # maximum number of neighbors

    cols=fits.ColDefs([
        fits.Column(name='object_id',format='K',array=objid[model_sel]), # HSC IDs
        #fits.Column(name='tract',format='J',array=tract[model_sel]), # tract locations
        #fits.Column(name='patch',format='J',array=patch[model_sel]), # patch locations
        fits.Column(name='model_sel',format='J',array=model_sel), # target data indices
        fits.Column(name='model_obj',format=str(Nm_max)+'J',array=model_obj), # matched object indices
        fits.Column(name='model_Nobj',format='J',array=model_Nobj), # number of unique objects
        fits.Column(name='model_ll',format=str(Nm_max)+'E',array=model_ll), # log-likelihoods
        fits.Column(name='model_Nbands',format=str(Nm_max)+'I',array=model_Nbands), # number of bands per fit
    ]) # create fits columns
    tbhdu=fits.BinTableHDU.from_columns(cols) # create fits file
    tbhdu.writeto(output_location+'frankenz_s16a_v1_'+target_name+'_model.fits',clobber=True) # write to disk
    del cols,tbhdu
    
    sys.stderr.write('done!\n')



    ######################################
    ##### GENERATE PROCESSED OUTPUTS #####
    ######################################

    sys.stderr.write('Computing redshifts...')

    # generate redshifts
    model_llmin=empty(Nobs,dtype='float32') # min(ln-likelihood), i.e. best-fit result
    lzpdf=empty((Nobs,rdict.Nz),dtype='float32') # ln(1+z) PDF
    zpdf=empty((Nobs,rdict.Nz_out),dtype='float32') # z PDF
    model_levidence=empty(Nobs,dtype='float32') # -2ln(evidence), i.e. sum of all likelihoods

    for i in xrange(Nobs):
        if i%5000==0: 
            sys.stdout.write(str(i)+' ')
            sys.stdout.flush()
        Nm=model_Nobj[i]
        midx,ll=model_obj[i][:Nm],model_ll[i][:Nm]
        model_llmin[i]=ll.min() # minimum value (for scaling)
        w=exp(-0.5*(ll-model_llmin[i])) # transform to scaled likelihood weights
        model_levidence[i]=-2*log(w.sum())+model_llmin[i] # -2ln(Evidence)

        pz=fz.pdf_kde_dict(rdict.sig_dict,rdict.sig_width,lzidx[midx],lzeidx[midx],w,
                           rdict.grid,rdict.delta,rdict.Ngrid) # KDE dictionary PDF
        lzpdf[i]=pz[rdict.zmin_idx_highres:rdict.zmax_idx_highres:int(rdict.res)]

    sys.stderr.write('done!\n')

    # resample PDFs
    zpdf=fz.pdfs_resample(rdict.zgrid,lzpdf/rdict.znorm,rdict.zgrid_out) # resample from ln(1+z) to z space
    
    sys.stderr.write('Computing redshift type/survey statistics...')

    # compute Ntype and Ptype
    model_Ntype=zeros((Nobs,Nzt),dtype='float32')
    model_Nsurvey=zeros((Nobs,Nzs),dtype='float32')
    model_Ptype=zeros((Nobs,Nzt),dtype='float32')
    model_Psurvey=zeros((Nobs,Nzs),dtype='float32')

    for i in xrange(Nobs):
        if i%5000==0: 
            sys.stdout.write(str(i)+' ')
            sys.stdout.flush()
        Nm=model_Nobj[i]
        midx,ll=model_obj[i][:Nm],model_ll[i][:Nm]
        ztypes=zt[midx]-1
        zsurveys=zs[midx]-1
        like=exp(-0.5*(ll-model_llmin[i]))
        like/=like.sum()
        for j in xrange(Nm):
            model_Ntype[i][ztypes[j]]+=1
            model_Nsurvey[i][zsurveys[j]]+=1
            model_Ptype[i][ztypes[j]]+=like[j]
            model_Psurvey[i][zsurveys[j]]+=like[j]

    sys.stderr.write('done!\n')

    sys.stderr.write('Saving results...')
    
    gc.collect() # clean up memory
    cols=fits.ColDefs([
        fits.Column(name='object_id',format='K',array=objid[model_sel]), # HSC IDs
        #fits.Column(name='tract',format='J',array=tract[model_sel]), # tract locations
        #fits.Column(name='patch',format='J',array=patch[model_sel]), # patch locations
        fits.Column(name='model_sel',format='J',array=model_sel), # target data indices
        fits.Column(name='model_llmin',format='E',array=model_llmin), # min(log-likelihood), i.e. best fit
        fits.Column(name='model_levidence',format='E',array=model_levidence), # log-evidence, i.e. sum over all likelihoods
        fits.Column(name='model_Ntype',format=str(Nzt)+'J',array=model_Ntype), # number of neighbors (by type)
        fits.Column(name='model_Ptype',format=str(Nzt)+'E',array=model_Ptype), # probability of neighbors (by type)
        fits.Column(name='model_Nsurvey',format=str(Nzs)+'J',array=model_Nsurvey), # number of neighbors (by survey)
        fits.Column(name='model_Psurvey',format=str(Nzs)+'E',array=model_Psurvey), # probability of neighbors (by survey)
        fits.Column(name='lzpdf',format='208E',array=lzpdf), # ln(1+z) PDF
        fits.Column(name='zpdf',format='601E',array=zpdf) # resampled z PDF
    ]) # create fits columns
    tbhdu=fits.BinTableHDU.from_columns(cols) # create fits file
    tbhdu.writeto(output_location+'frankenz_s16a_v1_'+target_name+'_redshift.fits',clobber=True) # write to disk
    del cols,tbhdu
    
    sys.stderr.write('done!\n')




    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file',help="Master configuration file")
    parser.add_argument('target_file',help="Target data file")
    parser.add_argument('target_name',help="Target data name")
    parser.add_argument('output_location',help="Location to save output files")
    parser.add_argument('-nm','--N_members',default=25,type=int,help="Number of members in the ensemble")
    parser.add_argument('-nn','--N_neighbors',default=10,type=int,help="Number of neighbors collected from each member")

    args = parser.parse_args()

    main(args.config_file, args.target_file, args.target_name, args.output_location, N_members=args.N_members, N_neighbors=args.N_neighbors)
