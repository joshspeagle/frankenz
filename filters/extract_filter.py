# extracting filter information

c_ang = 2.99792458e18 # speed of light in Angstroms/s

# galex filters
ffuv = open('splash/filters/galex1500.res','r')
fuv_band = array([])
fuv_band_sens = array([])
for line in ffuv:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    fuv_band = append(fuv_band,columns[0])
    fuv_band_sens = append(fuv_band_sens,columns[1])
ffuv.close()

#ln_band_nu = numpy.log(c_ang/fuv_band)
#ln_band = numpy.log(fuv_band)
#upper = trapz(ln_band*fuv_band_sens,ln_band_nu)
#lower = trapz(fuv_band_sens,ln_band_nu)
#lambda_eff = exp(upper/lower)

fnuv = open('splash/filters/galex2500.res','r')
nuv_band = array([])
nuv_band_sens = array([])
for line in fnuv:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    nuv_band = append(nuv_band,columns[0])
    nuv_band_sens = append(nuv_band_sens,columns[1])
fnuv.close()

# hubble filter
f = open('splash/filters/ACS_F814W.res','r')
f814w_band = array([])
f814w_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    f814w_band = append(f814w_band,columns[0])
    f814w_band_sens = append(f814w_band_sens,columns[1])
f.close()

# CFHT
fu = open('splash/filters/u_megaprime_sagem.res','r')
u_band = array([])
u_band_sens = array([])
for line in fu:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    u_band = append(u_band,columns[0])
    u_band_sens = append(u_band_sens,columns[1])
fu.close()

# subaru 
fb = open('splash/filters/B_subaru.res','r')
bp_band = array([])
bp_band_sens = array([])
for line in fb:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    bp_band = append(bp_band,columns[0])
    bp_band_sens = append(bp_band_sens,columns[1])
fb.close()

fv = open('splash/filters/V_subaru.res','r')
vp_band = array([])
vp_band_sens = array([])
for line in fv:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    vp_band = append(vp_band,columns[0])
    vp_band_sens = append(vp_band_sens,columns[1])
fv.close()

fg = open('splash/filters/g_subaru.res','r')
gp_band = array([])
gp_band_sens = array([])
for line in fg:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    gp_band = append(gp_band,columns[0])
    gp_band_sens = append(gp_band_sens,columns[1])
fg.close()

fr = open('splash/filters/r_subaru.res','r')
rp_band = array([])
rp_band_sens = array([])
for line in fr:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    rp_band = append(rp_band,columns[0])
    rp_band_sens = append(rp_band_sens,columns[1])
fr.close()

fi = open('splash/filters/i_subaru.res','r')
ip_band = array([])
ip_band_sens = array([])
for line in fi:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    ip_band = append(ip_band,columns[0])
    ip_band_sens = append(ip_band_sens,columns[1])
fi.close()

fi = open('splash/filters/i_subaru.res','r')
ic_band = array([])
ic_band_sens = array([])
for line in fi:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    ic_band = append(ic_band,columns[0])
    ic_band_sens = append(ic_band_sens,columns[1])
fi.close()

f = open('splash/filters/z_subaru.res','r')
zp_band = array([])
zp_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    zp_band = append(zp_band,columns[0])
    zp_band_sens = append(zp_band_sens,columns[1])
f.close()

fz = open('splash/filters/suprime_FDCCD_z.res','r')
zpp_band = array([])
zpp_band_sens = array([])
for line in fz:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    zpp_band = append(zpp_band,columns[0])
    zpp_band_sens = append(zpp_band_sens,columns[1])
fz.close()

# CFHT/UKIRT WFcam/wircam or whatever
fj = open('splash/filters/J_wfcam.res','r')
j_band = array([])
j_band_sens = array([])
for line in fj:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    j_band = append(j_band,columns[0])
    j_band_sens = append(j_band_sens,columns[1])
fj.close()

fh = open('splash/filters/wircam_H.res','r')
h_band = array([])
h_band_sens = array([])
for line in fh:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    h_band = append(h_band,columns[0])
    h_band_sens = append(h_band_sens,columns[1])
fh.close()

fk = open('splash/filters/flamingos_Ks.res','r')
ks_band = array([])
ks_band_sens = array([])
for line in fk:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    ks_band = append(ks_band,columns[0])
    ks_band_sens = append(ks_band_sens,columns[1])
fk.close()

fk = open('splash/filters/wircam_Ks.res','r')
kc_band = array([])
kc_band_sens = array([])
for line in fk:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    kc_band = append(kc_band,columns[0])
    kc_band_sens = append(kc_band_sens,columns[1])
fk.close()

# ultravista
f = open('splash/filters/Y_uv.res','r')
y_uv_band = array([])
y_uv_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    y_uv_band = append(y_uv_band,columns[0])
    y_uv_band_sens = append(y_uv_band_sens,columns[1])
f.close()

f = open('splash/filters/J_uv.res','r')
j_uv_band = array([])
j_uv_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    j_uv_band = append(j_uv_band,columns[0])
    j_uv_band_sens = append(j_uv_band_sens,columns[1])
f.close()

f = open('splash/filters/H_uv.res','r')
h_uv_band = array([])
h_uv_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    h_uv_band = append(h_uv_band,columns[0])
    h_uv_band_sens = append(h_uv_band_sens,columns[1])
f.close()

f = open('splash/filters/K_uv.res','r')
k_uv_band = array([])
k_uv_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    k_uv_band = append(k_uv_band,columns[0])
    k_uv_band_sens = append(k_uv_band_sens,columns[1])
f.close()

# sloan data

f = open('splash/filters/u_SDSS.res','r')
u_s_band = array([])
u_s_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    u_s_band = append(u_s_band,columns[0])
    u_s_band_sens = append(u_s_band_sens,columns[1])
f.close()

f = open('splash/filters/g_SDSS.res','r')
g_s_band = array([])
g_s_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    g_s_band = append(g_s_band,columns[0])
    g_s_band_sens = append(g_s_band_sens,columns[1])
f.close()

f = open('splash/filters/r_SDSS.res','r')
r_s_band = array([])
r_s_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    r_s_band = append(r_s_band,columns[0])
    r_s_band_sens = append(r_s_band_sens,columns[1])
f.close()

f = open('splash/filters/i_SDSS.res','r')
i_s_band = array([])
i_s_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    i_s_band = append(i_s_band,columns[0])
    i_s_band_sens = append(i_s_band_sens,columns[1])
f.close()

f = open('splash/filters/z_SDSS.res','r')
z_s_band = array([])
z_s_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    z_s_band = append(z_s_band,columns[0])
    z_s_band_sens = append(z_s_band_sens,columns[1])
f.close()

# subaru intermediate bands

f = open('splash/filters/IB427.SuprimeCam.pb','r')
IB427_band = array([])
IB427_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    IB427_band = append(IB427_band,columns[0])
    IB427_band_sens = append(IB427_band_sens,columns[1])
f.close()

f = open('splash/filters/IB464.SuprimeCam.pb','r')
IB464_band = array([])
IB464_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    IB464_band = append(IB464_band,columns[0])
    IB464_band_sens = append(IB464_band_sens,columns[1])
f.close()

f = open('splash/filters/IB484.SuprimeCam.pb','r')
IB484_band = array([])
IB484_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    IB484_band = append(IB484_band,columns[0])
    IB484_band_sens = append(IB484_band_sens,columns[1])
f.close()

f = open('splash/filters/IB505.SuprimeCam.pb','r')
IB505_band = array([])
IB505_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    IB505_band = append(IB505_band,columns[0])
    IB505_band_sens = append(IB505_band_sens,columns[1])
f.close()

f = open('splash/filters/IB527.SuprimeCam.pb','r')
IB527_band = array([])
IB527_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    IB527_band = append(IB527_band,columns[0])
    IB527_band_sens = append(IB527_band_sens,columns[1])
f.close()

f = open('splash/filters/IB574.SuprimeCam.pb','r')
IB574_band = array([])
IB574_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    IB574_band = append(IB574_band,columns[0])
    IB574_band_sens = append(IB574_band_sens,columns[1])
f.close()

f = open('splash/filters/IB624.SuprimeCam.pb','r')
IB624_band = array([])
IB624_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    IB624_band = append(IB624_band,columns[0])
    IB624_band_sens = append(IB624_band_sens,columns[1])
f.close()

f = open('splash/filters/IB679.SuprimeCam.pb','r')
IB679_band = array([])
IB679_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    IB679_band = append(IB679_band,columns[0])
    IB679_band_sens = append(IB679_band_sens,columns[1])
f.close()

f = open('splash/filters/IB709.SuprimeCam.pb','r')
IB709_band = array([])
IB709_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    IB709_band = append(IB709_band,columns[0])
    IB709_band_sens = append(IB709_band_sens,columns[1])
f.close()

f = open('splash/filters/IB738.SuprimeCam.pb','r')
IB738_band = array([])
IB738_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    IB738_band = append(IB738_band,columns[0])
    IB738_band_sens = append(IB738_band_sens,columns[1])
f.close()

f = open('splash/filters/IB767.SuprimeCam.pb','r')
IB767_band = array([])
IB767_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    IB767_band = append(IB767_band,columns[0])
    IB767_band_sens = append(IB767_band_sens,columns[1])
f.close()

f = open('splash/filters/IB827.SuprimeCam.pb','r')
IB827_band = array([])
IB827_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    IB827_band = append(IB827_band,columns[0])
    IB827_band_sens = append(IB827_band_sens,columns[1])
f.close()

# newfirm bands

f = open('splash/filters/J1.res','r')
j1_band = array([])
j1_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    j1_band = append(j1_band,columns[0])
    j1_band_sens = append(j1_band_sens,columns[1])
f.close()

f = open('splash/filters/J2.res','r')
j2_band = array([])
j2_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    j2_band = append(j2_band,columns[0])
    j2_band_sens = append(j2_band_sens,columns[1])
f.close()

f = open('splash/filters/J3.res','r')
j3_band = array([])
j3_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    j3_band = append(j3_band,columns[0])
    j3_band_sens = append(j3_band_sens,columns[1])
f.close()

f = open('splash/filters/H1.res','r')
h1_band = array([])
h1_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    h1_band = append(h1_band,columns[0])
    h1_band_sens = append(h1_band_sens,columns[1])
f.close()

f = open('splash/filters/H2.res','r')
h2_band = array([])
h2_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    h2_band = append(h2_band,columns[0])
    h2_band_sens = append(h2_band_sens,columns[1])
f.close()

f = open('splash/filters/Ks_newfirm.res','r')
knf_band = array([])
knf_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    knf_band = append(knf_band,columns[0])
    knf_band_sens = append(knf_band_sens,columns[1])
f.close()

# subaru narrow bands

f = open('splash/filters/NB711.SuprimeCam.pb','r')
NB711_band = array([])
NB711_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    NB711_band = append(NB711_band,columns[0])
    NB711_band_sens = append(NB711_band_sens,columns[1])
f.close()

f = open('splash/filters/NB816.SuprimeCam.pb','r')
NB816_band = array([])
NB816_band_sens = array([])
for line in f:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    NB816_band = append(NB816_band,columns[0])
    NB816_band_sens = append(NB816_band_sens,columns[1])
f.close()

# spitzer bands

fch1 = open('splash/filters/irac_ch1.res','r')
ch1_band = array([])
ch1_band_sens = array([])
for line in fch1:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    ch1_band = append(ch1_band,columns[0])
    ch1_band_sens = append(ch1_band_sens,columns[1])
fch1.close()

fch2 = open('splash/filters/irac_ch2.res','r')
ch2_band = array([])
ch2_band_sens = array([])
for line in fch2:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    ch2_band = append(ch2_band,columns[0])
    ch2_band_sens = append(ch2_band_sens,columns[1])
fch2.close()

fch3 = open('splash/filters/irac_ch3.res','r')
ch3_band = array([])
ch3_band_sens = array([])
for line in fch3:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    ch3_band = append(ch3_band,columns[0])
    ch3_band_sens = append(ch3_band_sens,columns[1])
fch3.close()

fch4 = open('splash/filters/irac_ch4.res','r')
ch4_band = array([])
ch4_band_sens = array([])
for line in fch4:
    columns = line.strip().split()
    columns = array(columns).astype(float)
    ch4_band = append(ch4_band,columns[0])
    ch4_band_sens = append(ch4_band_sens,columns[1])
fch4.close()

##### Extracting filter effective band centers

# centers of the relevant bands (taken from Olivier+09, VISTA filter page,
# Fukugita+96, newfirm filter doc, and stsci WFC3 handbook c06_uvis06 table 6.2)
# Includes:
# Galex
# u (CFHT), b-z (subaru), zpp (subaru - calculated, but might be a bit off)
# IB,NB subaru
# i (CFHT)
# wfcam (J), wircam (H,Kc), flamingos (Ks)
# spitzer
# ultravista
# sloan
# newfirm
# hubble f814w
## ALL OF THESE HAVE BEEN PAINSTAKINGLY ORDERED - KEEP THEM THIS WAY
fuv_cent = 1551.3
nuv_cent = 2306.5
u_s_cent = 3540.0
u_cent = 3911.0
ib427_cent = 4256.3
b_cent = 4439.6
ib464_cent = 4633.3
gp_cent = 4728.3
g_s_cent = 4770.0
ib484_cent = 4845.9
ib505_cent = 5060.7
ib527_cent = 5258.9
vp_cent = 5448.9
ib574_cent = 5762.1
r_s_cent = 6222.0
ib624_cent = 6230.0
rp_cent = 6231.8
ib679_cent = 6778.8
ib709_cent = 7070.7
nb711_cent = 7119.6
ib738_cent = 7358.7
ic_cent = 7628.9
ip_cent = 7629.1
i_s_cent = 7632.0
ib767_cent = 7681.2
f814w_cent = 8024
nb816_cent = 8149.0
ib827_cent = 8240.9
zp_cent = 9021.6
z_s_cent = 9049.0
zpp_cent = 9077.4
y_uv_cent = 10210.0
j1_cent = 10484.0
j2_cent = 11903.0
j_cent = 12444.1
j_uv_cent = 12540.0
j3_cent = 12837.0
h1_cent = 15557.0
h_uv_cent = 16460.0
h2_cent = 17059.0
ks_cent = 21434.8
kc_cent = 21480.2
k_uv_cent = 21490.0
knf_cent = 21500.0
ch1_cent = 35262.5
ch2_cent = 44606.7
ch3_cent = 56762.4
ch4_cent = 77030.1
