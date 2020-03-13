import cosmoboost as cb
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

pars = cb.DEFAULT_PARS

lmax=pars['lmax']=3000
delta_ell = pars['delta_ell']=8
beta=pars['beta']
T_0 = pars["T_0"]

pars['d']=1
pars["method"] = "numerical"
kernel_n = cb.Kernel(pars, save_kernel=False, overwrite=True)

pars['d']=0
pars['method']='analytic'
kernel_a = cb.Kernel(pars, save_kernel=False, overwrite=True)

m=0
L = 501 

nu_0 = 217 #GHz

indx = cb.mL2indx(m,L,lmax)

K_a_mL_slice = kernel_a.mLl[indx]
K_a_nu_mL_slice = kernel_a.nu_mLl(nu_0)[indx]

K_n_mL_slice = kernel_n.mLl[indx]
K_n_nu_mL_slice = kernel_n.nu_mLl(nu_0)[indx]

dell = np.arange(-kernel_a.delta_ell,kernel_a.delta_ell+1)

# plt.plot(dell,(K_a_mL_slice),color="k",marker="o",lw=3, label='$T$')
# #plt.plot(dell,(K_nu_mL_slice),color="tab:red",ls="--",marker="o",label="$T_\\nu$ ({0} GHz)".format(nu_0))
#
# plt.plot(dell,(K_n_mL_slice),color="r",marker="o",lw=1, label='$T$')
# #plt.plot(dell,(K_nu_mL_slice),color="tab:red",ls="--",marker="o",label="$T_\\nu$ ({0} GHz)".format(nu_0))
# plt.xlabel(r"$\Delta \ell$")
# plt.ylabel(r"$|K_{m,\ell',\ell}|$")
#
# plt.xlim(-delta_ell,delta_ell)
# plt.grid()
# plt.legend()
# plt.show()

import os

T_0 = 2.725
ell=np.arange(lmax+1)

lib_dir = os.path.join(cb.COSMOBOOST_DIR,"lib")

Cl_camb = np.load(os.path.join(lib_dir,"sample_Cl.npz"))

Cl_TT = Cl_camb["TT"][:lmax+1]
Cl_EE = Cl_camb["EE"][:lmax+1]
Cl_BB = Cl_camb["BB"][:lmax+1]
Cl_TE = Cl_camb["TE"][:lmax+1]

# let's use it to simulate a CMB map
Cl = np.array([Cl_TT,Cl_EE,Cl_BB,Cl_TE])
alm_T, alm_E, alm_B = hp.synalm(Cl,lmax=lmax,new=True,verbose=True)

# this is our alm in the rest frame
alm_r = np.array([alm_T, alm_E, alm_B])

# this is the power spectrum of the simulation
Cl_r = hp.alm2cl(alm_r)

# plt.plot(ell,Cl_TT, label="TT (camb)",color='tab:red')
# plt.plot(ell,Cl_EE, label="EE (camb)",color='tab:blue')
# plt.plot(ell,Cl_TE, label="TE (camb)",color='tab:purple')
#
# plt.plot(ell,Cl_r[0],label="TT (sim)",color='tab:red',linestyle='--')
# plt.plot(ell,Cl_r[1],label="EE (sim)",color='tab:blue',linestyle='--')
# plt.plot(ell,Cl_r[3],label="TE (sim)",color='tab:purple',linestyle='--')
#
# plt.xlabel("$\ell$")
# plt.ylabel("$C_\ell$")
#
# plt.xscale("log")
# plt.yscale("log")
#
# plt.grid()
# plt.legend(loc="upper right",ncol=2)
# plt.xlim(2,lmax)
# plt.show()

alm_T_r = alm_T

# boost the temperature alm 
alm_T_b_a = cb.boost_alm(alm_T_r,kernel_a)
alm_T_b_n = cb.boost_alm(alm_T_r,kernel_n)

from cosmoboost.lib import jeong
from scipy.ndimage import gaussian_filter as GF

Cl_TT_r = Cl_r[0]
Cl_TT_b_a = hp.alm2cl(alm_T_b_a)
Cl_TT_b_n = hp.alm2cl(alm_T_b_n)

# Cl_TT_b_approx = cb.boost_Cl(Cl_TT_r,kernel)

# plt.plot(ell,ell**2*Cl_TT_r,linewidth=3,color='k',label='rest frame')
# plt.plot(ell,ell**2*Cl_TT_b_n,linewidth=1,color="tab:orange",label='boosted $a_{\ell m}$')
# plt.plot(ell,ell**2*Cl_TT_b_a,linewidth=0.5,ls='--',color="tab:red",label='boosted $C_{\ell}$')
# plt.xlabel("$\ell$")
# plt.ylabel("$\ell^2 C_\ell$")
# plt.yscale("log")
# plt.xlim(2,lmax)
# plt.ylim(1E-9,2E-7)
# plt.legend()
# plt.show()

# calculate the relative change of the boosted Cl using the accurate formalism
dCl_TT_b_a = (Cl_TT_b_a - Cl_TT_r)
dCl_Cl_TT_b_a = dCl_TT_b_a/Cl_TT_r
dCl_TT_b_n = (Cl_TT_b_n - Cl_TT_r)
dCl_Cl_TT_b_n = dCl_TT_b_n/Cl_TT_r
# # and for the approximation
# dCl_TT_b_approx = (Cl_TT_b_approx - Cl_TT_r)
# dCl_Cl_TT_b_approx = dCl_TT_b_approx/Cl_TT_r


# Gaussian smooth the results with a window
dL = 50
dCl_Cl_TT_b_a_GF = GF(dCl_Cl_TT_b_a, dL, mode="constant")
dCl_Cl_TT_b_n_GF = GF(dCl_Cl_TT_b_n, dL, mode="constant")
#dCl_Cl_TT_b_approx_GF = GF(dCl_Cl_TT_b_approx, dL, mode="constant")


# In[ ]:


# fig, ax = plt.subplots(2,1,figsize=(8,4),sharex=True)
# fig.subplots_adjust(hspace=0)
# ax[0].plot(100*dCl_Cl_TT_b_n,linewidth=0.5,alpha=0.7,color="tab:orange",label='boosted $a_{\ell m}$')
# ax[0].plot(100*dCl_Cl_TT_b_a,linewidth=0.5,alpha=0.7,color="tab:blue",label='boosted $a_{\ell m}$')
# ax[0].plot(100*dCl_Cl_TT_b_approx,linewidth=0.5, alpha=0.7,color="tab:red",label='boosted $C_\ell$')
# ax[1].plot(100*dCl_Cl_TT_b_n,linewidth=0.2,alpha=0.3,color="tab:orange")
# ax[1].plot(100*dCl_Cl_TT_b_approx,linewidth=0.3, alpha=0.4,color="tab:red")
# ax[1].plot(100*dCl_Cl_TT_b_n_GF,linewidth=2,color="tab:orange",label='boosted $a_{\ell m}$ (Gaussian smoothed)')
# ax[1].plot(100*dCl_Cl_TT_b_a_GF,linewidth=2,color="tab:blue",label='boosted $a_{\ell m}$ (Gaussian smoothed)')
# ax[1].set_ylim(-0.15,0.15)
# for axis in ax:
#     axis.set_xlabel("$\ell$")
#     axis.set_ylabel("$\Delta C_\ell/C_\ell (\%)$")
#     axis.set_xlim(2,lmax)
#     axis.grid()
#     axis.legend()
# plt.show()

def mask_cutbelowlat(cut_angle, lat_pix):
    """mask all the pixels beloe the latitude cut_angle [deg]"""
    mask = np.ones_like(lat_pix)
    mask[(lat_pix < cut_angle)] = 0.
    
    #approximate f_sky using the number of pixels
    f_sky = len(mask[mask == 1.]) / len(mask)

    return mask, f_sky

NSIDE=2048
NPIX = hp.nside2npix(NSIDE)
lon_pix, lat_pix = hp.pix2ang(NSIDE,np.arange(NPIX),lonlat=True)

mask_60,f_sky = mask_cutbelowlat(0,lat_pix)

boost_ang = -(lat_pix-90.)
# hp.mollview(boost_ang)

cos_map = np.cos(np.deg2rad(boost_ang))*mask_60
ones_map = np.ones_like(lat_pix)*mask_60
# hp.mollview(cos_map)
cos_avg = np.sum(cos_map)/np.sum(ones_map)
cos_avg
#jeong.jeong_boost_Cl()

T_map_r = hp.alm2map(alm_T_r,NSIDE)

T_map_b_a = hp.alm2map(alm_T_b_a,NSIDE)
T_map_b_n = hp.alm2map(alm_T_b_n,NSIDE)

T_map_r_ma = mask_60*T_map_r

T_map_b_a_ma = mask_60*T_map_b_a
T_map_b_n_ma = mask_60*T_map_b_n

T_map_b_a = hp.alm2map(alm_T_b_a,NSIDE)
T_map_b_n = hp.alm2map(alm_T_b_n,NSIDE)

T_map_b_a_ma = mask_60*T_map_b_a
T_map_b_n_ma = mask_60*T_map_b_n

# hp.mollview(mask_60, sub=221,title="Mask")
# hp.mollview(T_map_r_ma, sub=222,title="CMB")

Cl_TT_r_ma =(1/f_sky)*hp.anafast(T_map_r_ma,lmax=lmax)

Cl_TT_b_a_ma =(1/f_sky)*hp.anafast(T_map_b_a_ma,lmax=lmax)
Cl_TT_b_n_ma =(1/f_sky)*hp.anafast(T_map_b_n_ma,lmax=lmax)

# calculate the relative change of the boosted Cl using the accurate formalism
dCl_TT_b_a_ma = (Cl_TT_b_a_ma - Cl_TT_r_ma)
dCl_Cl_TT_b_a_ma = dCl_TT_b_a_ma/Cl_TT_r_ma

dCl_TT_b_n_ma = (Cl_TT_b_n_ma - Cl_TT_r_ma)
dCl_Cl_TT_b_n_ma = dCl_TT_b_n_ma/Cl_TT_r_ma

# Gaussian smooth the results with a window
dL = 50

dCl_Cl_TT_b_a_ma_GF = GF(dCl_Cl_TT_b_a_ma, dL, mode="constant")
dCl_Cl_TT_b_n_ma_GF = GF(dCl_Cl_TT_b_n_ma, dL, mode="constant")

Cl_TT_jeong = jeong.jeong_boost_Cl_1storder(ell, Cl_TT,beta= 0.00123, cos_avg= 0.5)

dCl_TT_b_jeong = (Cl_TT_jeong - Cl_TT_r)
dCl_Cl_TT_b_jeong = Cl_TT_jeong/Cl_TT_r

dCl_Cl_TT_b_jeong_GF = GF(dCl_Cl_TT_b_jeong, dL, mode="constant")

# plt.plot(ell,100*(dCl_Cl_TT_b_n_ma),linewidth=0.5,alpha=0.7,color="tab:orange",label="boosted and masked $a_{\ell m}$")
plt.plot(ell,100*(dCl_Cl_TT_b_n_ma_GF),linewidth=5,color="tab:orange",label="Gaussian smoothed")

# plt.plot(ell,100*(dCl_Cl_TT_b_a_ma),linewidth=0.5,alpha=0.7,color="tab:blue",label="boosted and masked $a_{\ell m}$")
plt.plot(ell,100*(dCl_Cl_TT_b_a_ma_GF),linewidth=3, ls="--", color="tab:blue",label="Gaussian smoothed")

plt.plot(ell,100*(dCl_Cl_TT_b_jeong_GF),linewidth=2,color="tab:red",label="Jeong")

plt.xlabel("$\ell$")
plt.ylabel("$\Delta C_\ell/C_\ell (\%)$")

plt.grid()
plt.legend()

plt.ylim(-1,1)
plt.xlim(100,lmax)
plt.savefig("dcl_cl.png")
