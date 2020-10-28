#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt

import healpy as hp
import numpy as np
from scipy.special import factorial 

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

plt.rcParams['font.size']=13
plt.rcParams['font.family']='stix'
plt.rcParams['text.usetex']=False
plt.rcParams['figure.figsize']= (6.5,4)
plt.rcParams['figure.dpi']=150


# Import the cosmoboost package: 

# In[2]:


import cosmoboost as cb


# # Initialize the Doppler and Aberration Kernel

# In order to initialize the kernel, you need to set up the parameters dictionary. The default values are stored in `cosmoboost.DEFAULT_PARS`. 

# In[3]:


#read the default parameters from cosmoboost
pars = cb.DEFAULT_PARS


# Let's see what it contains

# In[4]:


pars


# ## quick note on the parameters

# **The important ones**
# 
# `d`: Doppler weight of the observable
# 
# `s`: Spin weight of the observable (0 for temperature and 2 for polarization)
# 
# `beta`: Velocity of the frame [assumes motion in the $\hat{z}$ direction (north galactic pole)]
# 
# `lmax`: Limits the kernel calculation up to this ell mode 
# 
# `delta_ell`: Limits the kernel calculation to this many neighbors on each side of every ell mode. For example, `delta_ell=2` calculates the motion induced leakage from `ell-2`, `ell-1`, `ell+1`, and `ell+2`. By default the code uses the recommended number of modes. 
# 
# `T_0`: Average temperature of the background radiation in units of Kelvin.
# 
# **The rest**
# 
# `lmin`: Should always be set to zero. This is only defined for the sake of clarity and to specifiy that ell does NOT start from 2. 
# 
# `beta_expansion_order`: Calculates the generalized Doppler and aberration effect up to this order. 
# 
# `normalize`: If set to True, calculates everything in units of temperature. 
# 
# `derivative_dnu`: Frequency resolution used for derivatives in the generalized Doppler and aberration kernel. 
# 
# `frequency_function`: Determines the frequency function of the background radiation. Currently can be only set to `"CMB"` and `"tSZ"`.

# # Instantiating the Kernel 

# Now let's change some of the parameters

# In[5]:


lmax=pars['lmax']=1000
delta_ell = pars['delta_ell']=8
pars['d']=1

beta=pars['beta']
T_0 = pars["T_0"]
pars["method"] = "Bessel"


# In[6]:


pars


# Now let's initialize the kernel using these parameters m

# In[7]:


# initialize the kernel 
pars["method"]="Bessel"
kernel_a = cb.Kernel(pars, overwrite=True, save_kernel=False)


# In[ ]:





# In[8]:


kernel_a.pars


# In[23]:


lmax=pars['lmax']=2000
delta_ell = pars['delta_ell']=8
pars['d']=1

beta=pars['beta']
T_0 = pars["T_0"]
pars["method"] = "ODE"


# In[ ]:


pars["method"] = "ODE"
kernel_n = cb.Kernel(pars, overwrite=True, save_kernel=False)


# In[11]:


kernel_n.pars


# The elements of `kernel.mLl` are organized in a matrix such that each row corresponds to a ($m,\ell'$) pair, and each column to a $\Delta \ell$ value. The Each block starts with an $m$ and goes down through all possible values of $\ell'$ (0 to $m$). Here's an example of the matrix around the neighbor of $(m,\ell')=(0,501)$:

# |          -        |  -  |      -     | $\Delta \ell$ |      -     |  -  |
# |:-----------------:|:---:|:----------:|---------------|:----------:|:---:|
# |         -         | ... | -1         | 0             | +1         | ... |
# |         -         | ... | (0,500, 409)  | (0,500, 500)    | (0,500, 501) | ... |
# | ($m,\ell', \ell$) | ... | (0,501, 500) | (0,501, 501)    | (0,501, 502) | ... |
# |         -         | ... | (0,502, 501) | (0,502, 502)    | (0,502, 503) | ... |

# Now let's slice and print the K_{m, ell', ell} matrix for these values of $m$ and $\ell'$ which we call `(m,L)`

# In[12]:


kernel_a.mLl


# In[18]:


m= 49
L = 50

# and let's also calculate the generalized kernel at this frequency 
nu_0 = 217 #GHz 

# this function let's us slice the (m,L) row
indx = cb.mL2indx(m,L,lmax)


# In[ ]:





# In[19]:


# frequency-independent
K_a_mL_slice = kernel_a.mLl[indx]

# frequency-dependent (evaluated at 217 GHz)
#K_a_nu_mL_slice = kernel_a.nu_mLl(nu_0)[indx]

print("K_mLl = {}\n".format(K_a_mL_slice))
#print("K_mLl_nu({}GHz) = {}".format(nu_0, K_a_nu_mL_slice))


# In[20]:


# frequency-independent
K_n_mL_slice = kernel_n.mLl[indx]

# frequency-dependent (evaluated at 217 GHz)
#K_n_nu_mL_slice = kernel_n.nu_mLl(nu_0)[indx]

print("K_mLl = {}\n".format(K_n_mL_slice))
#print("K_mLl_nu({}GHz) = {}".format(nu_0, K_n_nu_mL_slice))


# In[21]:



# and here's the plot
dell = np.arange(-kernel_a.delta_ell,kernel_a.delta_ell+1)

plt.plot(dell,(K_a_mL_slice),color="k",marker="o",lw=3, label='$Bessel$')
plt.plot(dell,(K_n_mL_slice),color="r",marker=".",lw=1, label='$ODE$')
#plt.plot(dell,(K_nu_mL_slice),color="tab:red",ls="--",marker="o",label="$T_\\nu$ ({0} GHz)".format(nu_0))

#plt.plot(dell,(K_n_mL_slice),color="r",marker="o",lw=1, label='$T$')
#plt.plot(dell,(K_nu_mL_slice),color="tab:red",ls="--",marker="o",label="$T_\\nu$ ({0} GHz)".format(nu_0))
plt.xlabel(r"$\Delta \ell$")
plt.ylabel(r"$|K_{m,\ell',\ell}|$")

plt.xlim(-delta_ell,delta_ell)
#plt.yscale("log")
plt.grid()
plt.legend()
plt.show()


# In[22]:



# and here's the plot
dell = np.arange(-kernel_a.delta_ell,kernel_a.delta_ell+1)

plt.plot(dell,(K_a_mL_slice),color="k",marker="o",lw=3, label='$Bessel$')
plt.plot(dell,(K_n_mL_slice),color="r",marker=".",lw=1, label='$ODE$')
#plt.plot(dell,(K_nu_mL_slice),color="tab:red",ls="--",marker="o",label="$T_\\nu$ ({0} GHz)".format(nu_0))

#plt.plot(dell,(K_n_mL_slice),color="r",marker="o",lw=1, label='$T$')
#plt.plot(dell,(K_nu_mL_slice),color="tab:red",ls="--",marker="o",label="$T_\\nu$ ({0} GHz)".format(nu_0))
plt.xlabel(r"$\Delta \ell$")
plt.ylabel(r"$|K_{m,\ell',\ell}|$")

plt.xlim(-delta_ell,delta_ell)
#plt.yscale("log")
plt.grid()
plt.legend()
plt.show()


# The difference between the Kernel coefficients for thermodynmic temperature and brightness temperature are negligible in this range at 217 GHz.

# # Analytical Kernel 

# In[18]:


from scipy.integrate import quad
from scipy.special import legendre, lpmn, factorial, sph_harm
from functools import partial


# In[19]:


from pyshtools.legendre import legendre_lm


# In[ ]:


legendre_lm(1,1, 0)


# In[ ]:


def analytic_K_Llm(beta, L, l, m):
    """calculate the Kernel coefficients analytically
    Uses Eq. 10 in https://arxiv.org/pdf/1309.2285.pdf"""
    
    if (m > l) or (m > L):
        return 0
    
    def N(l, m): # Nlm coefficients in Eq. 13
        numerator = (2*l+1)*factorial(l-m)
        denominator = 4*np.pi*factorial(l+m)
        return np.sqrt(numerator/denominator)

    def mu(mu_p, beta): # The aberration formula Eq. 2
    
        return (mu_p-beta)/(1-beta*mu_p)
    
    A = (2 * np.pi) * N(l, m) * N(L,m)
    Plm = partial(legendre_lm, l, m, normalization="unnorm")
    PLm = partial(legendre_lm, L, m, normalization="unnorm")
    

    
    def integrand(mu_p, beta): #The integrand in Eq. 10
        gamma = 1/np.sqrt(1-beta**2)
        return PLm(mu(mu_p, beta)) * Plm(mu_p) / (gamma * 1-beta*mu_p)
    
    B = quad(integrand, -1 ,1, args=(beta))[0]
    
    return A*B

def PLm_ortho_check(L, l, m):
    
    A = (2 * np.pi) * N(l, m) * N(L,m)
    Plm = partial(legendre_lm, l, m, )
    PLm = partial(legendre_lm, L, m, )
    

    
    def integrand(mu_p):
        
        return PLm(mu_p) * Plm(mu_p)
    
    B = quad(integrand, -1 ,1)[0]
    
    return A*B
    


# In[ ]:


analytic_K_Llm(0.01, 20, 10, 0)


# In[ ]:



PLm_ortho_check(3,2,1)


# In[ ]:


lmax=pars['lmax']=500
delta_ell = pars['delta_ell']=8
pars['d']=1
pars['beta'] = 0.123
beta=pars['beta']
T_0 = pars["T_0"]
pars["method"] = "Bessel"

test_K_a = cb.Kernel(pars, overwrite=True, save_kernel=False)


# In[ ]:


lmax=pars['lmax']=500
delta_ell = pars['delta_ell']=8
pars['d']=1
pars['beta'] = 0.123
beta=pars['beta']
T_0 = pars["T_0"]
pars["method"] = "ODE"

test_K_n = cb.Kernel(pars, overwrite=True, save_kernel=False)


# In[ ]:


lmax


# In[ ]:


for m in [10, 20, 50]:
    
    L = m+3
    
    # and let's also calculate the generalized kernel at this frequency 
    #nu_0 = 217 #GHz 

    # this function let's us slice the (m,L) row
    indx = cb.mL2indx(m,L,lmax)

    K_slice_n = test_K_n.mLl[indx]
    K_slice_a = test_K_a.mLl[indx]
    
    #K_slice_n = test_K_n.Ll[L]
    #K_slice_a = test_K_a.Ll[L]
    x_axis = range(-delta_ell,delta_ell+1)
    plt.text(-7,0.1, f"(m, L) = ({m},{L})")
    plt.plot(x_axis, [analytic_K_Llm(test_K_a.beta, L+dl,L,m) for dl in range(-delta_ell,delta_ell+1)], 
             label="Direct Integration", lw=3, color="tab:orange")
    plt.plot(x_axis, K_slice_a, label="Bessel", ls="--", lw=2)
    plt.plot(x_axis, K_slice_n, label="ODE", ls="--", color="k")

    plt.xlim(-8,8)
    plt.grid()
    plt.legend()
    plt.show()
    
    #plt.savefig("ODE_Bessel_Direct_comparison_2.png", dpi=150)


# In[ ]:


np.set_printoptions(precision=2)


# In[ ]:


Blms, _ = cb.mh.get_Blm_Clm(delta_ell, lmax, s=0)
Bmatrix = Blms[test_K_a.Lmatrix, test_K_a.Mmatrix]
Bmatrix[np.isnan(Bmatrix)] = 0


# In[ ]:


Bmatrix[0]


# In[ ]:


dl = np.array([np.arange(delta_ell, -delta_ell - 1, -1)] * Bmatrix.shape[0])


# In[ ]:


import scipy
scipy.special.jv(dl, 2. * Bmatrix )[0]


# In[ ]:


analytic_K = [analytic_K_Llm(test_K.beta, L+dl,L,1) for dl in range(-delta_ell,delta_ell+1)]


# In[ ]:


np.all(np.isclose(analytic_K, K_slice,  rtol=1E-3, atol=1E-3))


# In[ ]:


np.all(np.isclose(K_slice_a, K_slice_n,  rtol=1E-2, atol=1E-2))


# In[ ]:


K_slice_n


# In[ ]:


K_slice_a


# # Simulate the sky

# In[ ]:


import os


T_0 = 2.725E6
ell=np.arange(lmax+1)

# here's a sample power spectrum generated with CAMB
lib_dir = os.path.join(cb.COSMOBOOST_DIR,"lib")

Cl_camb = np.load(os.path.join(lib_dir,"sample_Cl.npz"))

Cl_TT = 1E12*Cl_camb["TT"][:lmax+1]
Cl_EE = 1E12*Cl_camb["EE"][:lmax+1]
Cl_BB = 1E12*Cl_camb["BB"][:lmax+1]
Cl_TE = 1E12*Cl_camb["TE"][:lmax+1]

# let's use it to simulate a CMB map
Cl = np.array([Cl_TT,Cl_EE,Cl_BB,Cl_TE])
alm_T, alm_E, alm_B = hp.synalm(Cl,lmax=lmax,new=True,verbose=True)

# this is our alm in the rest frame
alm_r = np.array([alm_T, alm_E, alm_B])

# this is the power spectrum of the simulation
Cl_r = hp.alm2cl(alm_r)


# In[ ]:


# here's a plot the power spectra in the rest frame

plt.plot(ell,Cl_TT, label="TT (camb)",color='tab:red')
plt.plot(ell,Cl_EE, label="EE (camb)",color='tab:blue')
plt.plot(ell,Cl_TE, label="TE (camb)",color='tab:purple')

plt.plot(ell,Cl_r[0],label="TT (sim)",color='tab:red',linestyle='--')
plt.plot(ell,Cl_r[1],label="EE (sim)",color='tab:blue',linestyle='--')
plt.plot(ell,Cl_r[3],label="TE (sim)",color='tab:purple',linestyle='--')

plt.xlabel("$\ell$")
plt.ylabel("$C_\ell$")

plt.xscale("log")
plt.yscale("log")

plt.grid()
plt.legend(loc="upper right",ncol=2)
plt.xlim(2,lmax)
plt.show()


# # Boost the Whole Sky

# ## temperature only

# Boosting the alms using `CosmoBoost` is very simple

# In[ ]:


alm_T_r = alm_T

# boost the temperature alm 
alm_T_b_a = cb.boost_alm(alm_T_r,kernel_a)
alm_T_b_n = cb.boost_alm(alm_T_r,kernel_n)


# In[ ]:


#alm_T_b_a = cb.boost_alm(alm_T_r,kernel_a)
#alm_T_b_n = cb.boost_alm(alm_T_r,kernel_n)


# ## other options

# You can also pass `alm_r` to the `cb.boost_alm` function which will boost both temperature and polarization:
# 
# `alm_b = cb.boost_alm(alm_r,kernel)`
# 
# And in order to boost a frequency-dependent observable, you can pass the frequency as the last argument as such
# 
# `alm_b_217 = cb.boost_alm(alm_r, kernel, 217)`

# ## boosted power spectreum

# In[ ]:


from cosmoboost.lib import jeong
from scipy.ndimage import gaussian_filter as GF


# In[ ]:


# calculate the temperature power spectrum in the rest and boosted frame
Cl_TT_r = Cl_r[0] 
Cl_TT_b_a = hp.alm2cl(alm_T_b_a)
Cl_TT_b_n = hp.alm2cl(alm_T_b_n)


# In[ ]:


delete_last = 10

ell = ell[:-delete_last]
Cl_TT_r = Cl_TT_r[:-delete_last]
Cl_TT_b_a = Cl_TT_b_a[:-delete_last]
Cl_TT_b_n = Cl_TT_b_n[:-delete_last]

#dCl_Cl_TT_b_a[np.isinf(dCl_Cl_TT_b_a)]=0
#dCl_Cl_TT_b_n[np.isinf(dCl_Cl_TT_b_n)]=0


# `CosmoBoost` also provides an additional boosting method `cb.boost_Cl` which can be directly applied to the power spectrum. However, since this method assumes azimuthal symmetry, it underestimates the effect.

# Now let's plot the power spectra together

# In[ ]:


plt.plot(ell,ell**2*Cl_TT_r,linewidth=3,color='k',label='rest frame')
plt.plot(ell,ell**2*Cl_TT_b_n,linewidth=1,color="tab:orange",label='boosted $a_{\ell m}$')
plt.plot(ell,ell**2*Cl_TT_b_a,linewidth=0.5,ls='--',color="tab:red",label='boosted $C_{\ell}$')


plt.xlabel("$\ell$")
plt.ylabel("$\ell^2 C_\ell$")

plt.yscale("log")
plt.xlim(2,lmax)
#plt.ylim(1E-9,2E-7)
plt.legend()


plt.show()


# In[ ]:


from scipy.interpolate import interp1d


# In order to see the difference between them better, we calculate the relative change between the boosted and the rest frame spectra.

# In[ ]:


# calculate the relative change of the boosted Cl using the accurate formalism
dCl_TT_b_a = (Cl_TT_b_a - Cl_TT_r)
dCl_Cl_TT_b_a = dCl_TT_b_a/Cl_TT_r
dCl_Cl_TT_b_a[np.isinf(dCl_Cl_TT_b_a)]=0


dCl_TT_b_n = (Cl_TT_b_n - Cl_TT_r)
dCl_Cl_TT_b_n = dCl_TT_b_n/Cl_TT_r
dCl_Cl_TT_b_n[np.isinf(dCl_Cl_TT_b_n)]=0


# # and for the approximation
# dCl_TT_b_approx = (Cl_TT_b_approx - Cl_TT_r)
# dCl_Cl_TT_b_approx = dCl_TT_b_approx/Cl_TT_r


# Gaussian smooth the results with a window
dL = 50

dCl_TT_b_a_GF = GF(dCl_TT_b_a, dL, mode="constant", truncate=5)
dCl_TT_b_n_GF = GF(dCl_TT_b_n, dL, mode="constant", truncate=5)

dCl_Cl_TT_b_a_GF = GF(dCl_Cl_TT_b_a, dL, mode="constant",truncate=5)
dCl_Cl_TT_b_n_GF = GF(dCl_Cl_TT_b_n, dL, mode="constant",truncate=5)
#dCl_Cl_TT_b_approx_GF = GF(dCl_Cl_TT_b_approx, dL, mode="constant")


# In[ ]:


fig, ax = plt.subplots(2,1,figsize=(8,4),sharex=True)
fig.subplots_adjust(hspace=0)

ax[0].plot(100*dCl_TT_b_n,linewidth=0.5,alpha=0.7,color="tab:orange",label='boosted $a_{\ell m}$')
ax[0].plot(100*dCl_TT_b_a,linewidth=0.5,alpha=0.7,color="tab:blue",label='boosted $a_{\ell m}$')
#ax[0].plot(100*dCl_Cl_TT_b_approx,linewidth=0.5, alpha=0.7,color="tab:red",label='boosted $C_\ell$')


#ax[1].plot(100*dCl_Cl_TT_b_n,linewidth=0.2,alpha=0.3,color="tab:orange")
#ax[1].plot(100*dCl_Cl_TT_b_approx,linewidth=0.3, alpha=0.4,color="tab:red")

ax[1].plot(1E2*dCl_TT_b_n_GF,linewidth=2,color="tab:orange",label='boosted $a_{\ell m}$ (Gaussian smoothed)')
ax[1].plot(1E2*dCl_TT_b_a_GF,linewidth=2,color="tab:blue",label='boosted $a_{\ell m}$ (Gaussian smoothed)')


ax[0].set_ylim(-0.5,0.5)
ax[1].set_ylim(-0.01,0.01)

for axis in ax:
    axis.set_xlabel("$\ell$")
    axis.set_ylabel("$\Delta C_\ell/C_\ell (\%)$")
    axis.set_xscale("log")
    axis.set_xlim(250,lmax)
    axis.grid()
    axis.legend()


plt.show()


# In[ ]:





# In[ ]:


fig, ax = plt.subplots(2,1,figsize=(8,4),sharex=True)
fig.subplots_adjust(hspace=0)

ax[0].plot(100*dCl_Cl_TT_b_n,linewidth=0.5,alpha=0.7,color="tab:orange",label='boosted $a_{\ell m}$')
ax[0].plot(100*dCl_Cl_TT_b_a,linewidth=0.5,alpha=0.7,color="tab:blue",label='boosted $a_{\ell m}$')
#ax[0].plot(100*dCl_Cl_TT_b_approx,linewidth=0.5, alpha=0.7,color="tab:red",label='boosted $C_\ell$')


#ax[1].plot(100*dCl_Cl_TT_b_n,linewidth=0.2,alpha=0.3,color="tab:orange")
#ax[1].plot(100*dCl_Cl_TT_b_approx,linewidth=0.3, alpha=0.4,color="tab:red")

ax[1].plot(100*dCl_Cl_TT_b_n_GF,linewidth=2,color="tab:orange",label='boosted $a_{\ell m}$ (Gaussian smoothed)')
ax[1].plot(100*dCl_Cl_TT_b_a_GF,linewidth=2,color="tab:blue",label='boosted $a_{\ell m}$ (Gaussian smoothed)')


ax[1].set_ylim(-0.15,0.15)

for axis in ax:
    axis.set_xlabel("$\ell$")
    axis.set_ylabel("$\Delta C_\ell/C_\ell (\%)$")
    axis.set_xlim(2,lmax)
    axis.grid()
    axis.legend()


plt.show()


# The effect of the boost is negligible for in whole-sky maps. However, this is not the case for partial sky observations as we will see in the next section. 

# In[ ]:


cb.DEFAULT_PARS


# # Boost a Masked Sky

# In[ ]:


def mask_cutbelowlat(cut_angle, lat_pix):
    """mask all the pixels beloe the latitude cut_angle [deg]"""
    mask = np.ones_like(lat_pix)
    mask[(lat_pix < cut_angle)] = 0.
    
    #approximate f_sky using the number of pixels
    f_sky = len(mask[mask == 1.]) / len(mask)

    return mask, f_sky


# In[ ]:


def mask_cutabovelat(cut_angle, lat_pix):
    """mask all the pixels beloe the latitude cut_angle [deg]"""
    mask = np.ones_like(lat_pix)
    mask[(lat_pix > cut_angle)] = 0.
    
    #approximate f_sky using the number of pixels
    f_sky = len(mask[mask == 1.]) / len(mask)

    return mask, f_sky


# In[ ]:


NSIDE=512
NPIX = hp.nside2npix(NSIDE)
lon_pix, lat_pix = hp.pix2ang(NSIDE,np.arange(NPIX),lonlat=True)


# In[ ]:


mask_60, f_sky = mask_cutbelowlat(0,lat_pix)


# In[ ]:


# cos_map = np.cos(np.deg2rad(boost_ang))*mask_60
# ones_map = np.ones_like(lat_pix)*mask_60
# hp.mollview(cos_map)
# cos_avg = np.sum(cos_map)/np.sum(ones_map)
# cos_avg
#jeong.jeong_boost_Cl()


# In[ ]:


T_map_r = hp.alm2map(alm_T_r,NSIDE)

T_map_b_a = hp.alm2map(alm_T_b_a,NSIDE)
T_map_b_n = hp.alm2map(alm_T_b_n,NSIDE)

T_map_r_ma = mask_60*T_map_r

T_map_b_a_ma = mask_60*T_map_b_a
T_map_b_n_ma = mask_60*T_map_b_n


# In[ ]:



T_map_b_a = hp.alm2map(alm_T_b_a,NSIDE)
T_map_b_n = hp.alm2map(alm_T_b_n,NSIDE)


T_map_b_a_ma = mask_60*T_map_b_a
T_map_b_n_ma = mask_60*T_map_b_n


# In[ ]:


hp.mollview(mask_60, sub=221,title="Mask")
hp.mollview(T_map_r_ma, sub=222,title="CMB")


# In[ ]:


ell=np.arange(lmax+1)


# In[ ]:


Cl_TT_r_ma =(1/f_sky)*hp.anafast(T_map_r_ma,lmax=lmax)

Cl_TT_b_a_ma =(1/f_sky)*hp.anafast(T_map_b_a_ma,lmax=lmax)
Cl_TT_b_n_ma =(1/f_sky)*hp.anafast(T_map_b_n_ma,lmax=lmax)


# In[ ]:


delete_first= 10
delete_last = 1

ell_ma = ell[delete_first:-delete_last]
Cl_TT_r_ma = Cl_TT_r_ma[delete_first:-delete_last]
Cl_TT_b_a_ma = Cl_TT_b_a_ma[delete_first:-delete_last]
Cl_TT_b_n_ma = Cl_TT_b_n_ma[delete_first:-delete_last]


# In[ ]:


# calculate the relative change of the boosted Cl using the accurate formalism
dCl_TT_b_a_ma = (Cl_TT_b_a_ma - Cl_TT_r_ma)
dCl_Cl_TT_b_a_ma = dCl_TT_b_a_ma/Cl_TT_r_ma

dCl_TT_b_n_ma = (Cl_TT_b_n_ma - Cl_TT_r_ma)
dCl_Cl_TT_b_n_ma = dCl_TT_b_n_ma/Cl_TT_r_ma

dCl_Cl_TT_b_a_ma[np.isinf(dCl_Cl_TT_b_a_ma)]=0
dCl_Cl_TT_b_n_ma[np.isinf(dCl_Cl_TT_b_n_ma)]=0

# Gaussian smooth the results with a window
dL = 30

dCl_TT_b_a_ma_GF = GF(dCl_TT_b_a_ma, dL, mode="constant")
dCl_TT_b_n_ma_GF = GF(dCl_TT_b_n_ma, dL, mode="constant")


dCl_Cl_TT_b_a_ma_GF = GF(dCl_Cl_TT_b_a_ma, dL, mode="constant")
dCl_Cl_TT_b_n_ma_GF = GF(dCl_Cl_TT_b_n_ma, dL, mode="constant")


# In[ ]:


Cl_TT_jeong = jeong.jeong_boost_Cl_1storder(ell_ma, Cl_TT[delete_first:-delete_last], beta= 0.00123, cos_avg= 0.5, only_dCl=False)

dCl_TT_b_jeong = (Cl_TT_jeong - Cl_TT[delete_first:-delete_last])
dCl_Cl_TT_b_jeong = dCl_TT_b_jeong/Cl_TT[delete_first:-delete_last]

dCl_TT_b_jeong_GF = GF(dCl_TT_b_jeong, dL, mode="constant")

dCl_Cl_TT_b_jeong_GF = GF(dCl_Cl_TT_b_jeong, dL, mode="constant")


# In[ ]:


#plt.plot(ell,100*(dCl_Cl_TT_b_n_ma),linewidth=0.5,alpha=0.7,color="tab:orange",label="boosted and masked $a_{\ell m}$")
plt.plot(ell_ma,100*(dCl_TT_b_n_ma_GF),linewidth=5,color="tab:orange",label="Gaussian smoothed")

#plt.plot(ell,100*(dCl_Cl_TT_b_a_ma),linewidth=0.5,alpha=0.7,color="tab:blue",label="boosted and masked $a_{\ell m}$")
plt.plot(ell_ma,100*(dCl_TT_b_a_ma_GF),linewidth=3, ls="--", color="tab:blue",label="Gaussian smoothed")

plt.plot(ell_ma,100*(dCl_TT_b_jeong),linewidth=2,color="tab:red",label="Jeong")

plt.xlabel("$\ell$")
plt.ylabel("$\Delta C_\ell/C_\ell (\%)$")

plt.grid()
plt.legend()

plt.ylim(-.1,.1)
plt.xlim(100,lmax)
plt.show()


# In[ ]:


#plt.plot(ell,100*(dCl_Cl_TT_b_n_ma),linewidth=0.5,alpha=0.7,color="tab:orange",label="boosted and masked $a_{\ell m}$")
plt.figure(figsize=(10,5))

plt.plot(ell_ma,100*(dCl_Cl_TT_b_n_ma_GF),linewidth=5,color="tab:orange",label="Numerical (d=2)")

#plt.plot(ell,100*(dCl_Cl_TT_b_a_ma),linewidth=0.5,alpha=0.7,color="tab:blue",label="boosted and masked $a_{\ell m}$")
plt.plot(ell_ma,100*(dCl_Cl_TT_b_a_ma_GF),linewidth=3, ls="--", color="tab:blue",label="Analytical (d=2)")

plt.plot(ell_ma,100*(dCl_Cl_TT_b_jeong_GF),linewidth=2,color="tab:red",label="Jeong")

plt.xlabel("$\ell$")
plt.ylabel("$\Delta C_\ell/C_\ell (\%)$")

plt.grid()
plt.legend()

plt.ylim(-1,1)
plt.xlim(100,1500)
plt.savefig("Numerical_vs_Analytical_vs_Jeong_d=2.png", dpi=150, bbox_inches="tight")


# In[ ]:


#plt.plot(ell,100*(dCl_Cl_TT_b_n_ma),linewidth=0.5,alpha=0.7,color="tab:orange",label="boosted and masked $a_{\ell m}$")
plt.plot(ell,100*(dCl_Cl_TT_b_n_ma_GF),linewidth=5,color="tab:orange",label="Gaussian smoothed")

#plt.plot(ell,100*(dCl_Cl_TT_b_a_ma),linewidth=0.5,alpha=0.7,color="tab:blue",label="boosted and masked $a_{\ell m}$")
plt.plot(ell,100*(-dCl_Cl_TT_b_a_ma_GF),linewidth=3, ls="--", color="tab:blue",label="Gaussian smoothed")

plt.plot(ell,100*(dCl_Cl_TT_b_jeong_GF),linewidth=2,color="tab:red",label="Jeong")

plt.xlabel("$\ell$")
plt.ylabel("$\Delta C_\ell/C_\ell (\%)$")

plt.grid()
plt.legend()

plt.ylim(-1,1)
plt.xlim(100,lmax)
plt.show()


# In[ ]:


#plt.plot(ell,100*(dCl_Cl_TT_b_n_ma),linewidth=0.5,alpha=0.7,color="tab:orange",label="boosted and masked $a_{\ell m}$")
plt.plot(ell,100*(dCl_Cl_TT_b_n_ma_GF),linewidth=5,color="tab:orange",label="Gaussian smoothed")

#plt.plot(ell,100*(dCl_Cl_TT_b_a_ma),linewidth=0.5,alpha=0.7,color="tab:blue",label="boosted and masked $a_{\ell m}$")
plt.plot(ell,100*(dCl_Cl_TT_b_a_ma_GF),linewidth=3, ls="--", color="tab:blue",label="Gaussian smoothed")

plt.plot(ell,100*(dCl_Cl_TT_b_jeong_GF),linewidth=2,color="tab:red",label="Jeong")

plt.xlabel("$\ell$")
plt.ylabel("$\Delta C_\ell/C_\ell (\%)$")

plt.grid()
plt.legend()

plt.ylim(-1,1)
plt.xlim(100,lmax)
plt.show()


# We see that the effect of the boost on masked portions of the sky is not negligible and can exceed a few percent. 

# In[ ]:




