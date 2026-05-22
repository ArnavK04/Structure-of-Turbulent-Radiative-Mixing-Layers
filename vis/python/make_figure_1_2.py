import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import bin_convert
import analyse_bin
import math
import matplotlib.cm as cm
import gc  
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patheffects as pe

# Global constants
MPI_DEF = True
if MPI_DEF:
    from mpi4py import MPI

# Units :-
ATOMIC_MASS = 1.660539e-24  # g
LENGTH = 3.08568e+18 
TIME   = 3.15576e+13 
MASS   = 4.91417e+31  
VELOCITY = 9.77793e+4  
DENSITY  = 1.67262e-24
ENERGY   = 4.69834e+41
POWER    = 1.48881e+28
PRESSURE = 1.59916e-14
TEMPERATURE = 71.2937
MU = 0.62
N_UNIT = DENSITY/(MU*ATOMIC_MASS) 
COOLING_UNIT = PRESSURE / (TIME * (N_UNIT**2))  # Cooling rate unit
CHI = 100.0
GAMMA = 5.0/3.0

# De- dimensionalising facotrs

global cold_frac, NY_fin, NY_init
cold_frac = 0.6666667
v_h = 28.18181822
T_h = 1.0e6
T_c = 1.0e4
P_0 = 14.02645
rho_h = 0.001
T_0 = 1.0e5
delU = 31.
T_0_code = T_0/TEMPERATURE
rho_0 = P_0/T_0_code
B_h = (5.0/2.0)*P_0/rho_h
TMAX = 0.9e6
TMIN = 1.1e4
NX = 280
NY = 1040
NZ = 280
max_level = 0
DY = 40./NY
T_inflection = (T_h + T_c)/2
NY_init = int(max(NY*(cold_frac - 0.35), 0))
NY_fin = int(min(NY*(cold_frac + 0.35), NY-1))

global dir
#dir = r"../../my_outputs/noSMR_2_3_cutoffISMcoolfn/fid3D_32_cool/bin/"
#dir = r"../../my_outputs/fiducial1040_cool2D/bin/"
#dir = r"../../../Downloads/Aryabhatta_data/snaps2xlessvel/"
#dir = r"../../../Downloads/Niagara_data/snaps5xlessdens/"
dir = r"../../../Downloads/Trillium_data/snapsinvertedfiducial400less256_1024_1_3cold/"
#dir = r"../../../Downloads/Astro_zenith_data/snapsfidcool2D/"
#dir = r"../../my_outputs/fid3D_halfbox_1040_cool/bin/"
#dir = r"../../../Downloads/Chandra_data/snaps2xmorevel/"
#dir = r"../../../Downloads/Niagara3Dfidnpz/"
#dir = r"../../my_outputs/noSMR_2_3_cutoffISMcoolfn/fid3D_5xmoredens_1040_cool/bin/"

from save_2D_arrays_3D import ISMCoolFn

# both slice functions same as make_avg_arrays.py
def slice_to_half(arr):
    global NY_init, NY_fin
    return arr[NY_init:NY_fin]

def slice_to_half_2D(arr):
    global NY_init, NY_fin
    return arr[NY_init:NY_fin,:]

def plot_profiles():

    global dir, time_0, P_0, delU, T_h, T_inflection, time_mid

    Y = np.linspace(-20,20, NY)/(delU*time_0)

    with np.load(dir + f'KH_1D_arrays_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
        rho_av = slice_to_half(f['rho_av'])
        rho_mw_av = slice_to_half(f['rho_mw_av'])
        rho_sig = slice_to_half(f['rho_sig'])
        rho_vx1_av = slice_to_half(f['rho_vx1_av'])
        rho_vx1_sig = slice_to_half(f['rho_vx1_sig'])
        rho_vx2_av = slice_to_half(f['rho_vx2_av'])
        rho_vx2_sig = slice_to_half(f['rho_vx2_sig'])
        energy_dens_av = slice_to_half(f['energy_dens_av'])
        energy_dens_sig = slice_to_half(f['energy_dens_sig'])
        temp_vol71to250 = f['temp_vol_av']

    # fitting a tanh function to temp71to250

    Tc0 = np.min(temp_vol71to250)
    Th0 = np.max(temp_vol71to250)
    dTdz = np.gradient(temp_vol71to250, Y)
    x0_guess = Y[np.argmax(np.abs(dTdz))]
    slope_max = np.max(np.abs(dTdz))
    z0_guess = (Th0 - Tc0) / (2 * slope_max)
    p0 = [x0_guess, z0_guess, Th0, Tc0]

    # fitting a tanh function to temp71to250
    from scipy.optimize import curve_fit
    def T_tanh_model(z, x0, z0, Th, Tc):
        return ((0.5 * (Th + Tc) + 0.5 * (Th - Tc) * np.tanh((z - x0) / z0)))
    popt, pcov = curve_fit(T_tanh_model, Y, temp_vol71to250, p0=p0, bounds=([-np.inf, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
    print(f'Fitted parameters: x0={popt[0]}, z0={popt[1]}, Th={popt[2]}, Tc={popt[3]}')
    z0 = popt[1]
    x0 = popt[0]
    Y -= x0
    
    T_tanh = lambda z, z0, Th, Tc: (0.5 * (Th - Tc) * np.tanh(z / z0) + 0.5 * (Th + Tc))

    perr = np.sqrt(np.diag(pcov))
    z0_err = perr[1]

    Y = slice_to_half(Y)
    temp_vol71to250 = slice_to_half(temp_vol71to250)

    fit_curve = T_tanh(Y, z0, T_h, T_c)/T_h

    with np.load(dir + f'KH_fluxes_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
        rho_avXv2_av = slice_to_half(f['rho_avXv2_av'])
        rho_avXv2_sig = slice_to_half(f['rho_avXv2_sig'])
        rhov2_av = slice_to_half(f['rhov2_av'])
        rhov2_sig = slice_to_half(f['rhov2_sig'])
        delrho_delv2_av = slice_to_half(f['delrho_delv2_av'])
        delrho_delv2_sig = slice_to_half(f['delrho_delv2_sig'])
        rhov2_avXv1_av = slice_to_half(f['rhov2_avXv1_av'])
        rhov2_avXv1_sig = slice_to_half(f['rhov2_avXv1_sig'])
        rhov2v1_av = slice_to_half(f['rhov2v1_av'])
        rhov2v1_sig = slice_to_half(f['rhov2v1_sig'])
        R_xz = slice_to_half(f['R_xz'])
        R_xz_sig = slice_to_half(f['R_xz_sig'])
        rhov2_avXv2_av = slice_to_half(f['rhov2_avXv2_av'])
        rhov2_avXv2_sig = slice_to_half(f['rhov2_avXv2_sig'])
        R_zz = slice_to_half(f['R_zz'])
        R_zz_sig = slice_to_half(f['R_zz_sig'])
        p_av = slice_to_half(f['p_av'])
        p_sig = slice_to_half(f['p_sig'])
        P_rhovx2sqr_av = slice_to_half(f['P_rhovx2sqr'])
        P_rhovx2sqr_sig = slice_to_half(f['P_rhovx2sqr_sig'])
        Be_av_rhov2_av = slice_to_half(f['Be_av_rhov2_av'])
        Be_av_rhov2_sig = slice_to_half(f['Be_av_rhov2_sig'])
        del_Be_del_rhov2_av = slice_to_half(f['del_Be_del_rhov2_av'])
        del_Be_del_rhov2_sig = slice_to_half(f['del_Be_del_rhov2_sig'])
        edot_cool_cum_dx2_av = slice_to_half(f['edot_cool_cum_dx2'])
        edot_cool_cum_dx2_sig = slice_to_half(f['edot_cool_cum_dx2_sig'])
        net_heating_av = slice_to_half(f['net_heating'])
        net_heating_sig = slice_to_half(f['net_heating_sig'])

    with np.load(dir + f'KH_fluxes_time_averaged{n1}to{n2-1}with{jump}.npz', 'r') as f:
        rho_avXv2_av71to130 = f['rho_avXv2_av']
        rho_avXv2_sig71to130 = f['rho_avXv2_sig']
        rhov2_av71to130 = f['rhov2_av']
        rhov2_sig71to130 = f['rhov2_sig']
        delrho_delv2_av71to130 = f['delrho_delv2_av']
        delrho_delv2_sig71to130 = f['delrho_delv2_sig']
        rhov2_avXv1_av71to130 = f['rhov2_avXv1_av']
        rhov2_avXv1_sig71to130 = f['rhov2_avXv1_sig']
        rhov2v1_av71to130 = f['rhov2v1_av']
        rhov2v1_sig71to130 = f['rhov2v1_sig']
        R_xz71to130 = f['R_xz']
        R_xz_sig71to130 = f['R_xz_sig']
        rhov2_avXv2_av71to130 = f['rhov2_avXv2_av']
        rhov2_avXv2_sig71to130 = f['rhov2_avXv2_sig']
        R_zz71to130 = f['R_zz']
        R_zz_sig71to130 = f['R_zz_sig']
        p_av71to130 = f['p_av']
        p_sig71to130 = f['p_sig']
        P_rhovx2sqr_av71to130 = f['P_rhovx2sqr']
        P_rhovx2sqr_sig71to130 = f['P_rhovx2sqr_sig']
        Be_av_rhov2_av71to130 = f['Be_av_rhov2_av']
        Be_av_rhov2_sig71to130 = f['Be_av_rhov2_sig']
        del_Be_del_rhov2_av71to130 = f['del_Be_del_rhov2_av']
        del_Be_del_rhov2_sig71to130 = f['del_Be_del_rhov2_sig']
        edot_cool_cum_dx2_av71to130 = f['edot_cool_cum_dx2']
        edot_cool_cum_dx2_sig71to130 = f['edot_cool_cum_dx2_sig']
        net_heating_av71to130 = f['net_heating']
        net_heating_sig71to130 = f['net_heating_sig']

    with np.load(dir + f'KH_fluxes_time_averaged{n2}to{n3-1}with{jump}.npz', 'r') as f:
        rho_avXv2_av131to190 = f['rho_avXv2_av']
        rho_avXv2_sig131to190 = f['rho_avXv2_sig']
        rhov2_av131to190 = f['rhov2_av']
        rhov2_sig131to190 = f['rhov2_sig']
        delrho_delv2_av131to190 = f['delrho_delv2_av']
        delrho_delv2_sig131to190 = f['delrho_delv2_sig']
        rhov2_avXv1_av131to190 = f['rhov2_avXv1_av']
        rhov2_avXv1_sig131to190 = f['rhov2_avXv1_sig']
        rhov2v1_av131to190 = f['rhov2v1_av']
        rhov2v1_sig131to190 = f['rhov2v1_sig']
        R_xz131to190 = f['R_xz']
        R_xz_sig131to190 = f['R_xz_sig']
        rhov2_avXv2_av131to190 = f['rhov2_avXv2_av']
        rhov2_avXv2_sig131to190 = f['rhov2_avXv2_sig']
        R_zz131to190 = f['R_zz']
        R_zz_sig131to190 = f['R_zz_sig']
        p_av131to190 = f['p_av']
        p_sig131to190 = f['p_sig']
        P_rhovx2sqr_av131to190 = f['P_rhovx2sqr']
        P_rhovx2sqr_sig131to190 = f['P_rhovx2sqr_sig']
        Be_av_rhov2_av131to190 = f['Be_av_rhov2_av']
        Be_av_rhov2_sig131to190 = f['Be_av_rhov2_sig']
        del_Be_del_rhov2_av131to190 = f['del_Be_del_rhov2_av']
        del_Be_del_rhov2_sig131to190 = f['del_Be_del_rhov2_sig']
        edot_cool_cum_dx2_av131to190 = f['edot_cool_cum_dx2']
        edot_cool_cum_dx2_sig131to190 = f['edot_cool_cum_dx2_sig']
        net_heating_av131to190 = f['net_heating']
        net_heating_sig131to190 = f['net_heating_sig']

    with np.load(dir + f'KH_fluxes_time_averaged{n3}to{n4}with{jump}.npz', 'r') as f:
        rho_avXv2_av191to250 = f['rho_avXv2_av']
        rho_avXv2_sig191to250 = f['rho_avXv2_sig']
        rhov2_av191to250 = f['rhov2_av']
        rhov2_sig191to250 = f['rhov2_sig']
        delrho_delv2_av191to250 = f['delrho_delv2_av']
        delrho_delv2_sig191to250 = f['delrho_delv2_sig']
        rhov2_avXv1_av191to250 = f['rhov2_avXv1_av']
        rhov2_avXv1_sig191to250 = f['rhov2_avXv1_sig']
        rhov2v1_av191to250 = f['rhov2v1_av']
        rhov2v1_sig191to250 = f['rhov2v1_sig']
        R_xz191to250 = f['R_xz']
        R_xz_sig191to250 = f['R_xz_sig']
        rhov2_avXv2_av191to250 = f['rhov2_avXv2_av']
        rhov2_avXv2_sig191to250 = f['rhov2_avXv2_sig']
        R_zz191to250 = f['R_zz']
        R_zz_sig191to250 = f['R_zz_sig']
        p_av191to250 = f['p_av']
        p_sig191to250 = f['p_sig']
        P_rhovx2sqr_av191to250 = f['P_rhovx2sqr']
        P_rhovx2sqr_sig191to250 = f['P_rhovx2sqr_sig']
        Be_av_rhov2_av191to250 = f['Be_av_rhov2_av']
        Be_av_rhov2_sig191to250 = f['Be_av_rhov2_sig']
        del_Be_del_rhov2_av191to250 = f['del_Be_del_rhov2_av']
        del_Be_del_rhov2_sig191to250 = f['del_Be_del_rhov2_sig']
        edot_cool_cum_dx2_av191to250 = f['edot_cool_cum_dx2']
        edot_cool_cum_dx2_sig191to250 = f['edot_cool_cum_dx2_sig']
        net_heating_av191to250 = f['net_heating']
        net_heating_sig191to250 = f['net_heating_sig']

    with np.load(dir + f'KH_1D_arrays_time_averaged{n1}to{n2-1}with{jump}.npz', 'r') as f:
        rho_av_71to130 = slice_to_half(f['rho_av'])
        rho_sig_71to130 = slice_to_half(f['rho_sig'])
        rho_vx1_av_71to130 = slice_to_half(f['rho_vx1_av'])
        rho_vx1_sig_71to130 = slice_to_half(f['rho_vx1_sig'])
        rho_vx2_av_71to130 = slice_to_half(f['rho_vx2_av'])
        rho_vx2_sig_71to130 = slice_to_half(f['rho_vx2_sig'])
        energy_dens_av_71to130 = slice_to_half(f['energy_dens_av'])
        energy_dens_sig_71to130 = slice_to_half(f['energy_dens_sig'])

    with np.load(dir + f'KH_1D_arrays_time_averaged{n2}to{n3-1}with{jump}.npz', 'r') as f:
        rho_av_131to190 = slice_to_half(f['rho_av'])
        rho_sig_131to190 = slice_to_half(f['rho_sig'])
        rho_vx1_av_131to190 = slice_to_half(f['rho_vx1_av'])
        rho_vx1_sig_131to190 = slice_to_half(f['rho_vx1_sig'])
        rho_vx2_av_131to190 = slice_to_half(f['rho_vx2_av'])
        rho_vx2_sig_131to190 = slice_to_half(f['rho_vx2_sig'])
        energy_dens_av_131to190 = slice_to_half(f['energy_dens_av'])
        energy_dens_sig_131to190 = slice_to_half(f['energy_dens_sig'])

    with np.load(dir + f'KH_1D_arrays_time_averaged{n3}to{n4}with{jump}.npz', 'r') as f:
        rho_av_191to250 = slice_to_half(f['rho_av'])
        rho_sig_191to250 = slice_to_half(f['rho_sig'])
        rho_vx1_av_191to250 = slice_to_half(f['rho_vx1_av'])
        rho_vx1_sig_191to250 = slice_to_half(f['rho_vx1_sig'])
        rho_vx2_av_191to250 = slice_to_half(f['rho_vx2_av'])
        rho_vx2_sig_191to250 = slice_to_half(f['rho_vx2_sig'])
        energy_dens_av_191to250 = slice_to_half(f['energy_dens_av'])
        energy_dens_sig_191to250 = slice_to_half(f['energy_dens_sig'])

    fig, ax = plt.subplots(4,2,figsize=(8, 10), constrained_layout=True)

    ax[0,0].plot(Y, rho_av_71to130/rho_h,  color='blue', linewidth=1.75, label=r"$ \langle \rangle_t : 146t_0-270t_0$")
    ax[0,0].plot(Y, rho_av_131to190/rho_h,  color='red', linewidth=1.75, label=r"$ \langle \rangle_t : 270t_0-394t_0$")
    ax[0,0].plot(Y, rho_av_191to250/rho_h,  color='green', linewidth=1.75, label=r"$ \langle \rangle_t : 394t_0-515t_0$")
    ax[0,0].plot(Y,rho_av/rho_h,  color='black', linewidth=3, label=r"$\langle \rho \rangle/\rho_h : : 146t_0-515t_0$")
    ax[0,0].plot(Y, rho_mw_av/rho_h,  color='black', linestyle='--', linewidth=1.75, label=r"$\langle \rho \rangle_{mw}/\rho_h$", alpha = 0.6)
    ax[0,0].fill_between(Y, (rho_av - rho_sig)/rho_h, (rho_av + rho_sig)/rho_h, color='gray', alpha=0.4)
    ax[0,0].set_xlim(Y[0], Y[-1])
    ax[0,0].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False, labelsize=10)
    #ax[0,0].set_ylim(-5, 130)
    ax[0,0].text(0.01, 0.97, r"$\langle \rho \rangle_t/\rho_h$", fontsize=16, ha='left', va='top', transform=ax[0,0].transAxes)
    #ax[0,0].legend(loc='lower left', fontsize=13, frameon=False, handlelength=1)
    ax[0,0].grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
    for spine in ax[0,0].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)  # thickness of border
        spine.set_edgecolor('black')  # color of border

    line1, = ax[0,1].plot(Y, rho_avXv2_av/(rho_h*delU),  color='red', linewidth=1.75, label=r"$\langle \rho \rangle \langle u_z \rangle/\rho_h \Delta u$")
    ax[0,1].fill_between(Y, (rho_avXv2_av - rho_avXv2_sig)/(rho_h*delU), (rho_avXv2_av + rho_avXv2_sig)/(rho_h*delU), color='red', alpha=0.2)
    line2, = ax[0,1].plot(Y, delrho_delv2_av/(rho_h*delU),  color='blue', linewidth=1.75, label=r"$\langle \delta \rho \delta u_z \rangle/\rho_h \Delta u$")
    ax[0,1].fill_between(Y, (delrho_delv2_av - delrho_delv2_sig)/(rho_h*delU), (delrho_delv2_av + delrho_delv2_sig)/(rho_h*delU), color='blue', alpha=0.2)
    line3, = ax[0,1].plot(Y,rho_vx2_av/(rho_h*delU),  color='black', linewidth=3,label=r"$\langle \rho u_z \rangle/\rho_h \Delta u$")
    ax[0,1].fill_between(Y, (rho_vx2_av - rho_vx2_sig)/(rho_h*delU), (rho_vx2_av + rho_vx2_sig)/(rho_h*delU), color='gray', alpha=0.5)
    #ax[0,1].legend(loc='lower left', fontsize=13, frameon=False, handlelength=1)
    ax[0,1].set_xlim(Y[0], Y[-1])
    #ax[0,1].set_ylim(1.2*np.min(rho_avXv2_av/(rho_h*delU)), 1.2*np.max(delrho_delv2_av/(rho_h*delU)))
    ax[0,1].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False, labelsize=10)
    ax[0,1].text(0.01, 0.97, r"$Mass\ density\ flux$", fontsize=16, ha='left', va='top', transform=ax[0,1].transAxes)
    ax[0,1].grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
    for spine in ax[0,1].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)  # thickness of border
        spine.set_edgecolor('black')  # color of border
    legend1 = ax[0,1].legend(handles=[line1], loc='lower center', bbox_to_anchor=(0.235, 0.49),
                    ncol=1, frameon=False, fontsize=13, handlelength=1)

    # Add the first legend manually so the second doesn't overwrite it
    ax[0,1].add_artist(legend1)

    # Second legend: below the plot
    ax[0,1].legend(handles=[line2, line3], loc='upper center', bbox_to_anchor=(0.25, 0.47),
            ncol=1, frameon=False, fontsize=13, handlelength=1)

    line4, = ax[1,1].plot(Y,R_xz/(rho_h*delU*delU),  color='blue', linewidth=1.75, label=r"$\langle \delta \rho u_z \delta u_x \rangle/\rho_h \Delta u^2$")
    ax[1,1].fill_between(Y, (R_xz - R_xz_sig)/(rho_h*delU*delU), (R_xz + R_xz_sig)/(rho_h*delU*delU), color='blue', alpha=0.2)
    line5, = ax[1,1].plot(Y,rhov2_avXv1_av/(rho_h*delU*delU),  color='red', linewidth=1.75, label=r"$\langle \rho u_z \rangle \langle u_x \rangle/\rho_h \Delta u^2$")
    ax[1,1].fill_between(Y, (rhov2_avXv1_av - rhov2_avXv1_sig)/(rho_h*delU*delU), (rhov2_avXv1_av + rhov2_avXv1_sig)/(rho_h*delU*delU), color='red', alpha=0.2)
    line6, = ax[1,1].plot(Y,rhov2v1_av/(rho_h*delU*delU),  color='black', linewidth=3, label=r"$\langle \rho u_x u_z \rangle/\rho_h \Delta u^2$")
    ax[1,1].fill_between(Y, (rhov2v1_av - rhov2v1_sig)/(rho_h*delU*delU), (rhov2v1_av + rhov2v1_sig)/(rho_h*delU*delU), color='gray', alpha=0.5)
    ax[1,1].legend(loc='lower left', fontsize=13, frameon=False, handlelength=1)
    ax[1,1].set_xlim(Y[0], Y[-1])
    #ax[1,1].set_ylim(-1.05, 0.55)
    ax[1,1].text(0.01, 0.97, r"$X-momentum\ density\ flux$", fontsize=16, ha='left', va='top', transform=ax[1,1].transAxes)
    ax[1,1].grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
    ax[1,1].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False, labelsize=10)
    for spine in ax[1,1].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)  # thickness of border
        spine.set_edgecolor('black')  # color of border
    legend1 = ax[1,1].legend(handles=[line4], loc='lower center', bbox_to_anchor=(0.3, 0.65),
                    ncol=1, frameon=False, fontsize=13, handlelength=1)

    # Add the first legend manually so the second doesn't overwrite it
    ax[1,1].add_artist(legend1)

    # Second legend: below the plot
    ax[1,1].legend(handles=[ line5, line6], loc='upper center', bbox_to_anchor=(0.29, 0.35),
            ncol=1, frameon=False, fontsize=13, handlelength=1)

    ax[2,1].plot(Y,p_av/(P_0),  color='green', linewidth=1.75, label=r"$\langle p \rangle/p_0$")
    ax[2,1].fill_between(Y, (p_av - p_sig)/(P_0), (p_av + p_sig)/(P_0), color='green', alpha=0.2)
    ax[2,1].plot(Y,rhov2_avXv2_av/(P_0),  color='blue', linewidth=1.75, label=r"$\langle \rho u_z \rangle \langle u_z \rangle/p_0$")
    ax[2,1].fill_between(Y, (rhov2_avXv2_av - rhov2_avXv2_sig)/(P_0), (rhov2_avXv2_av + rhov2_avXv2_sig)/(P_0), color='blue', alpha=0.2)
    ax[2,1].plot(Y,R_zz/(P_0),  color='red', linewidth=1.75, label=r"$\langle \delta \rho u_z \delta u_z \rangle/p_0$")
    ax[2,1].fill_between(Y, (R_zz - R_zz_sig)/(P_0), (R_zz + R_zz_sig)/(P_0), color='red', alpha=0.2)
    ax[2,1].plot(Y,P_rhovx2sqr_av/(P_0),  color='black', linewidth=3, label=r"$\langle p  + \rho {u_z}^2 \rangle/p_0$")
    ax[2,1].fill_between(Y, (P_rhovx2sqr_av - P_rhovx2sqr_sig)/(P_0), (P_rhovx2sqr_av + P_rhovx2sqr_sig)/(P_0), color='gray', alpha=0.5)
    ax[2,1].legend(loc='center left', fontsize=13, frameon=False, handlelength=1, bbox_to_anchor=(0.00, 0.45))
    ax[2,1].set_xlim(Y[0], Y[-1])
    ax[2,1].grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
    ax[2,1].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False, labelsize=10)
    #ax[2,1].set_ylim(-0.1, 1.3)
    ax[2,1].text(0.01, 0.97, r"$Z-momentum\ density\ flux$", fontsize=16, ha='left', va='top', transform=ax[2,1].transAxes)
    for spine in ax[2,1].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)  # thickness of border
        spine.set_edgecolor('black')  # color of border
    

    line7, = ax[3,1].plot(Y,edot_cool_cum_dx2_av/(P_0*delU),  color='blue', linewidth=1.75, label=r"$\int \langle n^2 \Lambda (T) \rangle \, dz /p_0 \Delta u$")
    ax[3,1].fill_between(Y, (edot_cool_cum_dx2_av - edot_cool_cum_dx2_sig)/(P_0*delU), (edot_cool_cum_dx2_av + edot_cool_cum_dx2_sig)/(P_0*delU), color='blue', alpha=0.2)
    line8, = ax[3,1].plot(Y,Be_av_rhov2_av/(P_0*delU),  color='green', linewidth=1.75, label=r"$\langle \mathcal{B} \rangle \langle \rho u_z \rangle/p_0 \Delta u$")
    ax[3,1].fill_between(Y, (Be_av_rhov2_av - Be_av_rhov2_sig)/(P_0*delU), (Be_av_rhov2_av + Be_av_rhov2_sig)/(P_0*delU), color='green', alpha=0.2)
    line9, = ax[3,1].plot(Y,del_Be_del_rhov2_av/(P_0*delU),  color='red', linewidth=1.75, label=r"$\langle \delta \mathcal{B} \delta \rho u_z \rangle/p_0 \Delta u$")
    ax[3,1].fill_between(Y, (del_Be_del_rhov2_av - del_Be_del_rhov2_sig)/(P_0*delU), (del_Be_del_rhov2_av + del_Be_del_rhov2_sig)/(P_0*delU), color='red', alpha=0.2)
    line10, = ax[3,1].plot(Y,net_heating_av/(P_0*delU),  color='black', linewidth=3, label=r"$\langle \epsilon_{net} \rangle/p_0 \Delta u$")
    ax[3,1].fill_between(Y, (net_heating_av - net_heating_sig)/(P_0*delU), (net_heating_av + net_heating_sig)/(P_0*delU), color='gray', alpha=0.5)
    ax[3,1].legend(loc='lower left', fontsize=13, frameon=False, handlelength=1)
    ax[3,1].set_xlim(Y[0], Y[-1])
    ax[3,1].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=10)
    ax[3,1].grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
    #ax[3,1].set_ylim(1.2*np.min(Be_av_rhov2_av/(P_0*delU)), 1.2*np.max(edot_cool_cum_dx2_av/(P_0*delU)))
    ax[3,1].text(0.01, 0.97, r"$Energy\ density\ flux$", fontsize=16, ha='left', va='top', transform=ax[3,1].transAxes)
    for spine in ax[3,1].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)  # thickness of border
        spine.set_edgecolor('black')  # color of border
    ax[3,1].set_xlabel(r"$z/\Delta u t_0$", fontsize=14)
    legend1 = ax[3,1].legend(handles=[line7, line8], loc='lower center', bbox_to_anchor=(0.3, 0.45),
                    ncol=1, frameon=False, fontsize=13, handlelength=1)

    # Add the first legend manually so the second doesn't overwrite it
    ax[3,1].add_artist(legend1)

    # Second legend: below the plot
    ax[3,1].legend(handles=[line9, line10], loc='upper center', bbox_to_anchor=(0.255, 0.5),
            ncol=1, frameon=False, fontsize=13, handlelength=1)

    with np.load(dir + f'KH_1D_arrays_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
        vx1_vol71to250 = slice_to_half(f['vx1_vol_av'])/delU
        vx2_vol71to250 = slice_to_half(f['vx2_vol_av'])/delU
        vx2_vol71to250_sig = slice_to_half(f['vx2_vol_sig'])/delU
        vx1_vol71to250_sig = slice_to_half(f['vx1_vol_sig'])/delU
        temp_vol71to250 = slice_to_half(f['temp_vol_av'])/T_h
        temp_vol71to250_sig = slice_to_half(f['temp_vol_sig'])/T_h
        Be_vol71to250 = slice_to_half(f['Be_vol_av'])/B_h
        vx1_mw71to250 = slice_to_half(f['vx1_mw_av'])/delU
        vx2_mw71to250 = slice_to_half(f['vx2_mw_av'])/delU
        temp_mw71to250 = slice_to_half(f['temp_mw_av'])/T_h
        Be_mw71to250 = slice_to_half(f['Be_mw_av'])/B_h
    with np.load(dir + f'KH_1D_arrays_time_averaged{n1}to{n2-1}with{jump}.npz', 'r') as f:
        vx1_vol71to130 = slice_to_half(f['vx1_vol_av'])/delU
        vx2_vol71to130 = slice_to_half(f['vx2_vol_av'])/delU
        temp_vol71to130 = slice_to_half(f['temp_vol_av'])/T_h
        Be_vol71to130 = slice_to_half(f['Be_vol_av'])/B_h
    with np.load(dir + f'KH_1D_arrays_time_averaged{n2}to{n3-1}with{jump}.npz', 'r') as f:
        vx1_vol131to190 = slice_to_half(f['vx1_vol_av'])/delU
        vx2_vol131to190 = slice_to_half(f['vx2_vol_av'])/delU
        temp_vol131to190 = slice_to_half(f['temp_vol_av'])/T_h
        Be_vol131to190 = slice_to_half(f['Be_vol_av'])/B_h
    with np.load(dir + f'KH_1D_arrays_time_averaged{n3}to{n4}with{jump}.npz', 'r') as f:
        vx1_vol191to250 = slice_to_half(f['vx1_vol_av'])/delU
        vx2_vol191to250 = slice_to_half(f['vx2_vol_av'])/delU
        temp_vol191to250 = slice_to_half(f['temp_vol_av'])/T_h
        Be_vol191to250 = slice_to_half(f['Be_vol_av'])/B_h

    ax[1,0].plot(Y, vx1_vol71to130, color='blue', linewidth=1.75, alpha=0.6)
    ax[1,0].plot(Y, vx1_vol131to190, color='red', linewidth=1.75, alpha=0.6)
    ax[1,0].plot(Y, vx1_vol191to250, color='green', linewidth=1.75, alpha=0.6)
    ax[1,0].plot(Y, vx1_mw71to250, color='black', linestyle='--', linewidth=1.75, label=r"$mass-weighted$", alpha=0.6)
    ax[1,0].plot(Y, vx1_vol71to250, color='black', linewidth=3, label=r"$volume-weighted$")
    ax[1,0].fill_between(Y, (vx1_vol71to250 - vx1_vol71to250_sig), (vx1_vol71to250 + vx1_vol71to250_sig), color='gray', alpha=0.4)
    ax[1,0].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False, labelsize=10)
    ax[1,0].text(0.01, 0.92, r"$\langle u_x \rangle_t/ \Delta u$", fontsize=16, ha='left', va='top', transform=ax[1,0].transAxes)
    ax[1,0].grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
    ax[1,0].legend(loc='center left', fontsize=11, frameon=False, handlelength=1)
    #ax[0].legend(loc='best', fontsize=12, frameon=False, handlelength=1, bbox_to_anchor=(0.5, 0.35))
    ax[1,0].set_xlim(Y[0], Y[-1])
    for spine in ax[1,0].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)  # thickness of border
        spine.set_edgecolor('black')  # color of border

    ax[2,0].plot(Y, vx2_vol71to130, color='blue', linewidth=1.75, label=r"$\langle 146t_0-270t_0 \rangle_t $", alpha=0.6)
    ax[2,0].plot(Y, vx2_vol131to190, color='red', linewidth=1.75, label=r"$\langle 270t_0-394t_0 \rangle_t $", alpha=0.6)
    ax[2,0].plot(Y, vx2_vol191to250, color='green', linewidth=1.75, label=r"$\langle 394t_0-515t_0 \rangle_t $", alpha=0.6)
    ax[2,0].plot(Y, vx2_mw71to250, color='black', linestyle='--', linewidth=1.75, label=r"$\langle 146t_0-515t_0 \rangle_t $", alpha=0.6)
    ax[2,0].plot(Y, vx2_vol71to250, color='black', linewidth=3, label=r"$\langle 146t_0-515t_0 \rangle_t $")
    ax[2,0].fill_between(Y, (vx2_vol71to250 - vx2_vol71to250_sig), (vx2_vol71to250 + vx2_vol71to250_sig), color='gray', alpha=0.4)
    ax[2,0].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False, labelsize=10)
    ax[2,0].text(0.01, 0.9, r"$\langle u_z \rangle_t/ \Delta u$", fontsize=16, ha='left', va='top', transform=ax[2,0].transAxes)
    ax[2,0].grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
    ax[2,0].legend(loc='best', fontsize=13, frameon=False, handlelength=1)#, bbox_to_anchor=(0.65, 0.6))
    ax[2,0].set_xlim(Y[0], Y[-1])
    #ax[2,0].set_ylim(1.2*np.min(vx2_vol71to250), 0.05)
    for spine in ax[2,0].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)  # thickness of border
        spine.set_edgecolor('black')  # color of border

    ax[3,0].plot(Y, temp_vol71to130, color='blue', linewidth=1.75, alpha=0.6)
    ax[3,0].plot(Y, temp_vol131to190, color='red', linewidth=1.75, alpha=0.6)
    ax[3,0].plot(Y, temp_vol191to250, color='green', linewidth=1.75, alpha=0.6)
    ax[3,0].plot(Y, temp_mw71to250, color='black', linestyle='--', linewidth=1.75, alpha=0.6)
    ax[3,0].plot(Y, temp_vol71to250, color='black', linewidth=3)

    ax[3,0].plot(Y, fit_curve, color='orange', linestyle='--', linewidth=1.25, label= r"tanh fit, $\frac{z_0}{\Delta u t_0}$= " + f'{round(z0,2)}')
    ax[3,0].grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
    ax[3,0].fill_between(Y, (temp_vol71to250 - temp_vol71to250_sig), (temp_vol71to250 + temp_vol71to250_sig), color='gray', alpha=0.4)
    ax[3,0].tick_params(left=True, bottom=True, labelleft=True, labelbottom=True, labelsize=10)
    ax[3,0].text(0.01, 0.92, r"$\langle T \rangle_t/ T_h$", fontsize=16, ha='left', va='top', transform=ax[3,0].transAxes)
    ax[3,0].legend(loc='center left', fontsize=13, frameon=False, handlelength=1, bbox_to_anchor=(0.25, 0.65))
    ax[3,0].set_xlim(Y[0], Y[-1])
    ax[3,0].plot(Y, Be_vol71to250, color='cyan', linewidth=0.75, label=r"$\langle \frac{\mathcal{B}\mu m_p}{2.5k_B T_h}\rangle_t $" + "\n" + r"$volume-weighted$")
    ax[3,0].legend(loc='center left', fontsize=11, frameon=False, handlelength=1)
    #ax[3,0].set_ylim(-0.1, 1.1)
    for spine in ax[3,0].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)  # thickness of border
        spine.set_edgecolor('black')  # color of border
    ax[3,0].set_xlabel(r"$z/\Delta u t_0$", fontsize=16)

    plt.savefig(dir + 'figure2.png', dpi=600, bbox_inches='tight')
    plt.clf()
    plt.close('all')

    print("Done!")
    with np.load(dir + f'KH_1D_arrays_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
        uz = f['vx2_vol_av']/delU
        uz_sig = f['vx2_vol_sig']/delU
    with np.load(dir + f'KH_fluxes_time_averaged{n1}to{n4}with{jump}.npz', 'r') as f:
        R_xz = f['R_xz']/(rho_h*delU*delU)
        R_xz_sig = f['R_xz_sig']/(rho_h*delU*delU)
        edot_cool_cum_dx2_av = f['edot_cool_cum_dx2']/(P_0*delU)
        edot_cool_cum_dx2_sig = f['edot_cool_cum_dx2_sig']/(P_0*delU)
        R_zz = f['R_zz']/(rho_h*delU*delU)
        R_zz_sig = f['R_zz_sig']/(rho_h*delU*delU)
        del_Be_del_rhov2_av = f['del_Be_del_rhov2_av']/(P_0*delU)
        del_Be_del_rhov2_sig = f['del_Be_del_rhov2_sig']/(P_0*delU)
        delrho_delv2_av = f['delrho_delv2_av']/(rho_h*delU)
        delrho_delv2_sig = f['delrho_delv2_sig']/(rho_h*delU)

    Y = np.linspace(-20,20, NY)/(delU*time_0)

    print("done!")
    print(f'uz,h = {uz[-1]} +- {uz_sig[-1]}')
    print(f'z0 = {z0} +- {z0_err}')
    print(f'Sigma_dot_cool = {edot_cool_cum_dx2_av[-1]} +- {edot_cool_cum_dx2_sig[-1]}')

    # find where delrho_delv2_av is maximum and print the value and location
    max_index = np.argmax(delrho_delv2_av)
    print(f'delrho_delv2 max value = {delrho_delv2_av[max_index]} +- {delrho_delv2_sig[max_index]} at z = {Y[max_index]}')
    print(f'R_xz min value = {R_xz.min()} +- {R_xz_sig[np.argmin(R_xz)]} at z = {Y[np.argmin(R_xz)]}')
    print(f'R_zz max value = {R_zz.max()} +- {R_zz_sig[np.argmax(R_zz)]} at z = {Y[np.argmax(R_zz)]}')
    print(f'Q_t min value = {del_Be_del_rhov2_av.min()} +- {del_Be_del_rhov2_sig[np.argmin(del_Be_del_rhov2_av)]} at z = {Y[np.argmin(del_Be_del_rhov2_av)]}')

    # save all above values to a text file
    with open(dir + 'KH_fluxes_summary.txt', 'w') as f:
        f.write(f'uz,h = {uz[-1]} +- {uz_sig[-1]}\n')
        f.write(f'z0 = {z0} +- {z0_err}\n')
        f.write(f'Sigma_dot_cool = {edot_cool_cum_dx2_av[-1]} +- {edot_cool_cum_dx2_sig[-1]}\n')
        f.write(f'delrho_delv2 max value = {delrho_delv2_av[max_index]} +- {delrho_delv2_sig[max_index]} at z = {Y[max_index]}\n')
        f.write(f'R_xz min value = {R_xz.min()} +- {R_xz_sig[np.argmin(R_xz)]} at z = {Y[np.argmin(R_xz)]}\n')
        f.write(f'R_zz max value = {R_zz.max()} +- {R_zz_sig[np.argmax(R_zz)]} at z = {Y[np.argmax(R_zz)]}\n')
        f.write(f'Q_t min value = {del_Be_del_rhov2_av.min()} +- {del_Be_del_rhov2_sig[np.argmin(del_Be_del_rhov2_av)]} at z = {Y[np.argmin(del_Be_del_rhov2_av)]}\n')
        f.write(f'time_0 = {time_0}\n')
        f.write(f'time_mid = {time_mid}\n')
        f.write(f'time_mid/time_0 = {time_mid/time_0}\n')

def make_2D_paper_plots(i):

    global dir, NX, NY, NZ, F, max_level, delU, time_0, rho_h, T_inflection, jump

    with np.load(dir + 'KH_2D_Z_0_' + str(i).zfill(5) + '_C1.npz', 'r') as f:
        den = f['den']
        prs = f['prs']
        ex = f['extent']/(delU*time_0)
        temp = f['temp']
        number = f['number']
        F = f['Factor']
        v_turb_rms = f['v_turb_rms']

    with np.load(dir + 'KH_1D_arrays_snapshot_' + str(i).zfill(5) + 'C1.npz', 'r') as f:
        prs_vol = f['prs_vol']
        den_vol = f['den_vol']
        prs_vol = f['prs_vol']
        emis_vol = f['emis_vol']
        vx1_turb_rms_vol = f['vx1_turb_mw']         # it's actually mass-weighted, but name was set already so didnt change
        vx2_turb_rms_vol = f['vx2_turb_mw']
        vx3_turb_rms_vol = f['vx3_turb_mw']
        v_turb_rms_vol = f['v_turb_rms_mw']
    
    with np.load(dir + f"KH_1D_arrays_time_averaged{n1}to{n4}with{jump}.npz", 'r') as f:
        rho_av71to250 = f['rho_av']
        Y_range_final = f['Y_range_final']/(delU*time_0)
        emis_vol71to250 = f['emis_vol_av'] 
        temp_vol71to250 = f['temp_vol_av']   
        
    with np.load(dir + f"KH_fluxes_time_averaged{n1}to{n4}with{jump}.npz", 'r') as f:
        p_av71to250 = f['p_av']

    fig, ax = plt.subplots(2,2,figsize=(6, 10))
    plt.subplots_adjust(wspace=0.125, hspace=0.125)


    # using the corrected Y_lims that account for the movement of the interface due to the TRML velocity, so that the same region is plotted for all snapshots.
    with np.load(dir + f'KH_1D_arrays_snapshot{n_i}_{n_f}_' + str(i).zfill(5) + f'_C{F}_y_lims_corrected.npz', 'r') as f:
        Y_lims = f['Y_lims']/(delU*time_0)
        v_TRML_integrated = f['v_TRML_integrated']
    
    print(f"Y_lim range = {Y_lims[0]} to {Y_lims[-1]}")
    print(f"Y_range_final range = {Y_range_final[0]} to {Y_range_final[-1]}")

    Tc0 = np.min(temp_vol71to250)
    Th0 = np.max(temp_vol71to250)
    dTdz = np.gradient(temp_vol71to250, Y_lims)
    x0_guess = Y_lims[np.argmax(np.abs(dTdz))]
    slope_max = np.max(np.abs(dTdz))
    z0_guess = (Th0 - Tc0) / (2 * slope_max)
    p0 = [x0_guess, z0_guess, Th0, Tc0]

    # fitting a tanh function to temp71to250
    from scipy.optimize import curve_fit
    def T_tanh_model(z, x0, z0, Th, Tc):
        return ((0.5 * (Th + Tc) + 0.5 * (Th - Tc) * np.tanh((z - x0) / z0)))
    popt, pcov = curve_fit(T_tanh_model, Y_lims, temp_vol71to250, p0=p0, bounds=([-np.inf, 0, 0, 0], [np.inf, np.inf, np.inf, np.inf]))
    print(f'Fitted parameters: x0={popt[0]}, z0={popt[1]}, Th={popt[2]}, Tc={popt[3]}')
    z0 = popt[1]
    y0 = popt[0]


    Y_lims -= y0
    Y_range_final -= y0

    print(f"Corrected Y_lim range = {Y_lims[0]} to {Y_lims[-1]}")
    print(f"Corrected Y_range_final range = {Y_range_final[0]} to {Y_range_final[-1]}")

    im1 = ax[0,0].imshow(slice_to_half_2D(den)/rho_h, aspect='auto', origin='lower',cmap='inferno',extent = (ex[0], ex[1], Y_lims[NY_init], Y_lims[NY_fin]), vmin=-10, vmax=110)

    # Adding streamlines
    X_coords = np.linspace(ex[0], ex[1], NX)
    Y_coords = Y_lims
    Y_coords = slice_to_half(Y_coords)
    X, Y = np.meshgrid(X_coords, Y_coords)
    with np.load(dir + 'KH_2D_Z_0_' + str(i).zfill(5) + '_C1.npz', 'r') as f:
        vx1 = slice_to_half_2D(f['vx1'])
        vx2 = slice_to_half_2D(f['vx2'])

    plt.sca(ax[0,0])
    plt.streamplot(X, Y, vx1, vx2, color='gray', linewidth=1, density=1, arrowsize=1, arrowstyle='->')

    axlogx1 = ax[0,0].twiny()
    ax[0,0].set_xlim(ex[0], ex[1])
    ax[0,0].set_xscale('linear')
    axlogx1.set_xscale('linear')
    axlogx1.set_xlim(-10, 110)
    ax[0,0].set_ylim(Y_lims[NY_init], Y_lims[NY_fin])
    axlogx1.set_ylim(Y_lims[NY_init], Y_lims[NY_fin])

    cax = inset_axes(axlogx1, width="100%", height="3%", loc='lower center',
                    bbox_to_anchor=(0, -0.05, 1, 1),
                    bbox_transform=axlogx1.transAxes, borderpad=0)
    ax[0,0].text(0.75, 0.93, r"$\rho/\rho_h$", transform=ax[0,0].transAxes, ha='left', fontsize=16, color='white')
    axlogx1.plot(rho_av71to250/rho_h, Y_range_final, color='blue', linewidth=2, linestyle = '-', label=r"$\langle 146t_0-515t_0 \rangle_T$")
    axlogx1.plot(slice_to_half(den_vol)/rho_h, slice_to_half(Y_lims), color='grey', linewidth=2, label=r"$\langle \rho \rangle/\rho_h$")
    cbar = plt.colorbar(im1, cax=cax, orientation='horizontal')
    ax[0,0].tick_params(left=True, top = True, bottom=False, labeltop = True, labelleft=True, labelbottom=False)
    axlogx1.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False, labelleft=False, labelright=False)
    ax[0,0].set_xlabel(r"$x/\Delta u t_0$", fontsize=14)
    ax[0,0].set_ylabel(r"$z/\Delta u t_0$", fontsize=14)
    ax[0,0].xaxis.set_label_position('top')
    ax[0,0].yaxis.set_ticks_position('left')
    for spine in ax[0,0].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)  # thickness of border
        spine.set_edgecolor('black')  # color of border

    

    im2 = ax[0,1].imshow(slice_to_half_2D(prs)/P_0, aspect='auto', origin='lower', cmap='inferno', extent = (ex[0], ex[1], Y_lims[NY_init], Y_lims[NY_fin]), vmin = 0.3, vmax = 1.5)
    ax[0,1].set_xlim(ex[0], ex[1])
    ax[0,1].set_xscale('linear')
    axlogx2 = ax[0,1].twiny()
    axlogx2.set_xscale('linear')
    axlogx2.set_xlim(0.3, 1.5)
    ax[0,1].set_ylim(Y_lims[NY_init], Y_lims[NY_fin])
    axlogx2.set_ylim(Y_lims[NY_init], Y_lims[NY_fin])
    cax2 = inset_axes(axlogx2, width="100%", height="3%", loc='lower center',
                    bbox_to_anchor=(0, -0.05, 1, 1),
                    bbox_transform=axlogx2.transAxes, borderpad=0)
    axlogx2.plot(p_av71to250/P_0, Y_range_final, color='blue', linewidth=2, linestyle = '-', label=r"$\langle 146t_0-515t_0 \rangle_T$")
    axlogx2.plot(slice_to_half(prs_vol)/P_0, slice_to_half(Y_lims), color='grey', linewidth=2, label=r"$\langle p \rangle/p_0$")
    ax[0,1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, top = True, labeltop = True)
    axlogx2.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False, labelleft=False, labelright=False)
    ax[0,1].set_xlabel(r"$x/\Delta u t_0$", fontsize=14)
    ax[0,1].xaxis.set_label_position('top')
    ax[0,1].text(0.1, 0.93, r"$p/p_0$", transform=ax[0,1].transAxes, ha='left', fontsize=16, color='white')
    cbar2 = plt.colorbar(im2, cax=cax2, orientation='horizontal')

    for spine in ax[0,1].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)  # thickness of border
        spine.set_edgecolor('black')  # color of border

    b = 3. * prs
    em = 2. * den * den * np.vectorize(ISMCoolFn, otypes='d')(temp)/COOLING_UNIT
    t_cool = np.divide(b, (em), out=np.full_like(b, np.inf, dtype=float), where=em!=0)  # Cooling time in code units
    t_cool_av = 3.*prs_vol/(2.*(emis_vol)+1e-10)
    t_cool_av71to250 = 3.*p_av71to250/(2.*(emis_vol71to250)+1e-10)
    im3 = ax[1,0].imshow(slice_to_half_2D(t_cool)/time_0, aspect='auto', origin='lower', norm = 'log', cmap='inferno', extent = (ex[0], ex[1], Y_lims[NY_init], Y_lims[NY_fin]),vmin = 1e-1*np.min(slice_to_half(t_cool_av71to250)/time_0),vmax = 1e3*np.min(slice_to_half(t_cool_av71to250)/time_0))

    ax[1,0].set_xlim(ex[0], ex[1])
    ax[1,0].set_xscale('linear')
    ax_logx3 = ax[1,0].twiny()
    ax_logx3.set_xscale('log')
    ax[1,0].set_ylim(Y_lims[NY_init], Y_lims[NY_fin])
    ax_logx3.set_ylim(Y_lims[NY_init], Y_lims[NY_fin])
    ax_logx3.set_xlim(1e-1*np.min(slice_to_half(t_cool_av71to250)/time_0), 1e3*np.min(slice_to_half(t_cool_av71to250)/time_0))
    ax_logx3.plot(t_cool_av71to250/time_0, Y_range_final, color='blue', linewidth=2, label=r"$ \langle \rangle_t : 146t_0-515t_0$")
    ax_logx3.plot(slice_to_half(t_cool_av)/time_0, slice_to_half(Y_lims), color='grey', linewidth=2, label=r"$t \approx 350t_0$")
    cax3 = inset_axes(ax[1,0], width="100%", height="3%", loc='lower center',
                    bbox_to_anchor=(0, -0.05, 1, 1),
                    bbox_transform=ax[1,0].transAxes, borderpad=0)
    cbar3 = plt.colorbar(im3, cax=cax3, orientation='horizontal')
    ax[1,0].text(0.1, 0.93, r"$t_{cool}/t_0$", transform=ax[1,0].transAxes, ha='left', fontsize=16)
    legend3 = ax_logx3.legend(frameon=False, loc = 'lower left', fontsize=12)
    ax[1,0].yaxis.set_ticks_position('left')
    for spine in ax[1,0].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)  # thickness of border
        spine.set_edgecolor('black')  # color of border
    ax_logx3.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False, labelleft=False, labelright=False, length=0)
    ax[1,0].tick_params(left=True, bottom=False, labelleft=True, labelbottom=False, top=False, labeltop=False)
    ax_logx3.set_xticks([])
    ax_logx3.xaxis.set_visible(False)

    im4 = ax[1,1].imshow((slice_to_half_2D(v_turb_rms))/delU, aspect='auto', origin='lower', norm = 'log', cmap='inferno',extent = (ex[0], ex[1], Y_lims[NY_init], Y_lims[NY_fin]), vmin=1e-2, vmax=5)
    ax[1,1].set_xlim(ex[0], ex[1])
    ax[1,1].set_xscale('linear')
    #ax[1,0].set_ylabel(r"$z/\Delta u t_0$", fontsize=14)
    cax4 = inset_axes(ax[1,1], width="100%", height="3%", loc='lower center',
                    bbox_to_anchor=(0, -0.05, 1, 1),
                    bbox_transform=ax[1,1].transAxes, borderpad=0)
    cbar4 = plt.colorbar(im4, cax=cax4, orientation='horizontal')
    ax[1,1].text(0.1, 0.93, r"$u_{turb,rms}/\Delta u$", transform=ax[1,1].transAxes, ha='left', fontsize=15, color='white')

    ax_logx4 = ax[1,1].twiny()
    ax_logx4.set_xscale('log')
    ax_logx4.set_xlim(1e-2, 5)
    ax[1,1].set_ylim(Y_lims[NY_init], Y_lims[NY_fin])
    ax_logx4.set_ylim(Y_lims[NY_init], Y_lims[NY_fin])
    ax_logx4.plot(slice_to_half(vx1_turb_rms_vol)/delU, slice_to_half(Y_lims), color='lime', linewidth=1, linestyle = '--',label=r"$x$")
    ax_logx4.plot(slice_to_half(vx2_turb_rms_vol)/delU, slice_to_half(Y_lims), color='yellow', linewidth=1,linestyle = '--', label=r"$y$")
    ax_logx4.plot(slice_to_half(vx3_turb_rms_vol)/delU, slice_to_half(Y_lims), color='cyan', linewidth=1,linestyle = '--', label=r"$z$")
    ax_logx4.tick_params(top=False, bottom=False, labeltop=False, labelbottom=False, left=False, length=0)
    ax[1,1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False, top = False, labeltop=False)
    ax_logx4.plot(slice_to_half(v_turb_rms_vol)/delU, slice_to_half(Y_lims), color='white', linewidth=1,label=r"$3D$")
    legend = ax_logx4.legend(frameon=False, loc = 'lower right', fontsize=14)
    ax_logx4.set_xticks([])
    ax_logx4.xaxis.set_visible(False)
    for text in legend.get_texts():
        text.set_color("white")
    for spine in ax[1,1].spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1)  # thickness of border
        spine.set_edgecolor('black')  # color of border

    plt.suptitle(str(int(NX*F/2**max_level)) + 'x' + str(int(NY*F/2**max_level)) + 'x' + str(int(NZ*F/2**max_level)) + ' Snapshot ' + str(i) + ', SMR = ' + str(max_level) + ', Coarsening Factor = ' + str(F))
    plt.savefig(dir +"KH_2D_Z_0_" + str(i).zfill(5) + "_C1grey.png",bbox_inches='tight')

def plot_vturb_profiles():

    global dir, time_0, P_0, delU, T_h, T_inflection, time_mid

    with np.load(dir +f'KH_1D_arrays_time_averaged{n_i}to{n_f}with{jump}.npz', 'r') as f:
        v_turb_mw_whole_box = f['v_turb_mw_whole_box']/delU
        v_turb_mw_whole_box1 = f['v_turb_mw_whole_box1']/delU
        v_turb_mw_whole_box2 = f['v_turb_mw_whole_box2']/delU
        v_turb_mw_whole_box3 = f['v_turb_mw_whole_box3']/delU
        v_turb_volw_whole_box = f['v_turb_volw_whole_box']/delU
        v_turb_volw_whole_box1 = f['v_turb_volw_whole_box1']/delU
        v_turb_volw_whole_box2 = f['v_turb_volw_whole_box2']/delU
        v_turb_volw_whole_box3 = f['v_turb_volw_whole_box3']/delU
        v_turb_wholebox = f['v_turb_wholebox']/delU
        v_turb_wholebox1 = f['v_turb_wholebox1']/delU
        v_turb_wholebox2 = f['v_turb_wholebox2']/delU
        v_turb_wholebox3 = f['v_turb_wholebox3']/delU

    plt.figure(figsize=(8,6))
    plt.plot(v_turb_mw_whole_box, color='blue', label=r"$mass-weighted$", linestyle='-')
    plt.plot(v_turb_volw_whole_box, color='black', label=r"$volume-weighted$", linestyle='-')
    plt.plot(v_turb_wholebox, color='red', label=r"$unweighted$", linestyle='-')
    plt.plot(v_turb_mw_whole_box1, color='blue', label=r"$u_{turb,x}$", linestyle='--')
    plt.plot(v_turb_mw_whole_box2, color='blue', label=r"$u_{turb,z}$", linestyle=':')
    plt.plot(v_turb_volw_whole_box1, color='black', linestyle='--')
    plt.plot(v_turb_volw_whole_box2, color='black', linestyle=':')
    plt.plot(v_turb_wholebox1, color='red', linestyle='--')
    plt.plot(v_turb_wholebox2, color='red', linestyle=':')
    plt.plot(v_turb_mw_whole_box3, color='blue', label=r"$u_{turb, y}$", linestyle='-.')
    plt.plot(v_turb_volw_whole_box3, color='black', linestyle='-.')
    plt.plot(v_turb_wholebox3, color='red', linestyle='-.')
    plt.yscale('log')
    plt.ylabel(r"$u_{turb}/\Delta u$", fontsize=16)
    plt.legend(loc='best', fontsize=12, frameon=False)
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
    plt.savefig(dir + 'turbulent_velocity_profiles.png', bbox_inches='tight')

if __name__ == "__main__":

    global time_0, n1, n2, n3, n4, jump, time_mid
    
    T_0 = 1e5
    time_mid = 3. * (T_0/TEMPERATURE)**2 / (2*P_0 * np.vectorize(ISMCoolFn)(T_0)/COOLING_UNIT)
    print(f"Cooling time at T_0 = {T_0} K is {time_mid} code units")

    temp_arr = np.logspace(4, 6, 1000)
    P_0_array = P_0 * np.ones_like(temp_arr)
    Lambda_fn = np.vectorize(ISMCoolFn)(temp_arr)/COOLING_UNIT
    cooling_arr = np.divide(3. * (temp_arr/TEMPERATURE)**2 , 2*P_0_array * Lambda_fn, out=np.full_like(temp_arr, math.inf, dtype=float), 
                    where=Lambda_fn != 0)
    time_0 = np.min(cooling_arr)

    print(f"Characteristic cooling time (time_0) = {time_0} code units")
    print(f"Ratio of cooling time at 1e5 to minimum cooling time is {time_mid/time_0}")

    plt.figure(figsize=(8,6))
    plt.plot(np.logspace(4,6,1000), np.vectorize(ISMCoolFn)(np.logspace(4,6,1000)), color='blue')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')
    plt.savefig(dir + 'cooling_function.png', bbox_inches='tight')
    print(time_0)

    n_i = 0
    n_f = 250

    n1 = 71
    n4 = 250
    n2 = n1 + (n4 - n1 +1)//3
    n3 = n2 + (n4 - n1 +1)//3

    skip = 1
    jump = skip

    plot_profiles()
    make_2D_paper_plots(169)
    plot_vturb_profiles()
    print("1D arrays created successfully.")

