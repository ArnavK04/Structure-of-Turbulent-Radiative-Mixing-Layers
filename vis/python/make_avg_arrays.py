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
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Global constants
MPI_DEF = False
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
# cell centre for ith cell is at i*DY + DY/2 + YMIN
NY_init = int(max(NY*(cold_frac - 0.35), 0))
NY_fin = int(min(NY*(cold_frac + 0.35), NY))

global dir
#dir = r"../../my_outputs/noSMR_2_3_cutoffISMcoolfn/fid3D_32_cool/bin/"
#dir = r"../../my_outputs/fiducial1040_cool2D/bin/"
#dir = r"../../my_outputs/fid3D_2xlessvel_1040_cool/bin/"
#dir = r"../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/invertedfiducial4000less256_1024/bin/"
dir = r"../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/fid3D_1040_cool/bin/"
#dir = r"../../my_outputs/fid3D_halfbox_1040_cool/bin/"
#dir = r"../../../Downloads/Chandra_data/snapsfiducial16cool/"
#dir = r"../../../Downloads/Niagara3Dfidnpz/"
#dir = r"../../my_outputs/noSMR_2_3_cutoffISMcoolfn/fid3D_5xmoredens_1040_cool/bin/"

from save_2D_arrays_3D import ISMCoolFn

def slice_to_half(arr):
    global NY_init, NY_fin
    return arr[NY_init:NY_fin]

def slice_to_half_2D(arr):
    global NY_init, NY_fin
    return arr[NY_init:NY_fin,:]

def find_temporal_z0(start, end, jump):
    # load temperature 1-D profiles from each npz file and fit them to a tanh profile
    # calculate average characteristic length of fit and its temporal variation.
    global x0_norm
    avg_z0 = 0.0
    sum_z0_sqr = 0.0
    F=1

    T_tanh = lambda z, z0, Th, Tc: (0.5 * (Th - Tc) * np.tanh(z / z0) + 0.5 * (Th + Tc))

    for n in range(start, end + 1, jump):
        with np.load(dir + 'KH_1D_arrays_snapshot_' + str(n).zfill(5) + f'C{F}.npz', 'r') as f:
            Y = f['Y_lims']/(delU*time_0)
            temp_vol = f['temp_vol']

        Tc0 = np.min(temp_vol)
        Th0 = np.max(temp_vol)
        dTdz = np.gradient(temp_vol, Y)
        z0_guess = Y[np.argmax(np.abs(dTdz))]
        slope_max = np.max(np.abs(dTdz))
        x0_guess = (Th0 - Tc0) / (2 * slope_max)
        p0 = [x0_guess, z0_guess, Th0, Tc0]

        # fitting a tanh function to temp71to250
        from scipy.optimize import curve_fit
        def T_tanh_model(z, x0, z0, Th, Tc):
            return ((0.5 * (Th + Tc) + 0.5 * (Th - Tc) * np.tanh((z - x0) / z0)))
        popt, pcov = curve_fit(T_tanh_model, Y, temp_vol, p0=p0)
        print(f'Fitted parameters: x0={popt[0]}, z0={popt[1]}, Th={popt[2]}, Tc={popt[3]}, for snapshot {n}')
        z0 = popt[1]
        x0 = popt[0]
        Y -= x0
        if (n==start):
            x0_norm = x0
        fit_curve = T_tanh(Y, z0, T_h, T_c)/T_h

        np.savez_compressed(dir + f"KH_tanhfit_snapshot_{str(n).zfill(5)}_C{F}.npz",
                    z0 = z0,
                    x0 = x0,
                    Th = popt[2],
                    Tc = popt[3],
                    Y = Y,
                    temp_vol = temp_vol,
                    fit_curve = fit_curve)
        
        avg_z0 += z0
        sum_z0_sqr += z0**2

    avg_z0 /= (end - start + 1)
    sum_z0_sqr /= (end - start + 1)
    z0_sig = np.sqrt(sum_z0_sqr - avg_z0**2)
    print(f'AVERAGE Z_0 FROM SNAPSHOT {start} to {end} is {avg_z0} +/- {z0_sig}')

def make_1D(start, end, jump):
    F=1
    rho_av = np.zeros(NY)
    rho_vx1_av = np.zeros(NY)
    rho_vx2_av = np.zeros(NY)
    energy_dens_av = np.zeros(NY)
    rho2_av = np.zeros(NY)
    rho_vx12_av = np.zeros(NY)
    rho_vx22_av = np.zeros(NY)
    energy_dens_sqred_av = np.zeros(NY)
    emis_vol_av = np.zeros(NY)
    emis_vol2_av = np.zeros(NY)
    vx1_mw_av = np.zeros(NY)
    vx1_vol_av = np.zeros(NY)
    vx12_vol_av = np.zeros(NY)
    vx2_mw_av = np.zeros(NY)
    vx2_vol_av = np.zeros(NY)
    vx22_vol_av = np.zeros(NY)
    vx12_mw_av = np.zeros(NY)
    vx22_mw_av = np.zeros(NY)
    temp_mw_av = np.zeros(NY)
    temp_vol_av = np.zeros(NY)
    temp2_mw_av = np.zeros(NY)
    temp2_vol_av = np.zeros(NY)
    Be_mw_av = np.zeros(NY)
    Be2_mw_av = np.zeros(NY)
    Be_vol_av = np.zeros(NY)
    Be2_vol_av = np.zeros(NY)
    p_vol_av = np.zeros(NY)
    p2_vol_av = np.zeros(NY)
    total_vel_av = np.zeros(NY)
    rho_mw_av = np.zeros(NY)
    P_E__T_av = np.zeros(100)
    P_E__T_sig = np.zeros(100)
    P_E__T_log10T_av = np.zeros(100)
    P_E__T_log10T_sig = np.zeros(100)
    Sigma_dot_cool_hist = np.zeros(end - start + 1)
    energy_density_integrated = np.zeros(end - start + 1)

    if (start!=0):
        with np.load(dir + f"KH_1D_arrays_snapshot{n_i}_{n_f}_{str(start).zfill(5)}_C{F}_y_lims_corrected.npz", 'r') as f:
            v_TRML_integrated = f['v_TRML_integrated']
    else:
        v_TRML_integrated=0.0

    # finding dt
    with np.load(f"{dir}KH_fluxes_{str(start).zfill(5)}_C{F}.npz", 'r') as f:
        timei = f['time']
    with np.load(f"{dir}KH_fluxes_{str(start+jump).zfill(5)}_C{F}.npz", 'r') as f:
        time1 = f['time']
    dt = time1 - timei

    count = 0

    for n in range(start,end+1, jump):
        print(f"Processing snapshot {n}...")
        with np.load(f"{dir}KH_fluxes_{str(n).zfill(5)}_C{F}.npz", 'r') as f:
            v_TRML = f['v_TRML'][1]

        with np.load(dir + 'KH_1D_arrays_snapshot_' + str(n).zfill(5) + f'C{F}.npz', 'r') as f:
            Y_lims = f['Y_lims']
            Y_range_final = f['Y_lims']     # all files save same Y_lims, so just using the one from the last file
            rho_vol = f['den_vol']
            rho_sig = f['den_sig_vol']
            rho_vx1_vol = f['rho_vx1_vol']
            rho_vx2_vol = f['rho_vx2_vol']
            energy_dens_vol = f['total_energy_dens']
            emis_vol = f['emis_vol']
            vx1_vol = f['vx1_vol']
            vx2_vol = f['vx2_vol']
            vx1_mw = f['vx1_mw']
            vx2_mw = f['vx2_mw']
            temp_mw = f['temp_mw']
            temp_vol = f['temp_vol']
            Be_mw = f['Be_mw']
            Be_vol = f['Be_vol']
            p_vol = f['prs_vol']
            vx3_vol = f['vx3_vol']
            rho_mw = (rho_sig**2 + rho_vol**2)/ rho_vol

        energy_density_integrated[count] = (np.cumsum(energy_dens_vol)*(Y_lims[1] - Y_lims[0]))[-1]

        count+=1

        Y_lims -= v_TRML_integrated         # shifting Y_lims to account for the movement of the interface due to the TRML velocity.
        v_TRML_integrated += v_TRML * dt

        # interpolating all profiles to the same Y_lims (the one from the last file) before averaging, since the interface moves due to the TRML velocity and thus the transformed Y_lims of each file is different.
        from scipy.interpolate import CubicSpline, interp1d

        def interp(arr):
            return interp1d(Y_lims, arr, kind="linear", fill_value="extrapolate")(Y_range_final)
        
        rho_vol = interp(rho_vol)
        rho_vx1_vol = interp(rho_vx1_vol)
        rho_vx2_vol = interp(rho_vx2_vol)
        energy_dens_vol = interp(energy_dens_vol)
        emis_vol = interp(emis_vol)
        vx1_vol = interp(vx1_vol)
        vx2_vol = interp(vx2_vol)
        vx1_mw = interp(vx1_mw)
        vx2_mw = interp(vx2_mw)
        temp_mw = interp(temp_mw)
        temp_vol = interp(temp_vol)
        Be_mw = interp(Be_mw)
        Be_vol = interp(Be_vol)
        p_vol = interp(p_vol)
        vx3_vol = interp(vx3_vol)
        rho_mw = interp(rho_mw)

        integrated_cooling_rate = np.cumsum(emis_vol)*(Y_lims[1] - Y_lims[0])
        Sigma_dot_cool_hist[count-1] = integrated_cooling_rate[-1]
        Tmin = 1.1e4
        Tmax = 0.9e6
        T_prime = np.gradient(temp_vol, Y_lims)
        n2_lambdaT = emis_vol
        Sigma_cool = np.sum(emis_vol) * (Y_lims[1] - Y_lims[0])
        P_E__T = n2_lambdaT/(T_prime * Sigma_cool)
        temp_range = np.logspace(4, 6, 100)
        P_E__T = np.interp(temp_range, temp_vol, P_E__T)
        P_E__T = np.where(temp_range < Tmin, 0.0, P_E__T)
        P_E__T = np.where(temp_range > Tmax, 0.0, P_E__T)
        P_E__T /= np.trapezoid(P_E__T, temp_range)
        P_E__T_log10T = P_E__T * temp_range / np.log10(np.exp(1.))

        P_E__T_av += P_E__T
        P_E__T_sig += P_E__T * P_E__T
        P_E__T_log10T_av += P_E__T_log10T
        P_E__T_log10T_sig += P_E__T_log10T * P_E__T_log10T
        rho_mw_av += rho_mw
        total_vel_av += np.sqrt(vx1_vol**2 + vx2_vol**2 + vx3_vol**2)
        emis_vol_av += emis_vol
        emis_vol2_av += emis_vol * emis_vol
        rho_av += rho_vol
        rho_vx1_av += rho_vx1_vol
        rho_vx2_av += rho_vx2_vol
        energy_dens_av += energy_dens_vol
        rho2_av += rho_vol * rho_vol
        rho_vx12_av += rho_vx1_vol * rho_vx1_vol
        rho_vx22_av += rho_vx2_vol * rho_vx2_vol
        energy_dens_sqred_av += energy_dens_vol * energy_dens_vol
        vx1_vol_av += vx1_vol
        vx2_vol_av += vx2_vol
        vx12_vol_av += vx1_vol * vx1_vol
        vx22_vol_av += vx2_vol * vx2_vol
        vx1_mw_av += vx1_mw
        vx2_mw_av += vx2_mw
        vx12_mw_av += vx1_mw * vx1_mw
        vx22_mw_av += vx2_mw * vx2_mw
        temp_mw_av += temp_mw
        temp_vol_av += temp_vol
        temp2_mw_av += temp_mw * temp_mw
        temp2_vol_av += temp_vol * temp_vol
        Be_mw_av += Be_mw
        Be2_mw_av += Be_mw * Be_mw
        Be_vol_av += Be_vol
        Be2_vol_av += Be_vol * Be_vol
        p_vol_av += p_vol
        p2_vol_av += p_vol * p_vol

        np.savez_compressed(dir + f"KH_1D_arrays_snapshot{start}_{end}_{str(n).zfill(5)}_C{F}_y_lims_corrected.npz", Y_lims = Y_lims, v_TRML_integrated = v_TRML_integrated - v_TRML*dt )


    P_E__T_av /= count
    P_E__T_sig /= count
    P_E__T_log10T_av /= count
    P_E__T_log10T_sig /= count
    rho_mw_av /= count
    total_vel_av /= count
    rho_av /= count
    rho_vx1_av /= count
    rho_vx2_av /= count
    energy_dens_av /= count
    rho2_av /= count
    rho_vx12_av /= count
    rho_vx22_av /= count
    energy_dens_sqred_av /= count
    emis_vol_av /= count
    emis_vol2_av /= count
    vx1_vol_av /= count
    vx2_vol_av /= count
    vx12_vol_av /= count
    vx22_vol_av /= count
    vx1_mw_av /= count
    vx2_mw_av /= count
    vx12_mw_av /= count
    vx22_mw_av /= count
    temp_mw_av /= count
    temp_vol_av /= count
    temp2_mw_av /= count
    temp2_vol_av /= count
    Be_mw_av /= count
    Be2_mw_av /= count
    Be_vol_av /= count
    Be2_vol_av /= count
    p_vol_av /= count
    p2_vol_av /= count

    P_E__T_sig = np.sqrt(P_E__T_sig - P_E__T_av * P_E__T_av)
    P_E__T_log10T_sig = np.sqrt(P_E__T_log10T_sig - P_E__T_log10T_av * P_E__T_log10T_av)
    rho_sig = np.sqrt(rho2_av - rho_av * rho_av)
    rho_vx1_sig = np.sqrt(rho_vx12_av - rho_vx1_av * rho_vx1_av)
    rho_vx2_sig = np.sqrt(rho_vx22_av - rho_vx2_av * rho_vx2_av)
    energy_dens_sig = np.sqrt(energy_dens_sqred_av - energy_dens_av * energy_dens_av)
    emis_vol_sig = np.sqrt(emis_vol2_av - emis_vol_av * emis_vol_av)
    vx1_sig = np.sqrt(vx12_vol_av - vx1_vol_av * vx1_vol_av)
    vx2_sig = np.sqrt(vx22_vol_av - vx2_vol_av * vx2_vol_av)
    vx1_mw_sig = np.sqrt(vx12_mw_av - vx1_mw_av * vx1_mw_av)
    vx2_mw_sig = np.sqrt(vx22_mw_av - vx2_mw_av * vx2_mw_av)
    temp_mw_sig = np.sqrt(temp2_mw_av - temp_mw_av * temp_mw_av)
    temp_vol_sig = np.sqrt(temp2_vol_av - temp_vol_av * temp_vol_av)
    Be_mw_sig = np.sqrt(Be2_mw_av - Be_mw_av * Be_mw_av)
    Be_sig = np.sqrt(Be2_vol_av - Be_vol_av * Be_vol_av)
    p2_vol_av = np.sqrt(p2_vol_av - p_vol_av * p_vol_av)

    np.savez_compressed(dir + f"KH_1D_arrays_time_averaged{start}to{end}with{jump}.npz",
                P_E__T_av = P_E__T_av, P_E__T_sig = P_E__T_sig,
                P_E__T_log10T_av = P_E__T_log10T_av, P_E__T_log10T_sig = P_E__T_log10T_sig,
                total_vel_av = total_vel_av, temp_range = temp_range,
                rho_av = rho_av, rho_mw_av = rho_mw_av,
                rho_vx1_av = rho_vx1_av,
                rho_vx2_av = rho_vx2_av,
                energy_dens_av = energy_dens_av,
                rho_sig = rho_sig,
                rho_vx1_sig = rho_vx1_sig,
                rho_vx2_sig = rho_vx2_sig,
                energy_dens_sig = energy_dens_sig,
                emis_vol_av = emis_vol_av,
                emis_vol_sig = emis_vol_sig,
                vx1_vol_av = vx1_vol_av, vx1_sig = vx1_sig,
                vx2_vol_av = vx2_vol_av, vx2_sig = vx2_sig,
                vx1_mw_av = vx1_mw_av, vx1_mw_sig = vx1_mw_sig,
                vx2_mw_av = vx2_mw_av, vx2_mw_sig = vx2_mw_sig,
                temp_mw_av = temp_mw_av, temp_mw_sig = temp_mw_sig,
                temp_vol_av = temp_vol_av, temp_vol_sig = temp_vol_sig,
                vx1_vol_sig = vx1_sig, vx2_vol_sig = vx2_sig,
                Be_mw_av = Be_mw_av, Be_mw_sig = Be_mw_sig,
                Be_vol_av = Be_vol_av, Be_sig = Be_sig,
                p_vol_av = p_vol_av, p_sig = p2_vol_av,Sigma_dot_cool_hist = Sigma_dot_cool_hist, Y_range_final = Y_range_final, energy_density_integrated = energy_density_integrated)

    del rho_av, rho_vx1_av, rho_vx2_av, energy_dens_av, emis_vol_av
    del rho_sig, rho_vx1_sig, rho_vx2_sig, energy_dens_sig, emis_vol_sig
    gc.collect()

    rho_avXv2_av_ = np.zeros(NY)
    rho_avXv2_sig = np.zeros(NY)
    rhov2_av_ = np.zeros(NY)
    rhov2_sig = np.zeros(NY)
    delrho_delv2_av_ = np.zeros(NY)
    delrho_delv2_sig = np.zeros(NY)

    rhov2_avXv1_av_ = np.zeros(NY)
    rhov2_avXv1_sig = np.zeros(NY)
    rhov2v1_av_ = np.zeros(NY)
    rhov2v1_sig = np.zeros(NY)
    R_xz_ = np.zeros(NY)
    R_xz_sig = np.zeros(NY)

    rhov2_avXv2_av_ = np.zeros(NY)
    rhov2_avXv2_sig = np.zeros(NY)
    R_zz_ = np.zeros(NY)
    R_zz_sig = np.zeros(NY)
    p_av_ = np.zeros(NY)
    p_sig = np.zeros(NY)
    P_rhovx2sqr_av_ = np.zeros(NY)
    P_rhovx2sqr_sig = np.zeros(NY)

    Be_av_rhov2_av_ = np.zeros(NY)
    Be_av_rhov2_sig = np.zeros(NY)
    del_Be_del_rhov2_av_ = np.zeros(NY)
    del_Be_del_rhov2_sig = np.zeros(NY)
    edot_cool_cum_dx2_av_ = np.zeros(NY)
    edot_cool_cum_dx2_sig = np.zeros(NY)
    net_heating_av_ = np.zeros(NY)
    net_heating_sig = np.zeros(NY)
    Sigma_dot_cool_hist = np.zeros(end - start + 1)

    with np.load(dir + f"KH_1D_arrays_snapshot{n_i}_{n_f}_{str(start).zfill(5)}_C{F}_y_lims_corrected.npz", 'r') as f:
        v_TRML_integrated = f['v_TRML_integrated']

    # finding dt
    with np.load(f"{dir}KH_fluxes_{str(start).zfill(5)}_C{F}.npz", 'r') as f:
        timei = f['time']
    with np.load(f"{dir}KH_fluxes_{str(start+jump).zfill(5)}_C{F}.npz", 'r') as f:
        time1 = f['time']
    dt = time1 - timei

    count=0

    for n in range(start, end + 1, jump):
        print(f"Processing snapshot {n} for fluxes...")

        with np.load(dir + 'KH_1D_arrays_snapshot_' + str(n).zfill(5) + f'C{F}.npz', 'r') as f:
            Y_lims = f['Y_lims']
            Y_range_final = f['Y_lims']     # all files save same Y_lims, so just using the one from the last file

        with np.load(f"{dir}KH_fluxes_{str(n).zfill(5)}_C{F}.npz", 'r') as f:
            v_TRML = f['v_TRML'][1]
            rho_avXv2_av = f['rho_avXv2_av']
            rhov2_av = f['rhov2_av']
            delrho_delv2_av = f['delrho_delv2_av']

            rhov2_avXv1_av = f['rhov2_avXv1_av']
            rhov2v1_av = f['rhov2v1_av']
            R_xz = f['R_xz']

            rhov2_avXv2_av = f['rhov2_avXv2_av']
            R_zz = f['R_zz']
            p_av = f['p_av']
            P_rhovx2sqr_av = f['P_rhovx2sqr']

            Be_av_rhov2_av = f['Be_av_rhov2_av']
            del_Be_del_rhov2_av = f['del_Be_del_rhov2_av']
            edot_cool_cum_dx2_av = f['edot_cool_cum_dx2']
            net_heating_av = f['net_heating']

        Y_lims -= v_TRML_integrated         # shifting Y_lims to account for the movement of the interface due to the TRML velocity.
        v_TRML_integrated += v_TRML * dt

        count += 1

        Sigma_dot_cool_hist[count-1] = edot_cool_cum_dx2_av[-1]

        from scipy.interpolate import interp1d

        def interp(arr):
            return interp1d(Y_lims, arr, kind="linear", fill_value="extrapolate")(Y_range_final)

        # interpolating all profiles to the same Y_lims (the one from the last file) before averaging, since the interface moves due to the TRML velocity and thus the transformed Y_lims of each file is different.
        rho_avXv2_av = interp(rho_avXv2_av)
        rhov2_av = interp(rhov2_av)
        delrho_delv2_av = interp(delrho_delv2_av)
        rhov2_avXv1_av = interp(rhov2_avXv1_av)
        rhov2v1_av = interp(rhov2v1_av)
        R_xz = interp(R_xz)
        rhov2_avXv2_av = interp(rhov2_avXv2_av)
        R_zz = interp(R_zz)
        p_av = interp(p_av)
        P_rhovx2sqr_av = interp(P_rhovx2sqr_av)
        Be_av_rhov2_av = interp(Be_av_rhov2_av)
        del_Be_del_rhov2_av = interp(del_Be_del_rhov2_av)
        edot_cool_cum_dx2_av = interp(edot_cool_cum_dx2_av)
        net_heating_av = interp(net_heating_av)

        rho_avXv2_av_ += rho_avXv2_av
        rho_avXv2_sig += rho_avXv2_av * rho_avXv2_av
        rhov2_av_ += rhov2_av
        rhov2_sig += rhov2_av * rhov2_av
        delrho_delv2_av_ += delrho_delv2_av
        delrho_delv2_sig += delrho_delv2_av * delrho_delv2_av
        rhov2_avXv1_av_ += rhov2_avXv1_av
        rhov2_avXv1_sig += rhov2_avXv1_av * rhov2_avXv1_av
        rhov2v1_av_ += rhov2v1_av
        rhov2v1_sig += rhov2v1_av * rhov2v1_av
        R_xz_ += R_xz
        R_xz_sig += R_xz * R_xz
        rhov2_avXv2_av_ += rhov2_avXv2_av
        rhov2_avXv2_sig += rhov2_avXv2_av * rhov2_avXv2_av
        R_zz_ += R_zz
        R_zz_sig += R_zz * R_zz
        p_av_ += p_av
        p_sig += p_av * p_av
        P_rhovx2sqr_av_ += P_rhovx2sqr_av
        P_rhovx2sqr_sig += P_rhovx2sqr_av * P_rhovx2sqr_av
        Be_av_rhov2_av_ += Be_av_rhov2_av
        Be_av_rhov2_sig += Be_av_rhov2_av * Be_av_rhov2_av
        del_Be_del_rhov2_av_ += del_Be_del_rhov2_av
        del_Be_del_rhov2_sig += del_Be_del_rhov2_av * del_Be_del_rhov2_av
        edot_cool_cum_dx2_av_ += edot_cool_cum_dx2_av
        edot_cool_cum_dx2_sig += edot_cool_cum_dx2_av * edot_cool_cum_dx2_av
        net_heating_av_ += net_heating_av
        net_heating_sig += net_heating_av * net_heating_av

    rho_avXv2_av_ /= count
    rho_avXv2_sig /= count
    rhov2_av_ /= count
    rhov2_sig /= count
    delrho_delv2_av_ /= count
    delrho_delv2_sig /= count
    rhov2_avXv1_av_ /= count
    rhov2_avXv1_sig /= count
    rhov2v1_av_ /= count
    rhov2v1_sig /= count
    R_xz_ /= count
    R_xz_sig /= count
    rhov2_avXv2_av_ /= count
    rhov2_avXv2_sig /= count
    R_zz_ /= count
    R_zz_sig /= count
    p_av_ /= count
    p_sig /= count
    P_rhovx2sqr_av_ /= count
    P_rhovx2sqr_sig /= count
    Be_av_rhov2_av_ /= count
    Be_av_rhov2_sig /= count
    del_Be_del_rhov2_av_ /= count
    del_Be_del_rhov2_sig /= count
    edot_cool_cum_dx2_av_ /= count
    edot_cool_cum_dx2_sig /= count
    net_heating_av_ /= count
    net_heating_sig /= count

    rho_avXv2_sig = np.sqrt(rho_avXv2_sig - rho_avXv2_av_ * rho_avXv2_av_)
    rhov2_sig = np.sqrt(rhov2_sig - rhov2_av_ * rhov2_av_)
    delrho_delv2_sig = np.sqrt(delrho_delv2_sig - delrho_delv2_av_ * delrho_delv2_av_)
    rhov2_avXv1_sig = np.sqrt(rhov2_avXv1_sig - rhov2_avXv1_av_ * rhov2_avXv1_av_)
    rhov2v1_sig = np.sqrt(rhov2v1_sig - rhov2v1_av_ * rhov2v1_av_)
    R_xz_sig = np.sqrt(R_xz_sig - R_xz_ * R_xz_)
    rhov2_avXv2_sig = np.sqrt(rhov2_avXv2_sig - rhov2_avXv2_av_ * rhov2_avXv2_av_)
    R_zz_sig = np.sqrt(R_zz_sig - R_zz_ * R_zz_)
    p_sig = np.sqrt(p_sig - p_av_ * p_av_)
    P_rhovx2sqr_sig = np.sqrt(P_rhovx2sqr_sig - P_rhovx2sqr_av_ * P_rhovx2sqr_av_)
    Be_av_rhov2_sig = np.sqrt(Be_av_rhov2_sig - Be_av_rhov2_av_ * Be_av_rhov2_av_)
    del_Be_del_rhov2_sig = np.sqrt(del_Be_del_rhov2_sig - del_Be_del_rhov2_av_ * del_Be_del_rhov2_av_)
    edot_cool_cum_dx2_sig = np.sqrt(edot_cool_cum_dx2_sig - edot_cool_cum_dx2_av_ * edot_cool_cum_dx2_av_)
    net_heating_sig = np.sqrt(net_heating_sig - net_heating_av_ * net_heating_av_)

    np.savez_compressed(dir + f"KH_fluxes_time_averaged{start}to{end}with{jump}.npz",
                rho_avXv2_av = rho_avXv2_av_,
                rho_avXv2_sig = rho_avXv2_sig,
                rhov2_av = rhov2_av_,
                rhov2_sig = rhov2_sig,
                delrho_delv2_av = delrho_delv2_av_,
                delrho_delv2_sig = delrho_delv2_sig,
                rhov2_avXv1_av = rhov2_avXv1_av_,
                rhov2_avXv1_sig = rhov2_avXv1_sig,
                rhov2v1_av = rhov2v1_av_,
                rhov2v1_sig = rhov2v1_sig,
                R_xz = R_xz_,
                R_xz_sig = R_xz_sig,
                rhov2_avXv2_av = rhov2_avXv2_av_,
                rhov2_avXv2_sig = rhov2_avXv2_sig,
                R_zz = R_zz_,
                R_zz_sig = R_zz_sig,
                p_av = p_av_,
                p_sig = p_sig,
                P_rhovx2sqr = P_rhovx2sqr_av_,
                P_rhovx2sqr_sig = P_rhovx2sqr_sig,
                Be_av_rhov2_av = Be_av_rhov2_av_,
                Be_av_rhov2_sig = Be_av_rhov2_sig,
                del_Be_del_rhov2_av = del_Be_del_rhov2_av_,
                del_Be_del_rhov2_sig = del_Be_del_rhov2_sig,
                edot_cool_cum_dx2 = edot_cool_cum_dx2_av_,
                edot_cool_cum_dx2_sig = edot_cool_cum_dx2_sig,
                net_heating = net_heating_av_,
                net_heating_sig = net_heating_sig, Y_range_final = Y_range_final, Sigma_dot_cool_hist = Sigma_dot_cool_hist)

def make_PDF(start, end, jump):
    axis = 'none'
    slice = 1
    F=1
    hist_vol_av = np.zeros(100)
    hist_mass_av = np.zeros(100)
    hist_vol_sig = np.zeros(100)
    hist_mass_sig = np.zeros(100)
    hist_emis_av = np.zeros(100)
    hist_emis_sig = np.zeros(100)

    count = 0

    for n in range(start,end+1, jump):
        print(f"Processing snapshot {n}...")
        with np.load(dir + 'KH_tempPDF_snapshot_' + str(n).zfill(5) + f'C{F}_axis{axis}_slice{slice}.npz', 'r') as f:
            hist_vol = f['hist_vol']
            hist_mass = f['hist_mass']
            hist_emis = f['hist_emissivity']
            bin_centers = f['bin_centers']

        count += 1

        T = 10**bin_centers
        hist_vol = np.where(T < 1.1e4, 0.0, hist_vol)
        hist_vol = np.where(T > 0.9e6, 0.0, hist_vol)
        hist_vol /= np.trapezoid(hist_vol, bin_centers)
        hist_mass = np.where(T < 1.1e4, 0.0, hist_mass)
        hist_mass = np.where(T > 0.9e6, 0.0, hist_mass)
        hist_mass /= np.trapezoid(hist_mass, bin_centers)
        hist_emis = np.where(T < 1.1e4, 0.0, hist_emis)
        hist_emis = np.where(T > 0.9e6, 0.0, hist_emis)
        hist_emis /= np.trapezoid(hist_emis, bin_centers)

        hist_vol_av += hist_vol
        hist_mass_av += hist_mass
        hist_emis_av += hist_emis
        hist_vol_sig += hist_vol * hist_vol
        hist_mass_sig += hist_mass * hist_mass
        hist_emis_sig += hist_emis * hist_emis
    

    hist_vol_av /= count
    hist_mass_av /= count
    hist_emis_av /= count
    hist_vol_sig /= count
    hist_mass_sig /= count
    hist_emis_sig /= count

    hist_vol_sig = np.sqrt(hist_vol_sig - hist_vol_av * hist_vol_av)
    hist_mass_sig = np.sqrt(hist_mass_sig - hist_mass_av * hist_mass_av)
    hist_emis_sig = np.sqrt(hist_emis_sig - hist_emis_av * hist_emis_av)


    np.savez_compressed(dir + f"KH_PDFs_time_averaged{start}to{end}with{jump}.npz",
                        hist_vol_av=hist_vol_av,
                        hist_mass_av=hist_mass_av,
                        hist_emis_av=hist_emis_av,
                        hist_vol_sig=hist_vol_sig,
                        hist_mass_sig=hist_mass_sig,
                        hist_emis_sig=hist_emis_sig,
                        bin_centers=bin_centers)

    del hist_vol_av, hist_mass_av, hist_emis_av
    del hist_vol_sig, hist_mass_sig, hist_emis_sig
    gc.collect()

def make_spacetime_plots(end, trmlframeflag = False):

    global delU, P_0, T_h, rho_h, time_0, cold_frac, NY_init, NY_fin
    start = 0
    jump = 1
    y_size = NY_fin - NY_init

    full_velx = np.zeros((end - start + 1, y_size))
    full_vely = np.zeros((end - start + 1, y_size))
    full_velz = np.zeros((end - start + 1, y_size))
    full_temp = np.zeros((end - start + 1, y_size))
    full_den = np.zeros((end - start + 1, y_size))
    full_pres = np.zeros((end - start + 1, y_size))
    full_scalar = np.zeros((end - start + 1, y_size))
    full_pseudo_entropy = np.zeros((end - start + 1, y_size))

    F=1

    # finding dt
    with np.load(f"{dir}KH_fluxes_{str(start).zfill(5)}_C{F}.npz", 'r') as f:
        timei = f['time']
    with np.load(f"{dir}KH_fluxes_{str(start+jump).zfill(5)}_C{F}.npz", 'r') as f:
        time1 = f['time']
    with np.load(f"{dir}KH_fluxes_{str(end).zfill(5)}_C{F}.npz", 'r') as f:
        timef = f['time']
    dt = time1 - timei

    if trmlframeflag:

        v_TRML_integrated = 0.0
        frame = "trml_frame"

        with np.load(dir + 'KH_1D_arrays_snapshot_' + str(start).zfill(5) + f'C{F}.npz', 'r') as f:
            Y_range_final = slice_to_half(f['Y_lims'])

        for n in range(start,end+1, jump):

            print(f"Processing snapshot {n} for {frame} spacetime diagram...")

            with np.load(f"{dir}KH_fluxes_{str(n).zfill(5)}_C{F}.npz", 'r') as f:
                v_TRML = f['v_TRML'][1]

            with np.load(dir + 'KH_1D_arrays_snapshot_' + str(n).zfill(5) + f'C{F}.npz', 'r') as f:
                Y_lims = f['Y_lims']
                rho_vol = f['den_vol']
                vx1_vol = f['vx1_vol']
                vx2_vol = f['vx2_vol']
                temp_vol = f['temp_vol']
                p_vol = f['prs_vol']
                vx3_vol = f['vx3_vol']
                ps_vol = f['ps_vol']
                pseudo_entropy_vol = np.log10((p_vol/P_0)/((rho_vol/rho_h)**GAMMA))

            Y_lims -= v_TRML_integrated         # shifting Y_lims to account for the movement of the interface due to the TRML velocity.
            v_TRML_integrated += v_TRML * dt

            # interpolating all profiles to the same Y_lims (the one from the last file) before averaging, since the interface moves due to the TRML velocity and thus the transformed Y_lims of each file is different.
            from scipy.interpolate import interp1d

            def interp(arr):
                return interp1d(Y_lims, arr, kind="linear", fill_value="extrapolate")(Y_range_final)
            
            rho_vol = interp(rho_vol)
            vx1_vol = interp(vx1_vol)
            vx2_vol = interp(vx2_vol)
            temp_vol = interp(temp_vol)
            p_vol = interp(p_vol)
            vx3_vol = interp(vx3_vol)
            ps_vol = interp(ps_vol)
            pseudo_entropy_vol = interp(pseudo_entropy_vol)

            full_velx[n, :] = vx1_vol
            full_vely[n, :] = vx2_vol
            full_velz[n, :] = vx3_vol
            full_temp[n, :] = temp_vol
            full_den[n, :] = rho_vol
            full_pres[n, :] = p_vol
            full_scalar[n, :] = ps_vol
            full_pseudo_entropy[n, :] = pseudo_entropy_vol

        Y_range = Y_range_final

    else : 
        frame = "sim_frame"

        for n in range(start,end+1, jump):

            print(f"Processing snapshot {n} for {frame} spacetime diagram...")

            with np.load(f"{dir}KH_fluxes_{str(n).zfill(5)}_C{F}.npz", 'r') as f:
                v_TRML_1 = f['v_TRML'][0]
                v_TRML_2 = f['v_TRML'][1]
                v_TRML_3 = f['v_TRML'][2]

            with np.load(dir + 'KH_1D_arrays_snapshot_' + str(n).zfill(5) + f'C{F}.npz', 'r') as f:

                Y_lims = slice_to_half(f['Y_lims'])
                rho_vol = f['den_vol']
                vx1_vol = f['vx1_vol'] + v_TRML_1
                vx2_vol = f['vx2_vol'] + v_TRML_2
                temp_vol = f['temp_vol']
                p_vol = f['prs_vol']
                vx3_vol = f['vx3_vol'] + v_TRML_3
                ps_vol = f['ps_vol']
                pseudo_entropy_vol = np.log10((p_vol/P_0)/((rho_vol/rho_h)**GAMMA))

            full_velx[n, :] = slice_to_half(vx1_vol)
            full_vely[n, :] = slice_to_half(vx2_vol)
            full_velz[n, :] = slice_to_half(vx3_vol)
            full_temp[n, :] = slice_to_half(temp_vol)
            full_den[n, :] = slice_to_half(rho_vol)
            full_pres[n, :] = slice_to_half(p_vol)
            full_scalar[n, :] = slice_to_half(ps_vol)
            full_pseudo_entropy[n, :] = slice_to_half(pseudo_entropy_vol)

        Y_range = Y_lims

    # Configuration for data and labels
    datasets = [full_velx/delU, full_vely/delU, full_velz/delU, full_temp/T_h, full_den/rho_h, full_pres/P_0, full_scalar, full_pseudo_entropy]
    labels = [r"$u_x/\Delta u$", r"$u_z/\Delta u$",r"$u_y/\Delta u$",r"$T/T_h$", r"$\rho/\rho_h$",r"$p/p_0$", r"$s$" , r"$\log_{10}(\frac{p/p_0}{(\rho/\rho_h)^{\gamma}})$"]
    cmaps = ['inferno', 'inferno', 'inferno', 'inferno', 'inferno', 'inferno', 'inferno', 'inferno']
    Y_range = Y_range/(delU * time_0)
    Y_range -= x0_norm
    timei /= time_0
    time1 /= time_0
    timef /= time_0

    fig, ax = plt.subplots(2, 4, figsize=(14, 10), constrained_layout=True)
    ax = ax.flatten()

    ep = 5e-3

    # equally spaced contours for velx, velz
    epsilon_velx = ep*(np.max(datasets[0]) - np.min(datasets[0]))
    epsilon_velz = ep*(np.max(datasets[1]) - np.min(datasets[1]))
    epsilon_temp = ep*(np.max(datasets[3]) - np.min(datasets[3]))
    epsilon_dens = ep*(np.max(datasets[4]) - np.min(datasets[4]))
    epsilon_ps = ep*(np.max(datasets[6]) - np.min(datasets[6]))
    epsilon_pseudo_entropy = ep*(np.max(datasets[7]) - np.min(datasets[7]))

    velx_contours = np.linspace(np.min(datasets[0]) + epsilon_velx, np.max(datasets[0]) - epsilon_velx, 11)[1:-1]
    velz_contours = np.linspace(np.min(datasets[1]) + epsilon_velz, np.max(datasets[1]) - epsilon_velz, 8)[2:-2]
    temp_contours = np.logspace(np.log10(np.min(datasets[3]) + epsilon_temp), np.log10(np.max(datasets[3]) - epsilon_temp), 7)[1:-1]
    dens_contours = np.linspace(np.min(datasets[4]) + epsilon_dens, np.max(datasets[4]) - epsilon_dens, 9)[1:-1]
    ps_contours = np.linspace(np.min(datasets[6]) + epsilon_ps, np.max(datasets[6]) - epsilon_ps, 9)[1:-1]
    pseudo_entropy_contours = np.linspace(np.min(datasets[7]) + epsilon_pseudo_entropy, np.max(datasets[7]) - epsilon_pseudo_entropy, 9)[1:-1]

    for i in range(8):

        if i in [0, 1, 4, 6, 7]:

            field_data = datasets[i].T

            nx = field_data.shape[1]
            ny = field_data.shape[0]

            x_vals = np.linspace(timei, timef, nx)
            y_vals = np.linspace(Y_range[0], Y_range[-1], ny)
            X, Y = np.meshgrid(x_vals, y_vals)

            if i==0:
                my_contours = velx_contours
            elif i==1:
                my_contours = velz_contours
            elif i==4:
                my_contours = dens_contours
            elif i==6:
                my_contours = ps_contours
            elif i==7:
                my_contours = pseudo_entropy_contours

            im = ax[i].imshow(field_data, aspect='auto', origin='lower',
                        extent=(timei, timef, Y_range[0], Y_range[-1]),
                        cmap=cmaps[i])

            ax[i].contour(X, Y, field_data, levels=my_contours, colors='cyan', linewidths=1, linestyles='solid')

        elif (i==3):

            field_data = datasets[i].T

            nx = field_data.shape[1]
            ny = field_data.shape[0]

            x_vals = np.linspace(timei, timef, nx)
            y_vals = np.linspace(Y_range[0], Y_range[-1], ny)
            X, Y = np.meshgrid(x_vals, y_vals)

            my_contours = temp_contours

            im = ax[i].imshow(datasets[i].T, aspect='auto', origin='lower', 
                        extent=(timei, timef, Y_range[0], Y_range[-1]), 
                        cmap=cmaps[i], norm="log")
            
            ax[i].contour(X, Y, field_data, levels=my_contours, colors='cyan', linewidths=1, linestyles='solid')

        else : 
            im = ax[i].imshow(datasets[i].T, aspect='auto', origin='lower', 
                        extent=(timei, timef, Y_range[0], Y_range[-1]), 
                        cmap=cmaps[i])
        
        # Set limits and small fonts
        ax[i].set_xlim(timei, timef)
        ax[i].tick_params(axis='both', labelsize=12)
        ax[i].set_xlabel(r"$t/ t_{cool,min}$", fontsize=14)
        
        # Only label the Y-axis on the first plot to save space
        if i == 0 or i == 4:
            ax[i].set_ylabel(r"$z/\Delta u t_{cool,min}$", fontsize=14)
        else:
            ax[i].tick_params(labelleft=False)
        if i<4:
            ax[i].tick_params(labelbottom=False)

        # Make the individual colorbar vertical on the right
        cbar = fig.colorbar(im, ax=ax[i], orientation='horizontal', location='top', pad=0.02)
        cbar.ax.tick_params(labelsize=11)
        ax[i].text(0.05, 0.95,labels[i],transform=ax[i].transAxes,ha='left', va='top',fontsize=16,color='black',bbox=dict(facecolor='white',edgecolor='black',boxstyle='round,pad=0.2'))

        # Tight borders
        for spine in ax[i].spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)

    # Save the plot
    save_path = f"{dir}KH_spacetime_{frame}_time_averaged{start}to{end}with{jump}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def make_2D_paper_plots(i):

    global dir, NX, NY, NZ, F, max_level, delU, time_0, rho_h, T_inflection, jump, cold_frac, NY_init, NY_fin

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
        vx1_turb_rms_vol = f['vx1_turb_rms_vol']
        vx2_turb_rms_vol = f['vx2_turb_rms_vol']
        vx3_turb_rms_vol = f['vx3_turb_rms_vol']
        v_turb_rms_vol = f['v_turb_rms_vol']
    
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

    Tc0 = np.min(temp_vol71to250)
    Th0 = np.max(temp_vol71to250)
    dTdz = np.gradient(temp_vol71to250, Y_lims)
    z0_guess = Y_lims[np.argmax(np.abs(dTdz))]
    slope_max = np.max(np.abs(dTdz))
    x0_guess = (Th0 - Tc0) / (2 * slope_max)
    p0 = [x0_guess, z0_guess, Th0, Tc0]


    from scipy.optimize import curve_fit
    def T_tanh_model(z, x0, z0, Th, Tc):
        return ((0.5 * (Th + Tc) + 0.5 * (Th - Tc) * np.tanh((z - x0) / z0)))
    popt, pcov = curve_fit(T_tanh_model, Y_range_final, temp_vol71to250, p0=p0)
    print(f'Fitted parameters: x0={popt[0]}, z0={popt[1]}, Th={popt[2]}, Tc={popt[3]}')
    z0 = popt[1]
    y0 = popt[0]
    Y_lims -= y0
    Y_range_final -= y0

    im1 = ax[0,0].imshow(slice_to_half_2D(den)/rho_h, aspect='auto', origin='lower',cmap='inferno',extent = (ex[0], ex[1], NY_init, NY_fin), vmin=-10, vmax=110)

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
    ax[0,0].set_ylim(NY_init, NY_fin)
    axlogx1.set_ylim(NY_init, NY_fin)

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

    im2 = ax[0,1].imshow(slice_to_half_2D(prs)/P_0, aspect='auto', origin='lower', cmap='inferno', extent = (ex[0], ex[1], NY_init, NY_fin), vmin = 0.3, vmax = 1.5)
    ax[0,1].set_xlim(ex[0], ex[1])
    ax[0,1].set_xscale('linear')
    axlogx2 = ax[0,1].twiny()
    axlogx2.set_xscale('linear')
    axlogx2.set_xlim(0.3, 1.5)
    ax[0,1].set_ylim(NY_init, NY_fin)
    axlogx2.set_ylim(NY_init, NY_fin)
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
    im3 = ax[1,0].imshow(slice_to_half_2D(t_cool)/time_0, aspect='auto', origin='lower', norm = 'log', cmap='inferno', extent = (ex[0], ex[1], NY_init, NY_fin),vmin = 1e-1*np.min(slice_to_half(t_cool_av71to250)/time_0),vmax = 1e3*np.min(slice_to_half(t_cool_av71to250)/time_0))

    ax[1,0].set_xlim(ex[0], ex[1])
    ax[1,0].set_xscale('linear')
    ax_logx3 = ax[1,0].twiny()
    ax_logx3.set_xscale('log')
    ax[1,0].set_ylim(NY_init, NY_fin)
    ax_logx3.set_ylim(NY_init, NY_fin)
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

    im4 = ax[1,1].imshow((slice_to_half_2D(v_turb_rms))/delU, aspect='auto', origin='lower', norm = 'log', cmap='inferno',extent = (ex[0], ex[1], NY_init, NY_fin), vmin=1e-2, vmax=5)
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
    ax[1,1].set_ylim(NY_init, NY_fin)
    ax_logx4.set_ylim(NY_init, NY_fin)
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
    plt.close(fig)

def plot_sliced_PDFs(axis, number):
    slices = 19
    for n in range(number, number + 1):
        print(n)
        plt.figure(figsize=(16, 9))
        for i in range(0, slices+1):
            plt.subplot(4, 5, i+1)
            with np.load(dir+'KH_tempPDF_snapshot_'+str(n).zfill(5)+f'C{1}_axis{axis}_slice{i}.npz') as data:
                bin_centers_ = data['bin_centers']
                hist_mass_ = data['hist_mass']
                hist_vol_ = data['hist_vol']
                hist_emis_ = data['hist_emissivity']
            plt.plot(bin_centers_, hist_mass_, label=r'$\mathcal{P}_M$')
            plt.plot(bin_centers_, hist_vol_, label=r'$\mathcal{P}_V$')
            plt.plot(bin_centers_, hist_emis_, label=r'$\mathcal{P}_E$')
            plt.yscale('log')
            plt.grid(which='both', axis='both', linestyle='--', linewidth=0.25, color='gray')
            if (i+1)%5 != 1:
                plt.yticks([])
            if (i+1) <= 15:
                plt.xticks([])
            plt.xlim(3.9,6.05)
            plt.ylim(1e-3, 50)
        plt.suptitle(f'Sliced PDFs for Snapshot {n} along {axis}-axis, y = {round(-20+(1/2)*40,2)}-{round(-20+(7/8)*40, 2)}', fontsize=16)
        plt.legend()
        plt.tight_layout()
        #plt.show()
        plt.savefig(dir + f"KH_tempPDF_slice_{axis}_{n}.png", bbox_inches='tight')
        plt.clf()
        plt.close()

if __name__ == "__main__":

    global time_0

    T_0 = 1e5
    time_mid = 5. * (T_0/TEMPERATURE)**2 / (2*P_0 * np.vectorize(ISMCoolFn)(T_0)/COOLING_UNIT)
    print(f"Cooling time at T_0 = {T_0} K is {time_mid} code units")

    temp_arr = np.logspace(4, 6, 1000)
    P_0_array = P_0 * np.ones_like(temp_arr)
    Lambda_fn = np.vectorize(ISMCoolFn)(temp_arr)/COOLING_UNIT
    cooling_arr = np.divide(5. * (temp_arr/TEMPERATURE)**2 , 2*P_0_array * Lambda_fn, out=np.full_like(temp_arr, math.inf, dtype=float), 
                    where=Lambda_fn != 0)
    time_0 = np.min(cooling_arr)
    print(f"Characteristic cooling time (time_0) = {time_0} code units")
    print(f"Ratio of cooling time at 1e5 to minimum cooling time is {time_mid/time_0}")

    n_i = 0
    n_f = 250
    if MPI_DEF:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        nfiles_local = (n_f - n_i + 1) // size
        nproc_extra = (n_f - n_i + 1) % size
        N1_local = rank * nfiles_local + n_i
        N2_local = (rank + 1) * nfiles_local + n_i
    global n1, n2, n3, n4
    n1 = 71
    n4 = 250
    n2 = n1 + (n4 - n1 +1)//3
    n3 = n2 + (n4 - n1 +1)//3
    begin = n1
    finish = n4
    skip = 1
    global jump
    jump = skip
    if not MPI_DEF:
        find_temporal_z0(0,250,skip)
        make_1D(0,250,skip)
        print("1D arrays created successfully.")
        make_spacetime_plots(250, trmlframeflag = True)
        print("Spacetime plots in TRML frame created successfully.")
        make_spacetime_plots(250, trmlframeflag = False)
        print("Spacetime plots in simulation frame created successfully.")
        """make_1D(36,125,skip)
        make_1D(36,65,skip)
        make_1D(66,95,skip)
        make_1D(96,125,skip)
        make_PDF(36,125,skip)
        make_PDF(36,65,skip)
        make_PDF(66,95,skip)
        make_PDF(96,125,skip)"""
    if MPI_DEF:
        for N in range(N1_local, N2_local):
            make_2D_paper_plots(N)
            plot_sliced_PDFs('x',N)
            plot_sliced_PDFs('y',N)
            plot_sliced_PDFs('z',N)
        if rank < nproc_extra: # Process extra files across ranks
            n = size * nfiles_local + rank + n_i
            make_2D_paper_plots(n)
            plot_sliced_PDFs('x',n)
            plot_sliced_PDFs('y',n)
            plot_sliced_PDFs('z',n)
    if not MPI_DEF:        
        """find_temporal_z0(36,125,skip)
        find_temporal_z0(36,65,skip)
        find_temporal_z0(66,95,skip)
        find_temporal_z0(96,125,skip)"""
    print("1D arrays created successfully.")

