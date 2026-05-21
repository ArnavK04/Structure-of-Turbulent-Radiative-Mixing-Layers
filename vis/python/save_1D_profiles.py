# Read 2D arrays, make horizontally averaged profiles, 2D snapshots, and probability density function of different variables.

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

from save_2D_arrays_3D import ISMCoolFn

def SetGlobals(path_to_files, F):

    global max_level, fact, dir, DX, DY, DZ
    global NX, NY, NZ, XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX

    dir = path_to_files
    fname = dir + 'KH.hydro_w.' + str(0).zfill(5) + '.bin'
    file_data = bin_convert.read_binary(fname)

    nmb = file_data['n_mbs']                      # Number of mesh blocks
    max_level = max(file_data['mb_logical'][i][3] for i in range(nmb))  # Get the maximum level of refinement in the mesh blocks
    fact = 2**max_level
    
    NX = file_data['Nx1']*fact//F
    NY = file_data['Nx2']*fact//F
    NZ = file_data['Nx3']*fact//F
    XMIN = file_data['x1min']
    XMAX = file_data['x1max']
    YMIN = file_data['x2min']
    YMAX = file_data['x2max']
    ZMIN = file_data['x3min']
    ZMAX = file_data['x3max']

    DX = (XMAX - XMIN) / NX
    DY = (YMAX - YMIN) / NY
    DZ = (ZMAX - ZMIN) / NZ

def CoarseByFactor(array, factor):
    """
    Coarse a 3D array by a given factor using vectorized operations.
    Much faster than nested loops.
    """
    if factor <= 1:
        return array

    # Calculate the new shape
    new_shape = (array.shape[0] // factor, array.shape[1] // factor, array.shape[2] // factor)
    
    # Crop array to be evenly divisible by factor
    cropped = array[:new_shape[0]*factor, :new_shape[1]*factor, :new_shape[2]*factor]
    
    # Reshape to group elements that will be averaged together
    reshaped = cropped.reshape(new_shape[0], factor, new_shape[1], factor, new_shape[2], factor)
    
    del cropped
    gc.collect()
    # Take mean along the factor dimensions (axes 1, 3, 5)
    return np.mean(reshaped, axis=(1, 3, 5))   

def ReadBinFile(path_to_files, n, F):

    dir = path_to_files
    fname = dir + 'KH.hydro_w.' + str(n).zfill(5) + '.bin'
    file_data = bin_convert.read_binary(fname)    

    # Read and coarsen data one array at a time to minimize memory usage
    den = CoarseByFactor(analyse_bin.make_3D_array(file_data, 'dens'), F)
    vx1 = CoarseByFactor(analyse_bin.make_3D_array(file_data, 'velx'), F)
    vx3 = CoarseByFactor(analyse_bin.make_3D_array(file_data, 'velz'), F)
    ps = CoarseByFactor(analyse_bin.make_3D_array(file_data, 's_00'), F)
    
    # Calculate pressure directly from energy
    eint_array = analyse_bin.make_3D_array(file_data, 'eint')
    prs = CoarseByFactor((2./3.) * eint_array, F)
    del eint_array  # Free memory immediately
    gc.collect()

    temp = TEMPERATURE * prs/den
    
    vx2 = CoarseByFactor(analyse_bin.make_3D_array(file_data, 'vely'), F)
    
    # Don't pre-calculate temperature here - calculate it when needed
    times = file_data['time']
    
    # Clear file_data from memory
    del file_data
    gc.collect()
    
    return den, vx1, vx3, ps, prs, vx2, temp, times

def ComputeTRML_velocity(den, vx1, vx2, vx3, n, F):

    dens_cold = np.mean(den[:, :NY//3,:])    # cold density
    #v1_cold = np.mean(vx1[:, :NY//3, :])     # cold velocity-x1
    v2_cold = np.mean(vx2[:,:NY//3, :])     # cold velocity-x2
    #v3_cold = np.mean(vx3[:,:NY//3, :])    

    dens_hot = np.mean(den[:,NY*4//5:,:] )    # hot density
    #v1_hot = np.mean(vx1[:,NY*4//5:,:] )      # hot velocity-x1
    v2_hot = np.mean(vx2[:,NY*4//5:,:] )      # hot velocity-x2
    #v3_hot = np.mean(vx3[:,NY*4//5:,:] )      

    #v1_TRML_ = (v1_cold + v1_hot) / 2.0  # TRML velocity-x1
    v2_TRML_ = (dens_cold * v2_cold - dens_hot * v2_hot) / (dens_cold - dens_hot)  # TRML velocity-x2
    #v3_TRML_ = (v3_cold + v3_hot) / 2.0  

    v1_TRML_ = 0.0
    v3_TRML_ = 0.0

    return v1_TRML_, v2_TRML_, v3_TRML_

def Save1D_arrays(den, vx1, vx3, ps, prs, vx2, temp, tim, n, F):

    Y = np.linspace(YMIN, YMAX, NY)                 # Create an array of y coordinates

    den_1D = np.mean(den,axis=(0,2))
    den_var = np.mean((den - den_1D[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    den_sig = np.sqrt(den_var)    # Calculate the standard deviation of density
    vx1_1D = np.mean(vx1, axis=(0,2))
    vx1_var = np.mean((vx1 - vx1_1D[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    vx1_sig = np.sqrt(vx1_var)
    vx2_1D = np.mean(vx2, axis=(0,2))
    vx2_var = np.mean((vx2 - vx2_1D[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    vx2_sig = np.sqrt(vx2_var)
    vx3_1D = np.mean(vx3, axis=(0,2))
    vx3_var = np.mean((vx3 - vx3_1D[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    vx3_sig = np.sqrt(vx3_var)
    ps_1D = np.mean(ps, axis=(0,2))
    ps_var = np.mean((ps - ps_1D[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    ps_sig = np.sqrt(ps_var)
    prs_1D = np.mean(prs, axis=(0,2))
    prs_var = np.mean((prs - prs_1D[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    prs_sig = np.sqrt(prs_var)
    temp_1D = np.mean(temp, axis=(0,2))
    temp_var = np.mean((temp - temp_1D[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    temp_sig = np.sqrt(temp_var)

    # mass weighted

    vx1_mw = np.sum(vx1 * den, axis=(0,2)) / np.sum(den, axis=(0,2))
    vx1_var = np.sum(den * (vx1 - vx1_mw[np.newaxis, :, np.newaxis])**2, axis=(0,2))/ np.sum(den, axis=(0,2))
    vx1_sig_mw = np.sqrt(vx1_var)
    vx2_mw = np.sum(vx2 * den, axis=(0,2)) / np.sum(den, axis=(0,2))
    vx2_var = np.sum(den * (vx2 - vx2_mw[np.newaxis, :, np.newaxis])**2, axis=(0,2))/ np.sum(den, axis=(0,2))
    vx2_sig_mw = np.sqrt(vx2_var)
    vx3_mw = np.sum(vx3 * den, axis=(0,2)) / np.sum(den, axis=(0,2))
    vx3_var = np.sum(den * (vx3 - vx3_mw[np.newaxis, :, np.newaxis])**2, axis=(0,2))/ np.sum(den, axis=(0,2))
    vx3_sig_mw = np.sqrt(vx3_var)
    ps_mw = np.sum(ps * den, axis=(0,2)) / np.sum(den, axis=(0,2))
    ps_var = np.sum(den * (ps - ps_mw[np.newaxis, :, np.newaxis])**2, axis=(0,2))/ np.sum(den, axis=(0,2))
    ps_sig_mw = np.sqrt(ps_var)
    prs_mw = np.sum(prs * den, axis=(0,2)) / np.sum(den, axis=(0,2))
    prs_var = np.sum(den * (prs - prs_mw[np.newaxis, :, np.newaxis])**2, axis=(0,2))/ np.sum(den, axis=(0,2))
    prs_sig_mw = np.sqrt(prs_var)
    temp_mw = np.sum(temp * den, axis=(0,2)) / np.sum(den, axis=(0,2))
    temp_var = np.sum(den * (temp - temp_mw[np.newaxis, :, np.newaxis])**2, axis=(0,2))/ np.sum(den, axis=(0,2))
    temp_sig_mw = np.sqrt(temp_var)

    # fluxes

    rho_vx1 = np.mean(den * vx1, axis=(0,2))
    rho_vx1_var = np.mean((den * vx1 - rho_vx1[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    rho_vx1_sig = np.sqrt(rho_vx1_var)
    rho_avX_vx1_av = den_1D * vx1_1D
    del_rho = den - den_1D[np.newaxis, :, np.newaxis]
    del_vx1 = vx1 - vx1_1D[np.newaxis, :, np.newaxis]
    del_rhoX_del_vx1_av = np.mean(del_rho * del_vx1, axis=(0,2))
    rho_vx2 = np.mean(den * vx2, axis=(0,2))
    rho_avX_vx2_av = den_1D * vx2_1D
    del_vx2 = vx2 - vx2_1D[np.newaxis, :, np.newaxis]
    del_rhoX_del_vx2_av = np.mean(del_rho * del_vx2, axis=(0,2))
    rho_vx2_var = np.mean((den * vx2 - rho_vx2[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    rho_vx2_sig = np.sqrt(rho_vx2_var)
    rho_vx3 = np.mean(den * vx3, axis=(0,2))
    rho_vx3_var = np.mean((den * vx3 - rho_vx3[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    rho_vx3_sig = np.sqrt(rho_vx3_var)
    rho_vx1_vx2 = np.mean(den * vx1 * vx2, axis=(0,2))
    rho_vx1_vx2_var = np.mean((den * vx1 * vx2 - rho_vx1_vx2[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    rho_vx1_vx2_sig = np.sqrt(rho_vx1_vx2_var)
    rho_vx2_vx2 = np.mean(den * vx2 * vx2, axis=(0,2))
    rho_vx2_vx2_var = np.mean((den * vx2 * vx2 - rho_vx2_vx2[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    rho_vx2_vx2_sig = np.sqrt(rho_vx2_vx2_var)
    rho_vx2_vx3 = np.mean(den * vx2 * vx3, axis=(0,2))
    rho_vx2_vx3_var = np.mean((den * vx2 * vx3 - rho_vx2_vx3[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    rho_vx2_vx3_sig = np.sqrt(rho_vx2_vx3_var)


    # energy
    E = 0.5 * den * (vx1**2 + vx2**2 + vx3**2) + (3./2.)*prs
    total_energy = np.mean(E, axis=(0,2))
    total_energy_var = np.mean((E - total_energy[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    total_energy_sig = np.sqrt(total_energy_var)

    Be = 0.5 * (vx1**2 + vx2**2 + vx3**2) + (5./2.)*prs/den
    Be_vol = np.mean(Be, axis=(0,2))
    Be_vol_var = np.mean((Be - Be_vol[np.newaxis, :, np.newaxis])**2, axis=(0,2))
    Be_vol_sig = np.sqrt(Be_vol_var)
    Be_mw = np.mean(den * Be, axis=(0,2)) / np.mean(den, axis=(0,2))
    Be_mw_var = np.sum(den * (Be - Be_mw[np.newaxis, :, np.newaxis])**2, axis=(0,2)) / np.sum(den, axis=(0,2))
    Be_sig_mw = np.sqrt(Be_mw_var)

    emis = den * den * np.vectorize(ISMCoolFn, otypes = 'd')(temp) / COOLING_UNIT

    emis_vol = np.mean(emis, axis=(0,2))
    emis_vol_var = np.mean((emis - emis_vol[np.newaxis, :, np.newaxis]) **2, axis=(0,2))
    emis_sig = np.sqrt(emis_vol_var)
    emis_mw = np.mean(den * emis ,axis=(0,2))/np.mean(den,axis=(0,2))
    emis_mw_var = np.sum(den * (emis - emis_mw[np.newaxis, :, np.newaxis]) **2, axis=(0,2))/ np.sum(den, axis=(0,2))
    emis_sig_mw = np.sqrt(emis_mw_var)

    rhoXvx1 = den * vx1
    rhoXvx2 = den * vx2
    rhoXvx3 = den * vx3
    rhoXvx1_vol = np.mean(rhoXvx1, axis=(0,2))
    rhoXvx2_vol = np.mean(rhoXvx2, axis=(0,2))
    rhoXvx3_vol = np.mean(rhoXvx3, axis=(0,2))
    den_vol_ = np.mean(den, axis=(0,2))
    vx1_turb = (rhoXvx1 - rhoXvx1_vol[np.newaxis,:,np.newaxis]) / den_vol_[np.newaxis,:,np.newaxis]
    vx2_turb = (rhoXvx2 - rhoXvx2_vol[np.newaxis,:,np.newaxis]) / den_vol_[np.newaxis,:,np.newaxis]
    vx3_turb = (rhoXvx3 - rhoXvx3_vol[np.newaxis,:,np.newaxis]) / den_vol_[np.newaxis,:,np.newaxis]
    vx1_turb_rms_vol = np.sqrt(np.mean(vx1_turb**2, axis=(0,2)))
    vx2_turb_rms_vol = np.sqrt(np.mean(vx2_turb**2, axis=(0,2)))
    vx3_turb_rms_vol = np.sqrt(np.mean(vx3_turb**2, axis=(0,2)))

    v_turb_rms = np.sqrt(vx1_turb**2 + vx2_turb**2 + vx3_turb**2) 
    v_turb_rms_vol = np.sqrt(np.mean(v_turb_rms**2, axis=(0,2)))

    den_mean = np.mean(den)
    vx1_turb_vol_ = (rhoXvx1 - np.mean(rhoXvx1))/den_mean
    vx2_turb_vol_ = (rhoXvx2 - np.mean(rhoXvx2))/den_mean
    vx3_turb_vol_ = (rhoXvx3 - np.mean(rhoXvx3))/den_mean
    v_turb_mean_wholebox1 = np.sqrt(np.mean(vx1_turb_vol_**2))
    v_turb_mean_wholebox2 = np.sqrt(np.mean(vx2_turb_vol_**2))
    v_turb_mean_wholebox3 = np.sqrt(np.mean(vx3_turb_vol_**2))
    v_turb_mean_wholebox = np.sqrt(v_turb_mean_wholebox1**2 + v_turb_mean_wholebox2**2 + v_turb_mean_wholebox3**2)

    # mass weighted turbulent velocity
    fluc_vx1 = vx1 - vx1_mw[np.newaxis, :, np.newaxis]
    fluc_vx2 = vx2 - vx2_mw[np.newaxis, :, np.newaxis]
    fluc_vx3 = vx3 - vx3_mw[np.newaxis, :, np.newaxis]
    vx1_turb_mw = np.sqrt(np.sum(den * fluc_vx1**2, axis=(0,2)) / np.sum(den, axis=(0,2)))
    vx2_turb_mw = np.sqrt(np.sum(den * fluc_vx2**2, axis=(0,2)) / np.sum(den, axis=(0,2)))
    vx3_turb_mw = np.sqrt(np.sum(den * fluc_vx3**2, axis=(0,2)) / np.sum(den, axis=(0,2)))
    v_turb_rms_mw = np.sqrt(vx1_turb_mw**2 + vx2_turb_mw**2 + vx3_turb_mw**2)

    vx1_mw_mean_wholebox = np.sum(vx1 * den) / np.sum(den)
    vx2_mw_mean_wholebox = np.sum(vx2 * den) / np.sum(den)
    vx3_mw_mean_wholebox = np.sum(vx3 * den) / np.sum(den)
    v_turb_mw_whole_box1 = math.sqrt(np.sum(den * ((vx1 - vx1_mw_mean_wholebox)**2)) / np.sum(den))
    v_turb_mw_whole_box2 = math.sqrt(np.sum(den * ((vx2 - vx2_mw_mean_wholebox)**2)) / np.sum(den))
    v_turb_mw_whole_box3 = math.sqrt(np.sum(den * ((vx3 - vx3_mw_mean_wholebox)**2)) / np.sum(den))
    v_turb_mw_whole_box = math.sqrt(v_turb_mw_whole_box1**2 + v_turb_mw_whole_box2**2 + v_turb_mw_whole_box3**2)
    # Save 1D arrays to a compressed .npz file

    np.savez_compressed(dir + 'KH_1D_arrays_snapshot_' + str(n).zfill(5) + f'C{F}.npz',
                        Y_lims=Y, den_vol=den_1D,
                        den_sig_vol=den_sig,
                        vx1_vol=vx1_1D,
                        vx1_sig_vol=vx1_sig,
                        vx2_vol=vx2_1D,
                        vx2_sig_vol=vx2_sig,
                        vx3_vol=vx3_1D,
                        vx3_sig_vol=vx3_sig,
                        ps_vol=ps_1D,
                        ps_sig_vol=ps_sig,
                        prs_vol=prs_1D,
                        prs_sig_vol=prs_sig,
                        temp_vol=temp_1D,
                        temp_sig_vol=temp_sig,
                        vx1_mw=vx1_mw,
                        vx1_sig_mw=vx1_sig_mw,
                        vx2_mw=vx2_mw,
                        vx2_sig_mw=vx2_sig_mw,
                        vx3_mw=vx3_mw,
                        vx3_sig_mw=vx3_sig_mw,
                        ps_mw=ps_mw,
                        ps_sig_mw=ps_sig_mw,
                        prs_mw=prs_mw,
                        prs_sig_mw=prs_sig_mw,
                        temp_mw=temp_mw,
                        temp_sig_mw=temp_sig_mw,
                        rho_vx1_vol=rho_vx1, 
                        rho_vx1_sig_vol=rho_vx1_sig, 
                        rho_avX_vx1_av=rho_avX_vx1_av, 
                        del_rhoX_del_vx1_av=del_rhoX_del_vx1_av, 
                        rho_vx2_vol=rho_vx2, 
                        rho_vx2_sig_vol=rho_vx2_sig, 
                        rho_avX_vx2_av=rho_avX_vx2_av, 
                        del_rhoX_del_vx2_av=del_rhoX_del_vx2_av, 
                        rho_vx3_vol=rho_vx3, 
                        rho_vx3_sig_vol=rho_vx3_sig, 
                        rho_vx1_vx2=rho_vx1_vx2, 
                        rho_vx1_vx2_sig=rho_vx1_vx2_sig, 
                        rho_vx2_vx2=rho_vx2_vx2, 
                        rho_vx2_vx2_sig=rho_vx2_vx2_sig, 
                        rho_vx2_vx3=rho_vx2_vx3, 
                        rho_vx2_vx3_sig=rho_vx2_vx3_sig,
                        total_energy_dens=total_energy,
                        total_energy_sig=total_energy_sig,
                        Be_vol = Be_vol, Be_vol_sig=Be_vol_sig,
                        Be_mw=Be_mw, Be_sig_mw=Be_sig_mw,
                        emis_vol=emis_vol, vx1_turb_rms_vol=vx1_turb_rms_vol,
                        emis_sig=emis_sig, emis_mw=emis_mw, emis_sig_mw=emis_sig_mw,
                        vx2_turb_rms_vol=vx2_turb_rms_vol, vx3_turb_rms_vol=vx3_turb_rms_vol,
                        v_turb_rms_vol=v_turb_rms_vol,
                        vx1_turb_mw=vx1_turb_mw, vx2_turb_mw=vx2_turb_mw, vx3_turb_mw=vx3_turb_mw,
                        v_turb_rms_mw=v_turb_rms_mw,
                        v_turb_rms_whole_box=v_turb_mw_whole_box, v_turb_rms_whole_box1=v_turb_mw_whole_box1, 
                        v_turb_rms_whole_box2=v_turb_mw_whole_box2, v_turb_rms_whole_box3=v_turb_mw_whole_box3,
                        v_turb_mean_wholebox = v_turb_mean_wholebox, v_turb_mean_wholebox1=v_turb_mean_wholebox1,
                        v_turb_mean_wholebox2=v_turb_mean_wholebox2,
                        number=n, time=tim, factor=F) 
    
def MakeSlicedPDFs(den, vx1, vx3, ps, prs, vx2, temp, tim, n, F, slices, axis='y'):

    if axis == 'y':
        slice_indices = np.linspace(0, den.shape[1]-1, slices+1, dtype=int)
        for i in range(slices):
            slice_data = {
                'den': den[:, slice_indices[i]:slice_indices[i+1], :].T,
                'vx1': vx1[:, slice_indices[i]:slice_indices[i+1], :].T,
                'vx3': vx3[:, slice_indices[i]:slice_indices[i+1], :].T,
                'ps': ps[:, slice_indices[i]:slice_indices[i+1], :].T,
                'prs': prs[:, slice_indices[i]:slice_indices[i+1], :].T,
                'vx2': vx2[:, slice_indices[i]:slice_indices[i+1], :].T,
                'temp': temp[:, slice_indices[i]:slice_indices[i+1], :].T
            }
            MakePDF_snapshots(**slice_data, tim=tim, n=n, F=F, axis=axis, slice=i)
    elif axis == 'x':
        slice_indices = np.linspace(0, den.shape[2]-1, slices+1, dtype=int)
        for i in range(slices):
            slice_data = {
                'den': den[: , : , slice_indices[i]:slice_indices[i+1]],
                'vx1': vx1[: , : , slice_indices[i]:slice_indices[i+1]],
                'vx3': vx3[: , : , slice_indices[i]:slice_indices[i+1]],
                'ps': ps[: , : , slice_indices[i]:slice_indices[i+1]],
                'prs': prs[: , : , slice_indices[i]:slice_indices[i+1]],
                'vx2': vx2[: , : , slice_indices[i]:slice_indices[i+1]],
                'temp': temp[: , : , slice_indices[i]:slice_indices[i+1]]
            }
            MakePDF_snapshots(**slice_data, tim=tim, n=n, F=F, axis=axis, slice=i)
    else:
        slice_indices = np.linspace(0, den.shape[0]-1, slices+1, dtype=int)
        for i in range(slices):
            slice_data = {
                'den': den[slice_indices[i]:slice_indices[i+1], : , : ],
                'vx1': vx1[slice_indices[i]:slice_indices[i+1], : , : ],
                'vx3': vx3[slice_indices[i]:slice_indices[i+1], : , : ],
                'ps': ps[slice_indices[i]:slice_indices[i+1], : , : ],
                'prs': prs[slice_indices[i]:slice_indices[i+1], : , : ],
                'vx2': vx2[slice_indices[i]:slice_indices[i+1], : , : ],
                'temp': temp[slice_indices[i]:slice_indices[i+1], : , : ]
            }
            MakePDF_snapshots(**slice_data, tim=tim, n=n, F=F, axis=axis, slice=i)
    
def MakePDF_snapshots(den, vx1, vx3, ps, prs, vx2, temp, tim, n, F, axis, slice):   

    den_flat = den.flatten()
    temp_flat = temp.flatten()

    del den, temp
    gc.collect()

    wtV = np.ones_like(den_flat)  # Volume weighting
    wtM = den_flat  # Mass weighting
    wtEM = den_flat * den_flat * np.vectorize(ISMCoolFn, otypes = 'd')(temp_flat)/COOLING_UNIT  # Energy weighting

    bins_temp = np.linspace(3.5, 6.5, 101)

    hist_vol, bin_edges = np.histogram(np.log10(temp_flat), bins=bins_temp, weights=wtV, density=True)
  
    global bin_centers
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    hist_mass, _ = np.histogram(np.log10(temp_flat), bins=bins_temp, weights=wtM, density=True)
    hist_emissivity, _ = np.histogram(np.log10(temp_flat), bins=bins_temp, weights=wtEM, density=True)

    del wtV, wtM, wtEM, den_flat, temp_flat
    gc.collect()

    np.savez_compressed(dir + 'KH_tempPDF_snapshot_' + str(n).zfill(5) + f'C{F}_axis{axis}_slice{slice}.npz',
                        bin_centers=bin_centers,
                        hist_vol=hist_vol,
                        hist_mass=hist_mass,
                        hist_emissivity=hist_emissivity, number = n,
                        time = tim, factor = F, axis = axis, slice = slice)

    """plt.suptitle(str(Nx1) + 'x' + str(Nx2) + ' Snapshot ' + str(n) + 'SMR = ' + str(max_level) + ', Coarsening Factor = ' + str(F))
    plt.tight_layout()
    plt.savefig(dir + 'KH_tempPDF_snapshot_' + str(n).zfill(5) + f'C{F}' + 'plthist.png')
    plt.clf()
    plt.close()      

    plt.figure(figsize=(16, 9))

    plt.hist(np.log10(temp).flatten(), bins=100, density=True, histtype='step', log=True, label='Volume') #volume PDF
    plt.hist(np.log10(temp).flatten(), bins=100, density=True, histtype='step', weights=den.flatten(), log=True, label='Mass') #mass PDF
    plt.hist(np.log10(temp).flatten(), bins=100, density=True, histtype='step', weights=(den*den*np.vectorize(ISMCoolFn)(temp)).flatten(), log=True, label='Emissivity') #emissivity PDF
    plt.ylim(1e-3, 1e2)
    plt.xlim(3.5, 6.5)

    plt.legend(['Volume weighted', 'Mass weighted', 'Emissivity weighted'], loc='upper right')"""
             
def MakeJointPDFs(den, vx1, vx3, ps, prs, vx2, temp, tim, n, F, weight):  
    """
    Make joint PDF of certain variables.
    """

    if (weight == 'm'):
        wt = den.flatten()
        W = 'M'
    elif (weight == 'v'):
        wt = np.ones_like(den).flatten()
        W = 'V'
    
    logTemp_bins = np.linspace(3.5, 6.5, 150)

    """plt.figure(figsize=(16, 9))
    plt.subplot(2, 2, 1)
    logDen_bins = np.linspace(math.log10(densmin), math.log10(densmax), 150)
    hist, xedges, yedges = np.histogram2d(np.log10(den).flatten(), np.log10(temp).flatten(), bins=[logDen_bins, logTemp_bins], weights=wt, density=True)
    bin_centers_x = 0.5*(xedges[1:] + xedges[:-1])
    bin_centers_y = 0.5*(yedges[1:] + yedges[:-1])
    hist += 1e-6
    plt.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', origin='lower', cmap='inferno', norm='log')
    plt.ylim(3.5, 6.5)
    plt.xlim(math.log10(densmin), math.log10(densmax))
    plt.colorbar()
    plt.xlabel(r'$log_{10}(\rho)$')
    plt.ylabel(r'$log_{10}(T)$')
    plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.subplot(2, 2, 2)
    vx1_bins = np.linspace(vx1min - 10, vx1max + 10, 150)
    hist, xedges, yedges = np.histogram2d(vx1.flatten(), np.log10(temp).flatten(), bins=[vx1_bins, logTemp_bins], weights=wt, density=True)
    bin_centers_x = 0.5*(xedges[1:] + xedges[:-1])
    bin_centers_y = 0.5*(yedges[1:] + yedges[:-1])
    hist += 1e-6
    plt.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', origin='lower', cmap='inferno', norm='log')
    plt.ylim(3.5, 6.5)
    plt.xlim(vx1min - 10, vx1max + 10)
    plt.colorbar()
    plt.xlabel(r'$v_{x1}$')
    plt.ylabel(r'$log_{10}(T)$')
    plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.subplot(2, 2, 3)
    prs_bins = np.linspace(0, prsmax, 150)
    hist, xedges, yedges = np.histogram2d(prs.flatten(), np.log10(temp).flatten(), bins=[prs_bins, logTemp_bins], weights=wt, density=True)
    bin_centers_x = 0.5*(xedges[1:] + xedges[:-1])
    bin_centers_y = 0.5*(yedges[1:] + yedges[:-1])
    hist += 1e-6
    plt.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', origin='lower', cmap='inferno', norm='log')
    plt.ylim(3.5, 6.5)
    plt.xlim(0, prsmax)
    plt.colorbar()
    plt.xlabel(r'$P$')
    plt.ylabel(r'$log_{10}(T)$')
    plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.subplot(2, 2, 4)
    vx2_bins = np.linspace(-70, 50, 150)
    hist, xedges, yedges = np.histogram2d(vx2.flatten(), np.log10(temp).flatten(), bins=[vx2_bins, logTemp_bins], weights=wt, density=True)
    bin_centers_x = 0.5*(xedges[1:] + xedges[:-1])
    bin_centers_y = 0.5*(yedges[1:] + yedges[:-1])
    hist += 1e-6
    plt.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', origin='lower', cmap='inferno', norm='log')
    plt.ylim(3.5, 6.5)
    plt.xlim(-70, 50)
    plt.colorbar()
    plt.xlabel(r'$v_{x2}$')
    plt.ylabel(r"$log_{10}(T)$")
    plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.suptitle(str(int(NX*F/2**max_level)) + 'x' + str(int(NY*F/2**max_level)) + 'x' + str(int(NZ*F/2**max_level)) + ' Snapshot ' + str(n) + ', time = ' + str(tim) + ', SMR = ' + str(max_level) + ', Coarsening Factor = ' + str(F))
    plt.tight_layout()
    plt.savefig(dir + 'KH_jointPDF_snapshot_' + str(n).zfill(5) + f'C{F}' + '.png')
    plt.clf()
    plt.close()"""  

def main():
    if MPI_DEF:
        comm = MPI.COMM_WORLD
        global rank, size
        rank = comm.Get_rank()
        size = comm.Get_size()
        def parse_args():
            parser = argparse.ArgumentParser()
            parser.add_argument('-i', type=str, required=True, help="Path to input file")
            parser.add_argument('-n1', type=int, default=0, help="Start index for snapshots")
            parser.add_argument('-n2', type=int, default=51, help="End index for snapshots")
            parser.add_argument('-F', type=int, default=1, help="Coarsening factor")
            parser.add_argument('-z', type=int, default=0, help="index to slice at, along z-axis")
            parser.add_argument('-y', type=int, default=0, help="index to slice at, along y-axis")
            parser.add_argument('-x', type=int, default=0, help="index to slice at, along x-axis")
            return parser.parse_args()

        # Only rank 0 parses args
        if rank == 0:
            args = parse_args()
            arg_dict = vars(args)  # convert Namespace to dict
        else:
            arg_dict = None

        # Broadcast parsed arguments to all ranks
        arg_dict = comm.bcast(arg_dict, root=0)

        path_to_files = arg_dict['i']
        n1 = arg_dict['n1']
        n2 = arg_dict['n2']
        F = arg_dict['F']
        global X, Y, Z
        Z = arg_dict['z']
        Y = arg_dict['y']
        X = arg_dict['x']
        SetGlobals(path_to_files, F)

        nfiles_local = (n2 - n1 + 1) // size
        nproc_extra = (n2 - n1 + 1) % size
        N1_local = rank * nfiles_local + n1
        N2_local = (rank + 1) * nfiles_local + n1

    else:
        N1_local = 0 
        N2_local = 151

    """v1_tRML_local = 0.0
    v2_tRML_local = 0.0
    v3_tRML_local = 0.0
        
    for i in range(N1_local, N2_local):
        print(f"Processing snapshot {i} for TMRL vel...")
        dens, velx, velz, ps, prs, vely, temp, times = ReadBinFile(path_to_files, i, F)
        v1_tRML_local_, v2_tRML_local_, v3_tRML_local_ = ComputeTRML_velocity(dens, velx, vely, velz, i, F)
        v1_tRML_local += v1_tRML_local_
        v2_tRML_local += v2_tRML_local_
        v3_tRML_local += v3_tRML_local_

    if MPI_DEF:
        if rank < nproc_extra: # Process extra files across ranks
            n = size * nfiles_local + rank + n1
            print(f"Processing extra snapshot {n} for TRML vel...")
            dens, velx, velz, ps, prs, vely, temp, times = ReadBinFile(path_to_files, n, F)
            v1_tRML_local_, v2_tRML_local_, v3_tRML_local_ = ComputeTRML_velocity(dens, velx, vely, velz, n, F)
            v1_tRML_local += v1_tRML_local_
            v2_tRML_local += v2_tRML_local_
            v3_tRML_local += v3_tRML_local_

    comm.Barrier()  # Ensure all processes have computed their local TRML velocities before proceeding
    global v1_TRML, v2_TRML, v3_TRML
    v1_TRML = comm.allreduce(v1_tRML_local, op=MPI.SUM) / (n2-n1+1)
    v2_TRML = comm.allreduce(v2_tRML_local, op=MPI.SUM) / (n2-n1+1)
    v3_TRML = comm.allreduce(v3_tRML_local, op=MPI.SUM) / (n2-n1+1)

    # not tranforming vx, vz as of now
    v1_TRML = 0.0
    v3_TRML = 0.0

    if rank == 0:
        print(f"TRML velocity-x1 = {v1_TRML}, TRML velocity-x2 = {v2_TRML}", f"TRML velocity-x3 = {v3_TRML}")

    comm.Barrier()"""  
        
    global densmin, densmax, vx1min, vx1max, prsmin, prsmax, vx2min, vx2max, tempmin, tempmax, vx3min, vx3max
    
    dir1_ = path_to_files
    fname_ = dir1_ + 'KH.hydro_w.' + str(0).zfill(5) + '.bin'
    file_data_ = bin_convert.read_binary(fname_)    

    den_ = CoarseByFactor(analyse_bin.make_3D_array(file_data_, 'dens'),F)
    vx1_ = CoarseByFactor(analyse_bin.make_3D_array(file_data_, 'velx'),F)
    prs_ = CoarseByFactor((2./3.)*analyse_bin.make_3D_array(file_data_, 'eint'),F)
    temp_ = prs_*TEMPERATURE/den_

    densmin = 0.2*np.min(den_)
    densmax = 5*np.max(den_)
    vx1min = 5*np.min(vx1_) 
    vx1max = 1.5*np.max(vx1_)
    prsmin = 0
    prsmax = 1.6*np.max(prs_)
    vx2min = -50
    vx2max = 15
    tempmin = 0.2*np.min(temp_)
    tempmax = 5*np.max(temp_)
    vx3min = -50
    vx3max = 50

    global slices
    slices = 40

    del den_, vx1_, prs_, temp_
    gc.collect()
 
    for i in range(N1_local, N2_local):
        print(f"Processing snapshot {i}...")
        dens, velx, velz, ps, prs, vely, temp, t = ReadBinFile(path_to_files, i, F)
        v1_tRML_local_, v2_tRML_local_, v3_tRML_local_ = ComputeTRML_velocity(dens, velx, vely, velz, i, F)
        vely = vely - v2_tRML_local_
        Save1D_arrays(dens, velx, velz, ps, prs, vely, temp, t, i, F)
        MakePDF_snapshots(dens, velx, velz, ps, prs, vely, temp, t, i, F, axis='none', slice=1)
        MakeSlicedPDFs(dens, velx, velz, ps, prs, vely, temp, t, i, F, slices, axis='y')
        MakeSlicedPDFs(dens, velx, velz, ps, prs, vely, temp, t, i, F, slices, axis='x')
        MakeSlicedPDFs(dens, velx, velz, ps, prs, vely, temp, t, i, F, slices, axis='z')
        # Explicitly delete variables to free memory
        del dens, velx, velz, ps, prs, vely, temp
        gc.collect()

    if MPI_DEF:
        if rank < nproc_extra: # Process extra files across ranks
            n = size * nfiles_local + rank + n1
            print(f"Processing extra snapshot {n}")
            dens, velx, velz, ps, prs, vely, temp, t = ReadBinFile(path_to_files, n, F)
            v1_tRML_local_, v2_tRML_local_, v3_tRML_local_ = ComputeTRML_velocity(dens, velx, vely, velz, n, F)
            vely = vely - v2_tRML_local_
            Save1D_arrays(dens, velx, velz, ps, prs, vely, temp, t, n, F)
            MakePDF_snapshots(dens, velx, velz, ps, prs, vely, temp, t, n, F, axis='none', slice=1)
            #MakeJointPDFs(dens, velx, velz, ps, prs, vely, temp, t, n, F, 'v')
            MakeSlicedPDFs(dens, velx, velz, ps, prs, vely, temp, t, n, F, slices, axis='y')
            MakeSlicedPDFs(dens, velx, velz, ps, prs, vely, temp, t, n, F, slices, axis='x')
            MakeSlicedPDFs(dens, velx, velz, ps, prs, vely, temp, t, n, F, slices, axis='z')
            
            # Clean up memory
            del dens, velx, velz, ps, prs, vely, temp
            gc.collect()
      
if __name__ == "__main__":
    main()












