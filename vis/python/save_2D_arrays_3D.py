
# Read the binary files, and save 2D slices of arrays from them.

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

# Set numpy to use a reasonable memory limit
np.seterr(over='ignore')  # Ignore overflow warnings to save memory on checks  

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

    if rank == 0:
        header = file_data['header'] 
        with open(dir + 'header.txt', 'w') as f:
            for value in header:
                f.write(f"{value}\n")

def ISMCoolFn(temp):
    # original data from Shure et al. paper, covers 4.12 < logt < 8.16
    lhd = [
        -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
        -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
        -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
        -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
        -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
        -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
        -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
        -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
        -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
        -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
        -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
        -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
        -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928]
  
    logt = math.log10(temp)
    
    # for temperatures less than 10^4 K, use Koyama & Inutsuka (2002)
    if (logt <= 4.2):
      if (temp < 1.05e4):
        return 0.0
      else: 
        return (2.0e-19*math.exp(-1.184e5/(temp + 1.0e3)) + 2.8e-28*math.sqrt(temp)*math.exp(-92.0/temp))
  
    if (temp > 0.95e6):
      return 0.0
    # for temperatures above 10^8.15 use CGOLS fit
    if (logt > 8.15):
      return pow(10.0, (0.45*logt - 26.065))
  
    # in between values of 4.2 < log(T) < 8.15
    # linear interpolation of tabulated SPEX cooling rate
    ipps  = (int)(25.0*logt) - 103
    if (ipps >= 100):
       ipps = 100
    if (ipps <= 0):
       ipps = 0
    x0 = 4.12 + 0.04*(float)(ipps)
    dx = logt - x0
    logcool = (lhd[ipps+1]*dx - lhd[ipps]*(dx - 0.04))*25.0
    return pow(10.0,logcool)

def ISMCoolFn_nocool(temp):
    return 0.0

def ISMCoolFn_inv400less(temp):
    mean  = 5.0
    sigma = 0.2

    if (temp < 1.05e4 or temp > 0.95e6):
        return 0.0
    
    logt = math.log10(temp)
    lambda_T = math.exp(0.5 * ((logt - mean)/sigma)*((logt - mean)/sigma))

    norm = 5.253246378625558e-28
    return (norm/4.0) * lambda_T

def ISMCoolFn_normlog(temp):
    mean  = 5.0
    sigma = 0.2

    if (temp < 1.05e4 or temp > 0.95e6):
        return 0.0
    
    logt = math.log10(temp)
    lambda_T = math.exp(-0.5 * ((logt - mean)/sigma)*((logt - mean)/sigma))

    norm = 5.4591716620684276e-21
    return norm * lambda_T

def ISMCoolFn_const100less(temp):
    # return a constant cooling rate 
    if (temp < 1.05e4) or (temp > 0.95e6):
        return 0.0

    norm_factor_const_lambda = 7.457876781873938e-24

    return norm_factor_const_lambda

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
    """
    Read binary file and return coarsened data.
    Memory optimized to avoid storing intermediate full-resolution arrays.
    """
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
    
    vx2 = CoarseByFactor(analyse_bin.make_3D_array(file_data, 'vely'), F)
    
    # Don't pre-calculate temperature here - calculate it when needed
    times = file_data['time']
    
    # Clear file_data from memory
    del file_data
    gc.collect()
    
    return den, vx1, vx3, ps, prs, vx2, times

def Make2D_snapshotsX(den, vx1, vx3, ps, prs, vx2, tim, n, F):  
    """
    Create 2D snapshots in X direction with memory optimization.
    """
    # Create slices - these are views, not copies, so memory efficient
    den_slice = analyse_bin.give_slice(den, X, 'x')
    vx1_slice = analyse_bin.give_slice(vx1, X, 'x')
    vx3_slice = analyse_bin.give_slice(vx3, X, 'x')
    ps_slice = analyse_bin.give_slice(ps, X, 'x')
    prs_slice = analyse_bin.give_slice(prs, X, 'x')
    vx2_slice = analyse_bin.give_slice(vx2, X, 'x')

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

    vx1_turb_slice = analyse_bin.give_slice(vx1_turb, X, 'x')
    vx2_turb_slice = analyse_bin.give_slice(vx2_turb, X, 'x')
    vx3_turb_slice = analyse_bin.give_slice(vx3_turb, X, 'x')

    v_turb_rms = np.sqrt(vx1_turb**2 + vx2_turb**2 + vx3_turb**2) 
    v_turb_rms_slice = analyse_bin.give_slice(v_turb_rms, X, 'x')
    
    # Calculate temperature only for the slice
    temp_slice = TEMPERATURE * prs_slice / den_slice
    
    # Save each array with meta data
    np.savez_compressed(dir + 'KH_2D_X_' + str(X) + '_' + str(n).zfill(5) + f"_C{F}.npz",
                         den=den_slice, vx1=vx1_slice, vx3=vx3_slice,
                         ps=ps_slice, prs=prs_slice, vx2=vx2_slice,
                         vx1_turb=vx1_turb_slice, vx2_turb=vx2_turb_slice, vx3_turb=vx3_turb_slice,
                         temp=temp_slice, v_turb_rms = v_turb_rms_slice, extent=[ZMIN, ZMAX, YMIN, YMAX], ind=X, time=tim, number=n, Factor=F)

    # Clean up slices
    del den_slice, vx1_slice, vx3_slice, ps_slice, prs_slice, vx2_slice, temp_slice
    gc.collect()

def Make2D_snapshotsZ(den, vx1, vx3, ps, prs, vx2, tim, n, F):
    """
    Create 2D snapshots in Z direction with memory optimization.
    """
    # Create slices - these are views, not copies, so memory efficient
    den_slice = analyse_bin.give_slice(den, Z, 'z')
    vx1_slice = analyse_bin.give_slice(vx1, Z, 'z')
    vx3_slice = analyse_bin.give_slice(vx3, Z, 'z')
    ps_slice = analyse_bin.give_slice(ps, Z, 'z')
    prs_slice = analyse_bin.give_slice(prs, Z, 'z')
    vx2_slice = analyse_bin.give_slice(vx2, Z, 'z')

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

    vx1_turb_slice = analyse_bin.give_slice(vx1_turb, Z, 'z')
    vx2_turb_slice = analyse_bin.give_slice(vx2_turb, Z, 'z')
    vx3_turb_slice = analyse_bin.give_slice(vx3_turb, Z, 'z')

    v_turb_rms = np.sqrt(vx1_turb**2 + vx2_turb**2 + vx3_turb**2) 
    v_turb_rms_slice = analyse_bin.give_slice(v_turb_rms, Z, 'z')

    # Calculate temperature only for the slice
    temp_slice = TEMPERATURE * prs_slice / den_slice

    np.savez_compressed(dir + 'KH_2D_Z_' + str(Z) + '_' + str(n).zfill(5) + f"_C{F}.npz",
                         den=den_slice, vx1=vx1_slice, vx3=vx3_slice,
                         ps=ps_slice, prs=prs_slice, vx2=vx2_slice,
                         vx1_turb=vx1_turb_slice, vx2_turb=vx2_turb_slice, vx3_turb=vx3_turb_slice,
                         temp=temp_slice, v_turb_rms=v_turb_rms_slice, extent=[XMIN, XMAX, YMIN, YMAX], ind=Z, time=tim, number=n, Factor=F)

    # Clean up slices
    del den_slice, vx1_slice, vx3_slice, ps_slice, prs_slice, vx2_slice, temp_slice
    gc.collect()

def Make2D_snapshotsY(den, vx1, vx3, ps, prs, vx2, tim, n, F):
    """
    Create 2D snapshots in Y direction with memory optimization.
    """
    # Create slices - these are views, not copies, so memory efficient
    den_slice = analyse_bin.give_slice(den, Y, 'y')
    vx1_slice = analyse_bin.give_slice(vx1, Y, 'y')
    vx3_slice = analyse_bin.give_slice(vx3, Y, 'y')
    ps_slice = analyse_bin.give_slice(ps, Y, 'y')
    prs_slice = analyse_bin.give_slice(prs, Y, 'y')
    vx2_slice = analyse_bin.give_slice(vx2, Y, 'y')

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

    vx1_turb_slice = analyse_bin.give_slice(vx1_turb, Y, 'y')
    vx2_turb_slice = analyse_bin.give_slice(vx2_turb, Y, 'y')
    vx3_turb_slice = analyse_bin.give_slice(vx3_turb, Y, 'y')

    v_turb_rms = np.sqrt(vx1_turb**2 + vx2_turb**2 + vx3_turb**2) 
    v_turb_rms_slice = analyse_bin.give_slice(v_turb_rms, Y, 'y')

    # Calculate temperature only for the slice
    temp_slice = TEMPERATURE * prs_slice / den_slice

    np.savez_compressed(dir + 'KH_2D_Y_' + str(Y) + '_' + str(n).zfill(5) + f"_C{F}.npz",
                         den=den_slice, vx1=vx1_slice, vx3=vx3_slice,
                         ps=ps_slice, prs=prs_slice, vx2=vx2_slice,
                         vx1_turb=vx1_turb_slice, vx2_turb=vx2_turb_slice, vx3_turb=vx3_turb_slice,
                         temp=temp_slice, v_turb_rms=v_turb_rms_slice, extent=[ZMIN, ZMAX, XMIN, XMAX],
                         ind=Y, time=tim, number=n, Factor=F)

    # Clean up slices
    del den_slice, vx1_slice, vx3_slice, ps_slice, prs_slice, vx2_slice, temp_slice
    gc.collect()

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
        
    global densmin, densmax, vx1min, vx1max, prsmin, prsmax, vx2min, vx2max, tempmin, tempmax, vx3min, vx3max
    
    # Get initial data for setting min/max values - use smaller subset to save memory
    dir1_ = path_to_files
    fname_ = dir1_ + 'KH.hydro_w.' + str(0).zfill(5) + '.bin'
    file_data_ = bin_convert.read_binary(fname_)    

    # Sample data for determining ranges - use coarsened version to save memory
    den_sample = CoarseByFactor(analyse_bin.make_3D_array(file_data_, 'dens'), F)
    vx1_sample = CoarseByFactor(analyse_bin.make_3D_array(file_data_, 'velx'), F)
    eint_sample = analyse_bin.make_3D_array(file_data_, 'eint')
    prs_sample = CoarseByFactor((2./3.) * eint_sample, F)
    del eint_sample  # Free immediately
    temp_sample = prs_sample * TEMPERATURE / den_sample

    # Set global min/max values for consistent scaling
    densmin = 0.2 * np.min(den_sample)
    densmax = 5 * np.max(den_sample)
    vx1min = 5 * np.min(vx1_sample) 
    vx1max = 1.5 * np.max(vx1_sample)
    prsmin = 0
    prsmax = 1.6 * np.max(prs_sample)
    vx2min = -50
    vx2max = 15
    tempmin = 0.2 * np.min(temp_sample)
    tempmax = 5 * np.max(temp_sample)
    vx3min = -50
    vx3max = 50

    # Clean up sample data
    del den_sample, vx1_sample, prs_sample, temp_sample, file_data_
    gc.collect()
 
    # Main processing loop - memory optimized
    for i in range(N1_local, N2_local):
        print(f"Processing snapshot {i}...")

        # Load data once per iteration
        dens, velx, velz, ps, prs, vely, t = ReadBinFile(path_to_files, i, F)
        
        # Process each view without copying - pass the original arrays
        Make2D_snapshotsX(dens, velx, velz, ps, prs, vely, t, i, F)
        Make2D_snapshotsZ(dens, velx, velz, ps, prs, vely, t, i, F)
        Make2D_snapshotsY(dens, velx, velz, ps, prs, vely, t, i, F)
        
        # Explicitly delete arrays and collect garbage after each iteration
        del dens, velx, velz, ps, prs, vely
        gc.collect()


    if MPI_DEF:
        if rank < nproc_extra: # Process extra files across ranks
            n = size * nfiles_local + rank + n1
            print(f"Processing extra snapshot {n}")
            dens, velx, velz, ps, prs, vely, t = ReadBinFile(path_to_files, n, F)
            Make2D_snapshotsX(dens, velx, velz, ps, prs, vely, t, n, F)
            Make2D_snapshotsZ(dens, velx, velz, ps, prs, vely, t, n, F)
            Make2D_snapshotsY(dens, velx, velz, ps, prs, vely, t, n, F)
            
            # Clean up
            del dens, velx, velz, ps, prs, vely
            gc.collect()
      
if __name__ == "__main__":
    main()





    

    
    



