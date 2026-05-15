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

def Read2DfromnpzX(path_to_files, n, F):
   dir = path_to_files
   fname = dir + 'KH_2D_X_' + str(X) + '_' + str(n).zfill(5) + f"_C{F}.npz"

   with np.load(fname, 'r') as f:
      den = f['den']
      vx1 = f['vx1']
      vx3 = f['vx3']
      ps = f['ps']
      prs = f['prs']
      vx2 = f['vx2']
      temp = f['temp']
      tim = f['time']
      extent = f['extent']
      ind = f['ind']
      tim = f['time']

   return den, vx1, vx3, ps, prs, vx2, temp, extent, ind, tim

def Read2DfromnpzY(path_to_files, n, F):
    dir = path_to_files
    fname = dir + 'KH_2D_Y_' + str(Y) + '_' + str(n).zfill(5) + f"_C{F}.npz"
    
    with np.load(fname, 'r') as f:
        den = f['den']
        vx1 = f['vx1']
        vx3 = f['vx3']
        ps = f['ps']
        prs = f['prs']
        vx2 = f['vx2']
        temp = f['temp']
        tim = f['time']
        extent = f['extent']
        ind = f['ind']
        tim = f['time']
    
    return den, vx1, vx3, ps, prs, vx2, temp, extent, ind, tim

def Read2DfromnpzZ(path_to_files, n, F):
    dir = path_to_files
    fname = dir + 'KH_2D_Z_' + str(Z) + '_' + str(n).zfill(5) + f"_C{F}.npz"
    
    with np.load(fname, 'r') as f:
        den = f['den']
        vx1 = f['vx1']
        vx3 = f['vx3']
        ps = f['ps']
        prs = f['prs']
        vx2 = f['vx2']
        temp = f['temp']
        tim = f['time']
        extent = f['extent']
        ind = f['ind']
        tim = f['time']
    
    return den, vx1, vx3, ps, prs, vx2, temp, extent, ind, tim

def Make2D_snapshotsX(den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim, n, F):
    
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 7, 1)
    plt.imshow(den, vmin = densmin, vmax = densmax, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower', norm='log')
    plt.title('Density')
    plt.colorbar()

    plt.subplot(1, 7, 2)
    plt.imshow(ps, vmin = 0, vmax = 1, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('Passive Scalar')
    plt.colorbar()
    plt.yticks([])

    plt.subplot(1, 7, 3)
    plt.imshow(prs, vmin = prsmin, vmax = prsmax, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('Pressure')
    plt.colorbar()
    plt.yticks([])

    plt.subplot(1, 7, 4)
    plt.imshow(temp, vmin = tempmin, vmax = tempmax, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower', norm = 'log')
    plt.title('Temperature')
    plt.colorbar()

    plt.subplot(1, 7, 5)
    plt.imshow(vx1, vmin = vx1min -20 , vmax = vx1max + 20, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('X-Velocity')
    plt.colorbar()
    plt.yticks([])

    plt.subplot(1, 7, 6)
    plt.imshow(vx2,vmin = vx2min, vmax = vx2max, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('Y-velocity')
    plt.colorbar()
    plt.yticks([])
    
    plt.subplot(1, 7, 7)
    plt.imshow(vx3,vmin = vx3min, vmax = vx3max, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('Z-velocity')
    plt.colorbar()
    plt.yticks([])

    plt.suptitle(str(int(NX*F/2**max_level)) + 'x' + str(int(NY*F/2**max_level)) + 'x' + str(int(NZ*F/2**max_level)) + ' Snapshot ' + str(n) + ', time = ' + str(tim) + ', SMR = ' + str(max_level) + ', Coarsening Factor = ' + str(F))
    plt.tight_layout()
    plt.savefig(dir + 'KH_2D_snapshot_' + 'X_' + str(ind) + str(n).zfill(5) + f'C{F}' + '.png', dpi = 300)
    plt.clf()
    plt.close()

    # Clear variables to free memory immediately
    del den, vx1, vx3, ps, prs, vx2, temp
    gc.collect()

def Make2D_snapshotsZ(den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim, n, F):
    
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 7, 1)
    plt.imshow(den, vmin = densmin, vmax = densmax, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower', norm='log')
    plt.title('Density')
    plt.colorbar()

    plt.subplot(1, 7, 2)
    plt.imshow(ps, vmin = 0, vmax = 1, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('Passive Scalar')
    plt.colorbar()
    plt.yticks([])

    plt.subplot(1, 7, 3)
    plt.imshow(prs, vmin = prsmin, vmax = prsmax, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('Pressure')
    plt.colorbar()
    plt.yticks([])

    plt.subplot(1, 7, 4)
    plt.imshow(temp, vmin = tempmin, vmax = tempmax, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower', norm = 'log')
    plt.title('Temperature')
    plt.colorbar()

    plt.subplot(1, 7, 5)
    plt.imshow(vx1, vmin = vx1min -20 , vmax = vx1max + 20, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('X-Velocity')
    plt.colorbar()
    plt.yticks([])

    plt.subplot(1, 7, 6)
    plt.imshow(vx2,vmin = vx2min, vmax = vx2max, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('Y-velocity')
    plt.colorbar()
    plt.yticks([])
    
    plt.subplot(1, 7, 7)
    plt.imshow(vx3,vmin = vx3min, vmax = vx3max, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('Z-velocity')
    plt.colorbar()
    plt.yticks([])

    plt.suptitle(str(int(NX*F/2**max_level)) + 'x' + str(int(NY*F/2**max_level)) + 'x' + str(int(NZ*F/2**max_level)) + ' Snapshot ' + str(n) + ', time = ' + str(tim) + ', SMR = ' + str(max_level) + ', Coarsening Factor = ' + str(F))
    plt.tight_layout()
    plt.savefig(dir + 'KH_2D_snapshot_' + 'Z_' + str(ind) + str(n).zfill(5) + f'C{F}' + '.png', dpi = 300)
    plt.clf()
    plt.close()

    # Clear variables to free memory immediately
    del den, vx1, vx3, ps, prs, vx2, temp
    gc.collect()

def Make2D_snapshotsY(den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim, n, F):

    plt.figure(figsize=(12, 12))
    
    plt.subplot(3, 3, 1)
    plt.imshow(den, vmin = densmin, vmax = densmax, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower', norm='log')
    plt.title('Density')
    plt.colorbar()
    plt.xticks([])

    plt.subplot(3, 3, 2)
    plt.imshow(ps, vmin = 0, vmax = 1, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('Passive Scalar')
    plt.colorbar()
    plt.yticks([])
    plt.xticks([])

    plt.subplot(3, 3, 3)
    plt.imshow(prs, vmin = prsmin, vmax = prsmax, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('Pressure')
    plt.colorbar()
    plt.yticks([])
    plt.xticks([])

    plt.subplot(3, 3, 4)
    plt.imshow(temp, vmin = tempmin, vmax = tempmax, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower', norm = 'log')
    plt.title('Temperature')
    plt.colorbar()
    plt.xticks([])

    plt.subplot(3, 3, 5)
    plt.imshow(vx1, vmin = vx1min -20 , vmax = vx1max + 20, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('X-Velocity')
    plt.colorbar()
    plt.yticks([])
    plt.xticks([])

    plt.subplot(3, 3, 6)
    plt.imshow(vx2,vmin = vx2min, vmax = vx2max, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('Y-velocity')
    plt.colorbar()
    plt.yticks([])
    plt.xticks([])
    
    plt.subplot(3, 3, 8)
    plt.imshow(vx3,vmin = vx3min, vmax = vx3max, extent=[ex[0], ex[1], ex[2], ex[3]], cmap = 'inferno', aspect='auto', origin='lower')
    plt.title('Z-velocity')
    plt.colorbar()

    plt.suptitle(str(int(NX*F/2**max_level)) + 'x' + str(int(NY*F/2**max_level)) + 'x' + str(int(NZ*F/2**max_level)) + ' Snapshot ' + str(n) + ', time = ' + str(tim) + ', SMR = ' + str(max_level) + ', Coarsening Factor = ' + str(F))
    plt.tight_layout()
    plt.savefig(dir + 'KH_2D_snapshot_' + 'Y_' + str(ind) + str(n).zfill(5) + f'C{F}' + '.png', dpi = 300)
    plt.clf()
    plt.close()

    # Clear variables to free memory immediately
    del den, vx1, vx3, ps, prs, vx2, temp
    gc.collect()

def main():
    if MPI_DEF:
        comm = MPI.COMM_WORLD
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
        
    # Calculate global min/max values using memory-efficient approach
    global densmin, densmax, vx1min, vx1max, prsmin, prsmax, vx2min, vx2max, tempmin, tempmax, vx3min, vx3max
    
    dir1_ = path_to_files
    fname_ = dir1_ + 'KH.hydro_w.' + str(0).zfill(5) + '.bin'
    file_data_ = bin_convert.read_binary(fname_)    

    # Process one variable at a time to minimize memory usage
    den_ = analyse_bin.make_3D_array(file_data_, 'dens')
    densmin = 0.2*np.min(den_)
    densmax = 5*np.max(den_)
    del den_
    gc.collect()
    
    vx1_ = analyse_bin.make_3D_array(file_data_, 'velx')
    vx1min = 5*np.min(vx1_) 
    vx1max = 1.5*np.max(vx1_)
    del vx1_
    gc.collect()
    
    prs_ = (2./3.)*analyse_bin.make_3D_array(file_data_, 'eint')
    prsmin = 0
    prsmax = 1.6*np.max(prs_)
    
    # Calculate temperature using existing pressure array
    den_temp = analyse_bin.make_3D_array(file_data_, 'dens')
    temp_ = prs_*TEMPERATURE/den_temp
    tempmin = 0.2*np.min(temp_)
    tempmax = 5*np.max(temp_)
    del prs_, den_temp, temp_
    gc.collect()
    
    # Set velocity limits
    vx2min = -50
    vx2max = 15
    vx3min = -50
    vx3max = 50
 
    for i in range(N1_local, N2_local):
        print(f"Processing snapshot {i}...")

        den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim = Read2DfromnpzX(path_to_files, i, F)
        Make2D_snapshotsX(den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim, i, F)
        
        den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim = Read2DfromnpzY(path_to_files, i, F)
        Make2D_snapshotsY(den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim, i, F)

        den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim = Read2DfromnpzZ(path_to_files, i, F)
        Make2D_snapshotsZ(den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim, i, F)

    if MPI_DEF:
        if rank < nproc_extra: # Process extra files across ranks
            n = size * nfiles_local + rank + n1
            print(f"Processing extra snapshot {n}")

            den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim = Read2DfromnpzX(path_to_files, n, F)
            Make2D_snapshotsX(den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim, n, F)

            den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim = Read2DfromnpzY(path_to_files, n, F)
            Make2D_snapshotsY(den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim, n, F)

            den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim = Read2DfromnpzZ(path_to_files, n, F)
            Make2D_snapshotsZ(den, vx1, vx3, ps, prs, vx2, temp, ex, ind, tim, n, F)

if __name__ == "__main__":
    main()












