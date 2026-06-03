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
from mpi4py import MPI

from save_2D_arrays_3D import ATOMIC_MASS, LENGTH, TIME, MASS, VELOCITY, DENSITY, ENERGY, POWER, PRESSURE, TEMPERATURE, MU, N_UNIT, COOLING_UNIT, CHI, GAMMA

from save_2D_arrays_3D import ISMCoolFn, CoarseByFactor

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

def ReadBinFile_temp(path_to_files, n, F):

    dir = path_to_files
    fname = dir + 'KH.hydro_w.' + str(n).zfill(5) + '.bin'
    file_data = bin_convert.read_binary(fname)    

    # Read and coarsen data one array at a time to minimize memory usage
    den = CoarseByFactor(analyse_bin.make_3D_array(file_data, 'dens'), F)
    
    # Calculate pressure directly from energy
    eint_array = analyse_bin.make_3D_array(file_data, 'eint')
    prs = CoarseByFactor((2./3.) * eint_array, F)
    del eint_array  # Free memory immediately
    gc.collect()

    temp = TEMPERATURE * prs/den
    
    # Don't pre-calculate temperature here - calculate it when needed
    times = file_data['time']

    with np.load(dir + 'KH_1D_arrays_snapshot_' + str(n).zfill(5) + f'C{F}.npz', 'r') as f:
        temp_volavg = f['temp_vol']
    
    # Clear file_data from memory
    del file_data
    gc.collect()
    
    return den, temp_volavg, temp, times

def MakeJointPDFs(tempavgspace, tempavgspacetime, temp, tim, n, F, weight):  
    """
    Make joint PDF of  T vs <T>.
    """

    wt = np.ones_like(temp).flatten()
    W = 'V'
    
    logTemp_bins = np.linspace(3.5, 6.5, 150)

    plt.figure(figsize=(16, 9))
    plt.subplot(1, 2, 1)

    hist, xedges, yedges = np.histogram2d(np.log10(tempavgspacetime).flatten(), np.log10(temp).flatten(), bins=[logTemp_bins, logTemp_bins], weights=wt, density=True)
    bin_centers_x = 0.5*(xedges[1:] + xedges[:-1])
    bin_centers_y = 0.5*(yedges[1:] + yedges[:-1])
    hist += 1e-10
    plt.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', origin='lower', cmap='inferno', norm='log')
    plt.ylim(3.5, 6.5)
    plt.xlim(3.5, 6.5)
    plt.colorbar()
    plt.xlabel(r'$log_{10}(\langle T \rangle_t)$')
    plt.ylabel(r'$log_{10}(T)$')
    plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.subplot(1, 2, 2)
    hist, xedges, yedges = np.histogram2d(np.log10(tempavgspace).flatten(), np.log10(temp).flatten(), bins=[logTemp_bins, logTemp_bins], weights=wt, density=True)
    bin_centers_x = 0.5*(xedges[1:] + xedges[:-1])
    bin_centers_y = 0.5*(yedges[1:] + yedges[:-1])
    hist += 1e-10
    plt.imshow(hist.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', origin='lower', cmap='inferno', norm='log')
    plt.ylim(3.5, 6.5)
    plt.xlim(3.5, 6.5)
    plt.colorbar()
    plt.xlabel(r'$log_{10}(\langle T \rangle)$')
    plt.ylabel(r'$log_{10}(T)$')
    plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.suptitle(str(int(NX*F/2**max_level)) + 'x' + str(int(NY*F/2**max_level)) + 'x' + str(int(NZ*F/2**max_level)) + ' Snapshot ' + str(n) + ', time = ' + str(tim) + ', SMR = ' + str(max_level) + ', Coarsening Factor = ' + str(F))
    plt.tight_layout()
    plt.savefig(dir + 'KH_jointPDFtemp_snapshot_' + str(n).zfill(5) + f'C{F}' + '.png')
    plt.clf()
    plt.close()

def main():
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
        parser.add_argument('-jump' , type=int, default=1, help="Interval for snapshots to process")
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
    jump = arg_dict['jump']
    SetGlobals(path_to_files, F)

    nfiles_local = (n2 - n1 + 1) // size
    nproc_extra = (n2 - n1 + 1) % size
    N1_local = rank * nfiles_local + n1
    N2_local = (rank + 1) * nfiles_local + n1
        
    global temp_vol_avg_timeavg

    with np.load(path_to_files + f"KH_1D_arrays_time_averaged{n1}to{n2}with{jump}.npz", 'r') as f:
        temp_vol_avg_timeavg = f['temp_vol_av'] 

    temp_vol_avg_timeavg3D = np.broadcast_to(
    temp_vol_avg_timeavg[np.newaxis, :, np.newaxis],
    (NX, NY, NZ))

    for i in range(N1_local, N2_local):
        print(f"Processing snapshot {i}...")
        den, temp_avgspace, temp, t = ReadBinFile_temp(path_to_files, i, F)
        temp_spaceavg3D = np.broadcast_to(
        temp_avgspace[np.newaxis, :, np.newaxis],
        (NX, NY, NZ))
        MakeJointPDFs(temp_spaceavg3D, temp_vol_avg_timeavg3D, temp, t, i, F, weight='V')

        # Explicitly delete variables to free memory
        del temp
        gc.collect()

    if rank < nproc_extra: # Process extra files across ranks
        n = size * nfiles_local + rank + n1
        print(f"Processing extra snapshot {n}")
        den, temp_avgspace, temp, t = ReadBinFile_temp(path_to_files, n, F)
        temp_spaceavg3D = np.broadcast_to(
        temp_avgspace[np.newaxis, :, np.newaxis],
        (NX, NY, NZ))
        MakeJointPDFs(temp_spaceavg3D, temp_vol_avg_timeavg3D, temp, t, n, F, weight='V')

        # Clean up memory
        del temp
        gc.collect()

    comm.Barrier()  # Wait for all ranks to finish writing images

    if rank == 0:
        import subprocess
        subprocess.run([
            'ffmpeg', '-y',
            '-framerate', '10',
            '-i', path_to_files + 'KH_jointPDFtemp_snapshot_%05dC' + str(F) + '.png',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            path_to_files + f'KH_jointPDFtemp_movie_C{F}.mp4'
        ], check=True)
        print("Movie saved.")
      
if __name__ == "__main__":
    main()