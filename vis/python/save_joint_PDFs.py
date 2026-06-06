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
import matplotlib.colors as mcolors
import gc
from mpi4py import MPI
from matplotlib.lines import Line2D
from scipy.interpolate import interp1d

from save_2D_arrays_3D import ATOMIC_MASS, LENGTH, TIME, MASS, VELOCITY, DENSITY, ENERGY, POWER, PRESSURE, TEMPERATURE, MU, N_UNIT, COOLING_UNIT, CHI, GAMMA

from save_2D_arrays_3D import CoarseByFactor

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

def get_contour_levels(hist, fractions=[0.50, 0.68, 0.95]):

    hist_flat = hist.flatten()
    hist_sorted = np.sort(hist_flat)  # ascending instead of descending
    cumsum = np.cumsum(hist_sorted)
    cumsum /= cumsum[-1]
    levels = []
    for f in fractions:
        idx = np.searchsorted(cumsum, f)
        idx = min(idx, len(hist_sorted) - 1)
        levels.append(hist_sorted[idx])
    return sorted(set(np.round(levels, 6)))

def MakeJointPDFs(tempavgspace, temp, tim, n, F, weight):  
    """
    Make joint PDF of  T vs <T>.
    """
    global binsizefact

    wt = np.ones_like(temp).flatten()
    W = 'V'

    # correcting the y-lims for the time-averaged temperature profile to account for the integrated TRML shift
    global dir, n1, n2, jump
    with np.load(dir + f"KH_1D_arrays_time_averaged{n1}to{n2}with{jump}.npz", 'r') as f:
        temp_vol_avg_timeavg = f['temp_vol_av'] 
    with np.load(dir + f"KH_1D_arrays_snapshot{n1}_{n2}_{str(0).zfill(5)}_C{F}_y_lims_corrected.npz", 'r') as f:
        v_TRML_integrated = f['v_TRML_integrated']
    with np.load(dir + 'KH_1D_arrays_snapshot_' + str(n).zfill(5) + f'C{F}.npz', 'r') as f:
        Y_lims = f['Y_lims']
        Y_lims_transformed = Y_lims + v_TRML_integrated
    # get actual pdf
    axis = 'none'
    slice = 1
    fname = dir + 'KH_tempPDF_snapshot_' + str(n).zfill(5) + f'C{F}_axis{axis}_slice{slice}.npz'
    with np.load(fname, 'r') as data:
        hist_vol = data['hist_vol']
        bin_centers = data['bin_centers']
        T = 10**bin_centers
        hist_vol = np.where((T >= 1.05e4) & (T <= 0.95e6), hist_vol, 0)
        hist_vol /= np.trapezoid(hist_vol, bin_centers)

    def interp(arr):
        return interp1d(Y_lims_transformed, temp_vol_avg_timeavg, kind="linear", fill_value="extrapolate")(Y_lims)
    
    temp_vol_avg_timeavg = interp(temp_vol_avg_timeavg)
    tempavgspacetime = np.broadcast_to(
    temp_vol_avg_timeavg[np.newaxis, :, np.newaxis],
    (NX, NY, NZ))
    
    logTemp_bins = np.linspace(np.log10(1.05e4), np.log10(0.95e6), int(binsizefact*45))
    #logTemp_bins = np.linspace(3.5, 6.5,int(binsizefact*70))

    plt.figure(figsize=(16, 9))
    plt.subplot(1, 2, 1)
    plt.gca().set_facecolor('black')
    hist_Ttimespace, xedges_Ttimespace, yedges_Ttimespace = np.histogram2d(np.log10(tempavgspacetime).flatten(), np.log10(temp).flatten(), bins=[logTemp_bins, logTemp_bins], weights=wt, density=True)
    hist_Ttimespace_unnormalized, _, _ = np.histogram2d(np.log10(tempavgspacetime).flatten(), np.log10(temp).flatten(), bins=[logTemp_bins, logTemp_bins], weights=wt, density=False)
    bin_centers_x = 0.5*(xedges_Ttimespace[1:] + xedges_Ttimespace[:-1])
    bin_centers_y = 0.5*(yedges_Ttimespace[1:] + yedges_Ttimespace[:-1])
    levels = np.logspace(-4, 1, 8)
    # list color themes from matplotlib
    arr = [1, 2, 3, 4, 5, 6, 7, 8]

    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=min(arr), vmax=max(arr))
    colors = [cmap(norm(val)) for val in arr]
    colors = colors[::-1]  # reverse the colors so that higher levels are darker

    print(f"Snapshot {n}: levels = {levels}, unique = {len(set(levels))}")
    hist_Ttimespace += 1e-11
    im = plt.imshow(hist_Ttimespace.T, extent=[xedges_Ttimespace[0], xedges_Ttimespace[-1], yedges_Ttimespace[0], yedges_Ttimespace[-1]], aspect='auto', origin='lower', cmap='inferno', norm='log', vmin=1e-5, vmax=1e2)

    if len(levels) >= 2:
        plt.contour(bin_centers_x, bin_centers_y, hist_Ttimespace.T, 
                levels=levels, 
                colors=colors,
                linewidths=1.5)
    plt.ylim(3.5, 6.5)
    plt.xlim(3.5, 6.5)
    plt.colorbar(im)
    plt.xlabel(r'$log_{10}(\langle T \rangle_t)$')
    plt.ylabel(r'$log_{10}(T)$')
    plt.grid(True, which="both", ls="--", alpha=0.7)

    legend_elements = [
        Line2D([0], [0], color=colors[0],  label=f'P = {levels[0]:.2e}'),
        Line2D([0], [0], color=colors[1],  label=f'P= {levels[1]:.2e}'),
        Line2D([0], [0], color=colors[2], label=f'P= {levels[2]:.2e}'),
        Line2D([0], [0], color=colors[3], label=f'P= {levels[3]:.2e}'),
        Line2D([0], [0], color=colors[4], label=f'P= {levels[4]:.2e}'),
        Line2D([0], [0], color=colors[5], label=f'P= {levels[5]:.2e}'),
        Line2D([0], [0], color=colors[6], label=f'P= {levels[6]:.2e}'),
        Line2D([0], [0], color=colors[7], label=f'P= {levels[7]:.2e}'),
    ]

    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # marginalized over y (sum along axis=1) → P(log10(<T>_t))
    P_x = np.sum(hist_Ttimespace * np.diff(yedges_Ttimespace)[np.newaxis, :], axis=1)
    T_x = 10**bin_centers_x
    P_x = np.where((T_x >= 1.05e4) & (T_x <= 0.95e6), P_x, 0)
    P_x /= np.sum(P_x * np.diff(xedges_Ttimespace))  # Normalize P_x

    # marginalized over x (sum along axis=0) → P(log10(T))  
    P_y = np.sum(hist_Ttimespace * np.diff(xedges_Ttimespace)[:, np.newaxis], axis=0)
    T_y = 10**bin_centers_y
    P_y = np.where((T_y >= 1.05e4) & (T_y <= 0.95e6), P_y, 0)
    P_y /= np.sum(P_y * np.diff(yedges_Ttimespace))  # Normalize P_y

    # pdf for <T> from simulation
    bins_temptimespace = np.linspace(3.5, 6.5, int(binsizefact*70))
    hist_tempvolav, bin_edges = np.histogram(np.log10(tempavgspacetime).flatten(), bins=bins_temptimespace, weights=wt, density=True)
    bin_centerstemptimespace = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    T_avg = 10**bin_centerstemptimespace
    hist_tempvolav = np.where((T_avg >= 1.05e4) & (T_avg <= 0.95e6), hist_tempvolav, 0)
    hist_tempvolav /= np.trapezoid(hist_tempvolav, bin_centerstemptimespace)

    plt.subplot(1, 2, 2)
    plt.gca().set_facecolor('black')
    plt.plot(bin_centers_x, P_x, color='cyan', label=r'$P_V(log_{10}(\langle T \rangle_t))$ from joint PDF')
    plt.plot(bin_centers_y, P_y, color='magenta', label=r'$P_V(log_{10}(T))$ from joint PDF')
    plt.plot(bin_centers, hist_vol, color='yellow', label=r'$P_V(log_{10}(T))$ from simulation') 
    plt.plot(bin_centerstemptimespace, hist_tempvolav, color='lime', label=r'$P_V(log_{10}(\langle T \rangle_t))$ from simulation')
    plt.yscale('log')
    plt.xlabel(r'$log_{10}(\langle T \rangle_t)$ and $log_{10}(T)$')
    plt.ylabel(r'$P_V$')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)
    plt.ylim(1e-3, 1e2)

    hist_Tspace_unnormalized, xedges_Tspace, yedges_Tspace = np.histogram2d(np.log10(tempavgspace).flatten(), np.log10(temp).flatten(), bins=[logTemp_bins, logTemp_bins], weights=wt, density=False)

    plt.suptitle(str(int(NX*F/2**max_level)) + 'x' + str(int(NY*F/2**max_level)) + 'x' + str(int(NZ*F/2**max_level)) + ' Snapshot ' + str(n) + ', time = ' + str(tim) + ', SMR = ' + str(max_level) + ', Coarsening Factor = ' + str(F))
    plt.tight_layout()
    plt.savefig(dir + 'KH_jointPDFtemp_snapshot_' + str(n).zfill(5) + f'C{F}' + '.png')
    plt.clf()
    plt.close()

    np.savez_compressed(dir + 'KH_jointPDFtemp_snapshot_' + str(n).zfill(5) + f'C{F}' + '.npz', hist_Ttimespace_unnormalized= hist_Ttimespace_unnormalized, hist_Tspace_unnormalized=hist_Tspace_unnormalized, xedges=xedges_Tspace, yedges=yedges_Tspace, levels=levels, colors = np.array(colors), vmin=1e-5, vmax=1e2)

def make_steady_joint_PDFs(ni, nf, F):
    """
    Sum the un normalized joint histograms for the snapshots from n1 to n2 with interval jump, 
    and then normalize the summed histogram to get the steady joint PDF.
    """
    global dir, binsizefact, n1, n2, jump
    hist_sum = None

    wt = np.ones((NX, NY, NZ)).flatten()
    with np.load(dir + f"KH_1D_arrays_time_averaged{ni}to{nf}with{jump}.npz", 'r') as f:
        temp_vol_avg_timeavg = f['temp_vol_av'] 
    tempavgspacetime = np.broadcast_to(
    temp_vol_avg_timeavg[np.newaxis, :, np.newaxis],
    (NX, NY, NZ))

    # get actual pdf
    axis = 'none'
    slice = 1
    filename = dir + f"KH_PDFs_time_averaged{ni}to{nf}with{jump}.npz"
    with np.load(filename, 'r') as f:
        hist_vol_av = f['hist_vol_av']
        hist_vol_sig = f['hist_vol_sig']
        bin_centers = f['bin_centers']
        T = 10**bin_centers

    for n in range(ni, nf+1, jump):
        print(f"Processing snapshot {n} for steady joint PDF...")
        fname = dir + 'KH_jointPDFtemp_snapshot_' + str(n).zfill(5) + f'C{F}' + '.npz'
        with np.load(fname, 'r') as data:
            hist_vol_ = data['hist_Ttimespace_unnormalized']
        if hist_sum is None:
            hist_sum = hist_vol_
        else:
            hist_sum += hist_vol_

    with np.load(dir + 'KH_jointPDFtemp_snapshot_' + str(ni).zfill(5) + f'C{F}' + '.npz', 'r') as data:
        xedges = data['xedges']
        yedges = data['yedges']
        levels = data['levels']
        colors = data['colors']
        colors = [tuple(color) for color in colors]

    hist_sum /= np.sum(hist_sum * np.diff(xedges)[:, np.newaxis] * np.diff(yedges)[np.newaxis, :])
    hist_sum += 1e-11

    plt.figure(figsize=(16, 9))
    plt.subplot(1, 2, 1)
    plt.gca().set_facecolor('black')

    im = plt.imshow(hist_sum.T, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', origin='lower', cmap='inferno', norm='log', vmin=1e-5, vmax=1e2)

    bin_centers_x = 0.5*(xedges[1:] + xedges[:-1])
    bin_centers_y = 0.5*(yedges[1:] + yedges[:-1])
    plt.contour(bin_centers_x, bin_centers_y, hist_sum.T, 
            levels=levels, 
            colors=colors,
            linewidths=1.5)
    plt.ylim(3.5, 6.5)
    plt.xlim(3.5, 6.5)
    plt.colorbar(im)
    plt.xlabel(r'$log_{10}(\langle T \rangle_t)$')
    plt.ylabel(r'$log_{10}(T)$')
    plt.grid(True, which="both", ls="--", alpha=0.7)

    legend_elements = [
        Line2D([0], [0], color=colors[0],  label=f'P = {levels[0]:.2e}'),
        Line2D([0], [0], color=colors[1],  label=f'P= {levels[1]:.2e}'),
        Line2D([0], [0], color=colors[2], label=f'P= {levels[2]:.2e}'),
        Line2D([0], [0], color=colors[3], label=f'P= {levels[3]:.2e}'),
        Line2D([0], [0], color=colors[4], label=f'P= {levels[4]:.2e}'),
        Line2D([0], [0], color=colors[5], label=f'P= {levels[5]:.2e}'),
        Line2D([0], [0], color=colors[6], label=f'P= {levels[6]:.2e}'),
        Line2D([0], [0], color=colors[7], label=f'P= {levels[7]:.2e}'),
    ]

    plt.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # marginalized over y (sum along axis=1) → P(log10(<T>_t))
    P_x = np.sum(hist_sum * np.diff(yedges)[np.newaxis, :], axis=1)
    T_x = 10**bin_centers_x
    P_x = np.where((T_x >= 1.05e4) & (T_x <= 0.95e6), P_x, 0)
    P_x /= np.sum(P_x * np.diff(xedges))  # Normalize P_x

    # marginalized over x (sum along axis=0) → P(log10(T))  
    P_y = np.sum(hist_sum * np.diff(xedges)[:, np.newaxis], axis=0)
    T_y = 10**bin_centers_y
    P_y = np.where((T_y >= 1.05e4) & (T_y <= 0.95e6), P_y, 0)
    P_y /= np.sum(P_y * np.diff(yedges))  # Normalize P_y

    # pdf for <T> from simulation
    bins_temptimespace = np.linspace(3.5, 6.5, int(binsizefact*70))
    hist_tempvolav, bin_edges = np.histogram(np.log10(tempavgspacetime).flatten(), bins=bins_temptimespace, weights=wt, density=True)
    bin_centerstemptimespace = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    T_avg = 10**bin_centerstemptimespace
    hist_tempvolav = np.where((T_avg >= 1.05e4) & (T_avg <= 0.95e6), hist_tempvolav, 0)
    hist_tempvolav /= np.trapezoid(hist_tempvolav, bin_centerstemptimespace)

    plt.subplot(1, 2, 2)
    plt.gca().set_facecolor('black')
    plt.plot(bin_centers_x, P_x, color='cyan', label=r'$P_V(log_{10}(\langle T \rangle_t))$ from joint PDF')
    plt.plot(bin_centers_y, P_y, color='magenta', label=r'$P_V(log_{10}(T))$ from joint PDF')
    plt.plot(bin_centers, hist_vol_av, color='yellow', label=r'$P_V(log_{10}(T))$ from simulation') 
    plt.fill_between(bin_centers, hist_vol_av - hist_vol_sig, hist_vol_av + hist_vol_sig, color='yellow', alpha=0.3)
    plt.plot(bin_centerstemptimespace, hist_tempvolav, color='lime', label=r'$P_V(log_{10}(\langle T \rangle_t))$ from simulation')
    plt.yscale('log')
    plt.xlabel(r'$log_{10}(\langle T \rangle_t)$ and $log_{10}(T)$')
    plt.ylabel(r'$P_V$')
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.legend(loc='upper left', fontsize=12)
    plt.ylim(1e-3, 1e2)

    plt.suptitle(str(int(NX*F/2**max_level)) + 'x' + str(int(NY*F/2**max_level)) + 'x' + str(int(NZ*F/2**max_level)) + ', SMR = ' + str(max_level) + ', Coarsening Factor = ' + str(F))
    plt.tight_layout()
    plt.savefig(dir + 'KH_jointPDFtemp_snapshot_' + f'{ni}to{nf}' + f'C{F}' + '.png')
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

    global n1, n2, jump
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
        
    global temp_vol_avg_timeavg, binsizefact

    binsizefact = NY/1024.

    for i in range(N1_local, N2_local):
        print(f"Processing snapshot {i}...")
        den, temp_avgspace, temp, t = ReadBinFile_temp(path_to_files, i, F)
        temp_spaceavg3D = np.broadcast_to(
        temp_avgspace[np.newaxis, :, np.newaxis],
        (NX, NY, NZ))
        MakeJointPDFs(temp_spaceavg3D, temp, t, i, F, weight='V')

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
        MakeJointPDFs(temp_spaceavg3D, temp, t, n, F, weight='V')

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
        if n1 == 0 and n2 == 125 and jump == 1:
            make_steady_joint_PDFs(36, n2, F)
      
if __name__ == "__main__":
    main()
