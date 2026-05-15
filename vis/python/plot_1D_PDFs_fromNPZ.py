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

def ReadPDFfromnpz(path_to_files, n, F):
    """
    Read the PDF data from a .npz file.
    """
    axis = 'none'
    slice = 1
    dir = path_to_files
    fname = dir + 'KH_tempPDF_snapshot_' + str(n).zfill(5) + f'C{F}_axis{axis}_slice{slice}.npz'

    with np.load(fname, 'r') as data:
        hist_vol = data['hist_vol']
        hist_mass = data['hist_mass']
        hist_emissivity = data['hist_emissivity']
        bin_centers = data['bin_centers']
        time = data['time']

    return hist_vol, hist_mass, hist_emissivity, bin_centers, time

def Read1Dfromnpz(path_to_files, n, F):
    """
    Read the 1D data from a .npz file.
    """
    dir = path_to_files
    fname = dir + 'KH_1D_arrays_snapshot_' + str(n).zfill(5) + f'C{F}.npz'

    with np.load(fname, 'r') as data:
        Y_lims = data['Y_lims']
        den_vol = data['den_vol']
        den_sig = data['den_sig_vol']
        vx1_vol = data['vx1_vol']
        vx1_sig = data['vx1_sig_vol']
        vx3_vol = data['vx3_vol']
        vx3_sig = data['vx3_sig_vol']
        ps_vol = data['ps_vol']
        ps_sig = data['ps_sig_vol']
        prs_vol = data['prs_vol']
        prs_sig = data['prs_sig_vol']
        vx2_vol = data['vx2_vol']
        vx2_sig = data['vx2_sig_vol']
        temp_vol = data['temp_vol']
        temp_sig = data['temp_sig_vol']
        time = data['time']

    return Y_lims, den_vol, den_sig, vx1_vol, vx1_sig, vx3_vol, vx3_sig, ps_vol, ps_sig, prs_vol, prs_sig, vx2_vol, vx2_sig, temp_vol, temp_sig, time

def ReadFluxesfromnpz(path_to_files, n, F):

    """
    rho_avXv2_av = rho_avXv2_av, rhov2_avXv1_av = rhov2_avXv1_av, rhov2_avXv2_av = rhov2_avXv2_av,
                        delrho_delv2_av = delrho_delv2_av, rhov2_av = rhov2_av, R_xz = R_xz, R_zz = R_zz,
                        p_av = p_av, Be_av_rhov2_av = Be_av_rhov2_av - Be_av_rhov2_av[0],
                        del_Be_del_rhov2_av = del_Be_del_rhov2_av - del_Be_del_rhov2_av[0],
                        edot_cool_cum_dx2 = edot_cool_cum_dx2, rhov2v1_av = rhov2v1_av,
                        P_rhovx2sqr = P_rhovx2sqr, net_heating = net_heating, number = n, time = tim, factor = F
    """

    dir = path_to_files
    fname = dir + 'KH_fluxes_' + str(n).zfill(5) + f'_C{F}.npz'

    with np.load(fname, 'r') as data:
        rho_avXv2_av = data['rho_avXv2_av']
        rhov2_avXv1_av = data['rhov2_avXv1_av']
        rhov2_avXv2_av = data['rhov2_avXv2_av']
        delrho_delv2_av = data['delrho_delv2_av']
        rhov2_av = data['rhov2_av']
        R_xz = data['R_xz']
        R_zz = data['R_zz']
        p_av = data['p_av']
        Be_av_rhov2_av = data['Be_av_rhov2_av']
        del_Be_del_rhov2_av = data['del_Be_del_rhov2_av']
        edot_cool_cum_dx2 = data['edot_cool_cum_dx2']
        rhov2v1_av = data['rhov2v1_av']
        P_rhovx2sqr = data['P_rhovx2sqr']
        net_heating = data['net_heating']
        number = data['number']
        time = data['time']
        factor = data['factor']

    return (rho_avXv2_av, rhov2_avXv1_av, rhov2_avXv2_av, delrho_delv2_av, rhov2_av, R_xz, R_zz,
            p_av, Be_av_rhov2_av, del_Be_del_rhov2_av, edot_cool_cum_dx2, rhov2v1_av,
            P_rhovx2sqr, net_heating, number, time, factor)

def Make1D_snapshots(Y_lims, den_vol, den_sig, vx1_vol, vx1_sig, vx3_vol, vx3_sig, ps_vol, ps_sig, prs_vol, prs_sig, vx2_vol, vx2_sig, temp_vol, temp_sig, tim, n, F):

    plt.figure(figsize=(16, 9))

    plt.subplot(3, 3, 1)
    plt.plot(Y_lims, den_vol, label = 'Volume weighted density', color = 'blue')
    plt.fill_between(Y_lims, den_vol - den_sig, den_vol + den_sig, alpha=0.5)
    plt.yscale('log')
    plt.ylim(densmin, densmax)
    plt.title('Mean density [vlm-wtd]', fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.subplot(3, 3, 2)
    plt.plot(Y_lims, ps_vol, label = 'Volume weighted passive scalar', color = 'blue')
    plt.fill_between(Y_lims, ps_vol - ps_sig, ps_vol + ps_sig, alpha=0.5)
    plt.ylim(-0.1, 1.1)
    plt.title('Passive Scalar [vlm-wtd]', fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.subplot(3, 3, 3)
    plt.plot(Y_lims, temp_vol, label = 'Volume weighted temperature', color = 'blue')
    plt.ylim(tempmin, tempmax)
    plt.fill_between(Y_lims, temp_vol - temp_sig, temp_vol + temp_sig, alpha=0.5)
    plt.yscale('log')
    plt.title('Temperature [vlm-wtd]', fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.subplot(3, 3, 4)
    plt.plot(Y_lims, prs_vol, label = 'Volume weighted pressure', color = 'blue')
    plt.fill_between(Y_lims, prs_vol - prs_sig, prs_vol + prs_sig, alpha=0.5)
    plt.ylim(prsmin, prsmax)
    plt.title('Pressure [vlm-wtd]', fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.subplot(3, 3, 5)
    plt.plot(Y_lims, vx1_vol, label = 'Volume weighted velocity', color = 'blue')
    plt.fill_between(Y_lims, vx1_vol - vx1_sig, vx1_vol + vx1_sig, alpha=0.5)
    plt.ylim(vx1min, vx1max)
    plt.title('X-velocity [vlm-wtd]', fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)

    plt.subplot(3, 3, 6)
    plt.plot(Y_lims, vx2_vol, label = 'Volume weighted y-velocity', color = 'blue', alpha=0.7)
    plt.fill_between(Y_lims, vx2_vol - vx2_sig, vx2_vol + vx2_sig, alpha=0.5)
    plt.ylim(vx2min, vx2max)
    plt.title('Y-velocity [vlm-wtd]', fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7)  

    plt.subplot(3, 3, 8)
    plt.plot(Y_lims, vx3_vol, label = 'Volume weighted z-velocity', color = 'blue', alpha=0.7)
    plt.fill_between(Y_lims, vx3_vol - vx3_sig, vx3_vol + vx3_sig, alpha=0.5)
    plt.ylim(vx3min, vx3max)
    plt.title('Z-velocity [vlm-wtd]', fontsize=10)
    plt.grid(True, which="both", ls="--", alpha=0.7) 

    plt.suptitle('X-Z plane averages ' + str(int(NX*F/2**max_level)) + 'x' + str(int(NY*F/2**max_level)) + 'x' + str(int(NZ*F/2**max_level)) + ' Snapshot ' + str(n) + ', time = ' + str(tim) + ', SMR = ' + str(max_level) + ', Coarsening Factor = ' + str(F))
    plt.tight_layout()
    plt.savefig(dir + 'KH_1Dhz_snapshot_' + str(n).zfill(5)+ f'C{F}' + '.png', dpi = 300)
    plt.clf()
    plt.close()

def MakePDF_snapshots(hist_vol, hist_mass, hist_emissivity, bin_centers, tim, n, F):   
    
    plt.figure(figsize=(16, 9))
    plt.plot(bin_centers, hist_vol, label='Volume weighted', color='blue', linestyle='-', linewidth=2)
    plt.plot(bin_centers, hist_mass, label='Mass weighted', color='orange', linestyle='-', linewidth=2)
    plt.plot(bin_centers, hist_emissivity, label='Emissivity weighted', color='green', linestyle='-', linewidth=2)
    plt.yscale('log')
    plt.ylim(1e-5, 1e2)
    plt.xlim(3.5, 6.5)
    plt.grid()
    plt.legend(loc='upper right')

    plt.suptitle(str(int(NX*F/2**max_level)) + 'x' + str(int(NY*F/2**max_level)) + 'x' + str(int(NZ*F/2**max_level)) + ' Snapshot ' + str(n) + ', time = ' + str(tim) + ', SMR = ' + str(max_level) + ', Coarsening Factor = ' + str(F))
    plt.tight_layout()
    plt.savefig(dir + 'KH_tempPDF_snapshot_' + str(n).zfill(5) + f'C{F}' + '.png')
    plt.clf()
    plt.close()   
            
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

    plt.figure(figsize=(16, 9))
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
    plt.close()  

def MakeFluxes_snapshots(Y, rho_avXv2_av, rhov2_avXv1_av, rhov2_avXv2_av, delrho_delv2_av, rhov2_av, R_xz, R_zz, p_av, Be_av_rhov2_av, del_Be_del_rhov2_av, edot_cool_cum_dx2, rhov2v1_av, P_rhovx2sqr, net_heating, number, time, factor):

    plt.figure(figsize=(16, 9))

    plt.subplot(2, 2, 1)
    plt.plot(Y, rho_avXv2_av, label=r'$\langle \rho \rangle \langle u_{z} \rangle$')
    plt.plot(Y, delrho_delv2_av, label=r'$\langle \delta \rho \delta u_{z} \rangle$')
    plt.plot(Y, rhov2_av, label=r'$\langle \rho u_{z} \rangle$')
    plt.title('Mass Flux')
    plt.xlabel('x2')
    plt.xlim(Y[0], Y[-1])
    plt.ylim(-flux1lim, flux1lim)
    plt.grid()
    plt.legend(loc = 'upper right', fontsize=4)

    plt.subplot(2, 2, 2)
    plt.plot(Y, rhov2_avXv1_av, label=r'$\langle \rho u_{z} \rangle \langle u_{x} \rangle$')
    plt.plot(Y, R_xz, label=r'$\langle \delta u_{x} \delta(\rho  u_{z}) \rangle$')
    plt.plot(Y, rhov2v1_av, label=r'$\langle \rho u_{x} u_{z} \rangle$')
    plt.title('x1-momentum Flux')
    plt.xlabel('x2')
    plt.xlim(Y[0], Y[-1])
    plt.ylim(-flux2lim, flux2lim)
    plt.grid()
    plt.legend(loc = 'upper right', fontsize=4)

    plt.subplot(2, 2, 3)
    plt.plot(Y, p_av, label=r'$\langle p \rangle$')
    plt.plot(Y, rhov2_avXv2_av, label=r'$\langle \rho v_{x2} \rangle \langle u_{z} \rangle$')
    plt.plot(Y, R_zz, label=r'$\langle \delta u_{z} \delta(\rho  u_{z}) \rangle$')
    plt.plot(Y, p_av+rhov2_avXv2_av+R_zz, label=r'$\langle p  + \rho u_{z}^2 \rangle$')
    plt.title('x2-momentum Flux')
    plt.xlabel('x2')
    plt.xlim(Y[0], Y[-1])
    plt.ylim(-flux3lim, flux3lim)
    plt.grid()
    plt.legend(loc = 'upper right', fontsize=4)

    plt.subplot(2, 2, 4)
    plt.plot(Y, Be_av_rhov2_av, label=r'$\left [ \langle B_e \rangle \langle \rho  u_{z} \rangle \right ]$')
    plt.plot(Y, del_Be_del_rhov2_av, label=r'$\left [ \langle \delta B_e \delta(\rho  u_{z}) \rangle \right ]$')
    plt.plot(Y, edot_cool_cum_dx2, label=r'$\int \langle n^2 \Lambda(T) \rangle dz$')
    plt.plot(Y, net_heating, label='(net heating)/area')
    plt.title('Energy Flux')
    plt.legend(loc = 'upper right', fontsize=4)
    plt.grid()
    plt.tight_layout()
    plt.xlabel('x2')
    plt.xlim(Y[0], Y[-1])
    #plt.ylim([-10,10])
    plt.ylim(-flux4lim, flux4lim) 
    
    plt.suptitle('Horizontally Averaged Fluxes')
    plt.tight_layout()
    plt.suptitle(str(NX*factor//2**max_level) + 'x' + str(NY*factor//2**max_level) + 'x' + str(NZ*factor//2**max_level) + f'Snapshot{number}' ' SMR = ' + str(max_level) + ', Coarsening Factor = ' + str(factor))
    plt.savefig(f"{dir}KH_mean_fluxes_{str(number).zfill(5)}" + f'C{factor}' + ".png", dpi = 300)
    plt.clf()
    plt.close()

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
        
    global densmin, densmax, vx1min, vx1max, prsmin, prsmax, vx2min, vx2max, tempmin, tempmax, vx3min, vx3max

    rho_avXv2_av, rhov2_avXv1_av, rhov2_avXv2_av, delrho_delv2_av, rhov2_av, R_xz, R_zz, p_av, Be_av_rhov2_av, del_Be_del_rhov2_av, edot_cool_cum_dx2, rhov2v1_av, P_rhovx2sqr, net_heating, number, time, factor = ReadFluxesfromnpz(path_to_files, n2, F)
    global flux1lim, flux2lim, flux3lim, flux4lim
    flux1lim = 1.1*np.max(np.abs(rho_avXv2_av))
    flux2lim = 1.1*np.max(np.abs(rhov2v1_av))
    flux3lim = 1.1*np.max(np.abs(p_av))
    flux4lim = 1.1*np.max(np.abs(Be_av_rhov2_av))

    dir1_ = path_to_files
    fname_ = dir1_ + 'KH.hydro_w.' + str(0).zfill(5) + '.bin'
    file_data_ = bin_convert.read_binary(fname_)    

    den_ = analyse_bin.make_3D_array(file_data_, 'dens')
    vx1_ = analyse_bin.make_3D_array(file_data_, 'velx')
    prs_ = analyse_bin.make_3D_array(file_data_, 'eint')
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

    del den_, vx1_, prs_, temp_
    gc.collect()
 
    for i in range(N1_local, N2_local):
        print(f"Processing snapshot {i}...")
        Y_lims, den_vol, den_sig, vx1_vol, vx1_sig, vx3_vol, vx3_sig, ps_vol, ps_sig, prs_vol, prs_sig, vx2_vol, vx2_sig, temp_vol, temp_sig, t = Read1Dfromnpz(path_to_files, i, F)
        histogram_vol, histogram_mass, histogram_emissivity, bin_centers, time = ReadPDFfromnpz(path_to_files, i, F)
        MakePDF_snapshots(histogram_vol, histogram_mass, histogram_emissivity, bin_centers, time, i, F)
        Make1D_snapshots(Y_lims, den_vol, den_sig, vx1_vol, vx1_sig, vx3_vol, vx3_sig, ps_vol, ps_sig, prs_vol, prs_sig, vx2_vol, vx2_sig, temp_vol, temp_sig, t, i, F)
        rho_avXv2_av, rhov2_avXv1_av, rhov2_avXv2_av, delrho_delv2_av, rhov2_av, R_xz, R_zz, p_av, Be_av_rhov2_av, del_Be_del_rhov2_av, edot_cool_cum_dx2, rhov2v1_av, P_rhovx2sqr, net_heating, number, time, factor = ReadFluxesfromnpz(path_to_files, i, F)
        MakeFluxes_snapshots(Y_lims, rho_avXv2_av, rhov2_avXv1_av, rhov2_avXv2_av, delrho_delv2_av, rhov2_av, R_xz, R_zz, p_av, Be_av_rhov2_av, del_Be_del_rhov2_av, edot_cool_cum_dx2, rhov2v1_av, P_rhovx2sqr, net_heating, i, time, F)
        #MakeJointPDFs(dens, velx, velz, ps, prs, vely, temp, t, i, F, 'v')

    if MPI_DEF:
        if rank < nproc_extra: # Process extra files across ranks
            n = size * nfiles_local + rank + n1
            print(f"Processing extra snapshot {n}")
            Y_lims, den_vol, den_sig, vx1_vol, vx1_sig, vx3_vol, vx3_sig, ps_vol, ps_sig, prs_vol, prs_sig, vx2_vol, vx2_sig, temp_vol, temp_sig, t = Read1Dfromnpz(path_to_files, n, F)
            histogram_vol, histogram_mass, histogram_emissivity, bin_centers, time = ReadPDFfromnpz(path_to_files, n, F)
            MakePDF_snapshots(histogram_vol, histogram_mass, histogram_emissivity, bin_centers, time, n, F)
            Make1D_snapshots(Y_lims, den_vol, den_sig, vx1_vol, vx1_sig, vx3_vol, vx3_sig, ps_vol, ps_sig, prs_vol, prs_sig, vx2_vol, vx2_sig, temp_vol, temp_sig, t, n, F)
            rho_avXv2_av, rhov2_avXv1_av, rhov2_avXv2_av, delrho_delv2_av, rhov2_av, R_xz, R_zz, p_av, Be_av_rhov2_av, del_Be_del_rhov2_av, edot_cool_cum_dx2, rhov2v1_av, P_rhovx2sqr, net_heating, number, time, factor = ReadFluxesfromnpz(path_to_files, n, F)
            MakeFluxes_snapshots(Y_lims, rho_avXv2_av, rhov2_avXv1_av, rhov2_avXv2_av, delrho_delv2_av, rhov2_av, R_xz, R_zz, p_av, Be_av_rhov2_av, del_Be_del_rhov2_av, edot_cool_cum_dx2, rhov2v1_av, P_rhovx2sqr, net_heating, n, time, F)
            #MakeJointPDFs(dens, velx, velz, ps, prs, vely, temp, t, i, F, 'v')
      
if __name__ == "__main__":
    main()












