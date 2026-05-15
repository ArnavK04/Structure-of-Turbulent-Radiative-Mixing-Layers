
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

def SetGlobals(path_to_files, F, N1, N2, batch_size):

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
    """
    if factor <= 1:
        return array

    # Calculate the new shape - ensure it's divisible by factor
    shape = array.shape
    new_shape = (shape[0] // factor, shape[1] // factor, shape[2] // factor)
    
    # Trim array to make it divisible by factor
    trimmed_shape = (new_shape[0] * factor, new_shape[1] * factor, new_shape[2] * factor)
    trimmed_array = array[:trimmed_shape[0], :trimmed_shape[1], :trimmed_shape[2]]
    
    # Reshape and take mean - much faster than nested loops
    reshaped = trimmed_array.reshape(new_shape[0], factor, new_shape[1], factor, new_shape[2], factor)
    coarse_array = reshaped.mean(axis=(1, 3, 5))

    return coarse_array   

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
    
    vx2 = CoarseByFactor(analyse_bin.make_3D_array(file_data, 'vely'), F)

    temp = TEMPERATURE * prs / den
    
    # Don't pre-calculate temperature here - calculate it when needed
    times = file_data['time']
    
    # Clear file_data from memory
    del file_data
    gc.collect()
    
    return den, vx1, vx3, ps, prs, vx2, temp, times

def hor_avg(array):
    """Calculates the horizontal average of a 2D array."""
    return np.mean(array, axis=(0,2))

def hor_std(array):
    """Calculates the horizontal standard deviation of a 2D array."""
    return np.std(array, axis=(0,2))

def compute_flux_profiles(den, vx1, vx3, ps, prs, vx2, temp, n, tim, F, v1_TRML, v2_TRML, v3_TRML):
    
    rhov2_av = hor_avg(den * (vx2 - v2_TRML)) #horizontal average of mass flux
    del_rhov2 = den * (vx2 - v2_TRML) - rhov2_av[np.newaxis,:,np.newaxis] #fluctuations in mass flux
    rho_av = hor_avg(den) #horizontal average of density
    del_rho = den - rho_av[np.newaxis, : ,np.newaxis] #fluctuations in density
    v2_av = hor_avg(vx2 - v2_TRML) #horizontal average of velocity
    del_v2 = vx2 - v2_TRML - v2_av[np.newaxis, :, np.newaxis] #fluctuations in velocity
    delrho_delv2_av = hor_avg(del_rho * del_v2) #horizontal average of density-velocity correlation
    v1_av = hor_avg(vx1-v1_TRML) #horizontal average of velocity-x1
    del_v1 = vx1 - v1_TRML - v1_av[np.newaxis,:, np.newaxis] #fluctuations in velocity-x1
    rhov2v1_av = hor_avg(den * (vx2 - v2_TRML) * (vx1-v1_TRML)) #product of horizontal averages of mass flux and velocity-x1
    R_xz = hor_avg(del_v1 * del_rhov2) #xz stress
    R_zz = hor_avg(del_v2 * del_rhov2) #zz stress
    p_av = hor_avg(prs) #horizontal average of pressure
    P_rhovx2sqr = p_av + hor_avg(den * (vx2 - v2_TRML)**2 ) #pressure + 1/2 * velocity^2
    Be = 0.5 * ((vx1-v1_TRML)**2 + (vx2 - v2_TRML)**2 + (vx3 - v3_TRML)**2) + (GAMMA / (GAMMA - 1)) * prs / den #Bernoulli parameter
    Be_av = hor_avg(Be) #horizontal average of Bernoulli parameter
    del_Be = Be - Be_av[np.newaxis, :, np.newaxis] #fluctuations in Bernoulli parameter
    Be_av_rhov2_av = Be_av * rhov2_av #product of horizontal averages
    del_Be_del_rhov2_av = hor_avg(del_Be * del_rhov2) #horizontal average of Bernoulli-density correlation
    edot_cool_avg = hor_avg(den**2 * np.vectorize(ISMCoolFn, otypes='d')(temp))/COOLING_UNIT  #average cooling rate in code units
    rho_avXv2_av = rho_av * v2_av  #product of horizontal averages of density and velocity
    rhov2_avXv1_av = rhov2_av * v1_av  #product of horizontal averages of mass flux and velocity-x1
    rhov2_avXv2_av = rhov2_av * v2_av  #product of horizontal averages of mass flux and velocity-x2

    edot_cool_cum_dx2 = np.cumsum(edot_cool_avg)*DY #cumulative cooling rate

    net_heating = edot_cool_cum_dx2 +  Be_av_rhov2_av - Be_av_rhov2_av[0] +  del_Be_del_rhov2_av - del_Be_del_rhov2_av[0]  #net heating rate

    np.savez_compressed(f"{dir}KH_fluxes_{str(n).zfill(5)}_C{F}.npz",
                        rho_avXv2_av = rho_avXv2_av, rhov2_avXv1_av = rhov2_avXv1_av, rhov2_avXv2_av = rhov2_avXv2_av,
                        delrho_delv2_av = delrho_delv2_av, rhov2_av = rhov2_av, R_xz = R_xz, R_zz = R_zz,
                        p_av = p_av, Be_av_rhov2_av = Be_av_rhov2_av - Be_av_rhov2_av[0],
                        del_Be_del_rhov2_av = del_Be_del_rhov2_av - del_Be_del_rhov2_av[0],
                        edot_cool_cum_dx2 = edot_cool_cum_dx2, rhov2v1_av = rhov2v1_av,
                        P_rhovx2sqr = P_rhovx2sqr, net_heating = net_heating, number = n, time = tim, factor = F, v_TRML = np.array([v1_TRML, v2_TRML, v3_TRML]))

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

def MakeFluxProfiles(F, N, batch_size = 25, Flux_dict = None, Flux_dict_sqred = None):

    r"""
    Make flux profiles and save them to a file.
    Y = np.linspace(YMIN, YMAX, NY)

    for i in range (0, len(Flux_dict)):

        if (i<len(Flux_dict)-1):
            c = batch_size
        elif (N% batch_size == 0):
            c = batch_size
        else:
            c = N % batch_size
        
        Avg_sqred = tuple((a/c)**2 for a in Flux_dict[i])  # Square of the averages
        sqred_avg = tuple((a/c) for a in Flux_dict_sqred[i])  # Average of the squares
        Variances = tuple(sqred_avg[j] - Avg_sqred[j] for j in range(len(Avg_sqred)))  # Variance calculation
        std =  tuple(np.sqrt(v) for v in Variances)  # Standard deviation

        Flux_dict[i] = tuple((a/c) for a in Flux_dict[i])  

        rho_avXv2_av, rhov2_avXv1_av, rhov2_avXv2_av, delrho_delv2_av, rhov2_av, R_xz, R_zz, p_av, Be_av_rhov2_av, del_Be_del_rhov2_av,edot_cool_cum_dx2, rhov2v1_av, P_rhov2_sqr, net_heating = Flux_dict[i]
        plt.subplot(2,2,1)    
        plt.plot(Y, rho_avXv2_av, label=r'$\langle \rho \rangle \langle v_{x2} \rangle$')
        plt.fill_between(Y, rho_avXv2_av - std[0], rho_avXv2_av + std[0], alpha=0.3)
        plt.plot(Y, delrho_delv2_av, label=r'$\langle \delta \rho \delta v_{x2} \rangle$')
        plt.fill_between(Y, delrho_delv2_av - std[3], delrho_delv2_av + std[3], alpha=0.3)
        plt.plot(Y, rhov2_av, label=r'$\langle \rho v_{x2} \rangle$')
        plt.fill_between(Y, rhov2_av - std[4], rhov2_av + std[4], alpha=0.3)
        plt.title('Mass Flux')
        plt.xlabel('x2')
        plt.xlim(YMIN, YMAX)
        #plt.ylim(-0.5,0.5)
        plt.grid()
        plt.legend(loc = 'upper right', fontsize=4)
        
        plt.subplot(2,2,2)
        plt.plot(Y, rhov2_avXv1_av, label=r'$\langle \rho v_{x2} \rangle \langle v_{x1} \rangle$')
        plt.fill_between(Y, rhov2_avXv1_av - std[1], rhov2_avXv1_av + std[1], alpha=0.3)
        plt.plot(Y, R_xz, label=r'$\langle \delta v_{x1} \delta(\rho  v_{x2}) \rangle$')
        plt.fill_between(Y, R_xz - std[5], R_xz + std[5], alpha=0.3)
        plt.plot(Y, rhov2v1_av, label=r'$\langle \rho v_{x1} v_{x2} \rangle$')
        plt.fill_between(Y, rhov2v1_av - std[11], rhov2v1_av + std[11], alpha=0.3)
        plt.title('x1-momentum Flux')
        plt.xlabel('x2')
        plt.xlim(YMIN, YMAX)
        #plt.ylim(-2,2)
        plt.grid()
        plt.legend(loc = 'upper right', fontsize=4)
        
        plt.subplot(2,2,3)
        plt.plot(Y, p_av, label=r'$\langle p \rangle$')
        plt.fill_between(Y, p_av - std[7], p_av + std[7], alpha=0.3)
        plt.plot(Y, rhov2_avXv2_av, label=r'$\langle \rho v_{x2} \rangle \langle v_{x2} \rangle$')
        plt.fill_between(Y, rhov2_avXv2_av - std[2], rhov2_avXv2_av + std[2], alpha=0.3)
        plt.plot(Y, R_zz, label=r'$\langle \delta v_{x2} \delta(\rho  v_{x2}) \rangle$')
        plt.fill_between(Y, R_zz - std[6], R_zz + std[6], alpha=0.3)
        plt.plot(Y, p_av+rhov2_avXv2_av+R_zz, label=r'$\langle p  + \rho v_{x2}^2 \rangle$')
        plt.fill_between(Y, P_rhov2_sqr - std[12], P_rhov2_sqr + std[12], alpha=0.3)
        plt.title('x2-momentum Flux')
        plt.xlabel('x2')
        plt.xlim(YMIN, YMAX)
        #plt.ylim(-2,20)
        plt.grid()
        plt.legend(fontsize=4)

        plt.subplot(2,2,4)
        plt.plot(Y, Be_av_rhov2_av, label=r'$\left [ \langle B_e \rangle \langle \rho  v_{x2} \rangle \right ]$')
        plt.fill_between(Y, Be_av_rhov2_av - std[8], Be_av_rhov2_av + std[8], alpha=0.3)
        plt.plot(Y, del_Be_del_rhov2_av, label=r'$\left [ \langle \delta B_e \delta(\rho  v_{x2}) \rangle \right ]$')
        plt.fill_between(Y, del_Be_del_rhov2_av - std[9], del_Be_del_rhov2_av + std[9], alpha=0.3)
        plt.plot(Y, edot_cool_cum_dx2, label=r'$\int \langle n^2 \Lambda(T) \rangle dx_2$')
        plt.fill_between(Y, edot_cool_cum_dx2 - std[10], edot_cool_cum_dx2 + std[10], alpha=0.3)
        plt.plot(Y, net_heating, label='(net heating)/area')
        plt.fill_between(Y, net_heating - std[13], net_heating + std[13], alpha=0.3)
        plt.title('Energy Flux')
        plt.legend(fontsize=4)
        plt.grid()
        plt.tight_layout()
        plt.xlabel('x2')
        plt.xlim(YMIN, YMAX)
        #plt.ylim([-10,10])
        #plt.ylim(-1000, 1000) 
        
        plt.suptitle('Horizontally Averaged Fluxes')
        plt.tight_layout()
        plt.suptitle(str(NX*F//2**max_level) + 'x' + str(NY*F//2**max_level) + 'x' + str(NZ*F//2**max_level) + ' Snapshot ' + str(i*batch_size) + '-' + str((i+1)*batch_size) + ' SMR = ' + str(max_level) + ', Coarsening Factor = ' + str(F))
        plt.savefig(f"{dir}KH_mean_fluxes_time_averaged_batch{batch_size}_{c}_{str(i).zfill(5)}" + f'C{F}' + ".png", dpi = 300)
        plt.clf()
        plt.close()"""
    
    return 0.0

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
            parser.add_argument('-b', type=int, default=25, help="Batch size for processing")
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
        weight = 'm'  # Default weight is mass
        batch = arg_dict['b']
        SetGlobals(path_to_files, F, n1, n2, batch)

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
        print(f"Processing snapshot {i} for TRML vel...")
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

    comm.Barrier()"""  

    for i in range(N1_local, N2_local):
        print(f"Adding flux profiles for snapshot {i}...")
        dens, velx, velz, ps, prs, vely, temp, times = ReadBinFile(path_to_files, i, F)
        v1_tRML_local_, v2_tRML_local_, v3_tRML_local_ = ComputeTRML_velocity(dens, velx, vely, velz, i, F)
        # v_1_tRML and v#_tRML are 0
        compute_flux_profiles(dens, velx, velz, ps, prs, vely, temp, i, times, F,v1_tRML_local_, v2_tRML_local_, v3_tRML_local_)

    if MPI_DEF:
        if rank < nproc_extra: # Process extra files across ranks
            n = size * nfiles_local + rank + n1
            print(f"Processing extra flux snapshot {n}")
            dens, velx, velz, ps, prs, vely, temp, times = ReadBinFile(path_to_files, n, F)
            v1_tRML_local_, v2_tRML_local_, v3_tRML_local_ = ComputeTRML_velocity(dens, velx, vely, velz, n, F)
            compute_flux_profiles(dens, velx, velz, ps, prs, vely, temp, n, times, F, v1_tRML_local_, v2_tRML_local_, v3_tRML_local_)

if __name__ == "__main__":
    main()
