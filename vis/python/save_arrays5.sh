#!/bin/bash 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=02:59:00
#SBATCH --job-name snaps_making_normlogcool22k_256and128
#SBATCH --output=fluxsave10_40_snapsmakingnormlogcool22k_256and128_1_3_%j.txt
#SBATCH --mail-user=arnavkumar@iisc.ac.in
#SBATCH --mail-type=ALL
#SBATCH --account=rrg-babul-ad
 
cd $SCRATCH/athenak/vis/python
 
module purge
module load gcc/14.3
module load openmpi/5.0.8
module load mpi4py/4.1.0
source $SCRATCH/athena_env/bin/activate
export MPLCONFIGDIR=$SCRATCH

srun python3 -u save_fluxes_3D.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/normlog22k_256_1024_1_3/bin/ -n1 0 -n2 125 -F 1
srun python3 -u save_fluxes_3D.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/normlog22k_128_512_1_3/bin/ -n1 0 -n2 125 -F 1
#srun python3 -u save_fluxes_3D.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/fiducialnovrel/bin/ -n1 0 -n2 125 -F 1

echo "fluxes saved"

srun python3 -u save_2D_arrays_3D.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/normlog22k_256_1024_1_3/bin/ -n1 0 -n2 125 -F 1
srun python3 -u save_2D_arrays_3D.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/normlog22k_128_512_1_3/bin/ -n1 0 -n2 125 -F 1
#srun python3 -u save_2D_arrays_3D.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/fiducialnovrel/bin/ -n1 0 -n2 125 -F 1
#srun python3 -u make_avg_arrays.py
echo "2d saved"

srun python3 -u save_1D_profiles.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/normlog22k_256_1024_1_3/bin/ -n1 0 -n2 125 -F 1
srun python3 -u save_1D_profiles.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/normlog22k_128_512_1_3/bin/ -n1 0 -n2 125 -F 1
#srun python3 -u save_1D_profiles.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/fiducialnovrel/bin/ -n1 0 -n2 125 -F 1
echo "1d saved" 

srun python3 -u plot_2Dslices_fromNPZ.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/normlog22k_256_1024_1_3/bin/ -n1 0 -n2 125 -F 1
srun python3 -u plot_2Dslices_fromNPZ.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/normlog22k_128_512_1_3/bin/ -n1 0 -n2 125 -F 1
#srun python3 -u plot_2Dslices_fromNPZ.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/fiducialnovrel/bin/ -n1 0 -n2 125 -F 1
echo "2d plotted"

srun python3 -u plot_1D_PDFs_fromNPZ.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/normlog22k_256_1024_1_3/bin/ -n1 0 -n2 125 -F 1
srun python3 -u plot_1D_PDFs_fromNPZ.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/normlog22k_128_512_1_3/bin/ -n1 0 -n2 125 -F 1
#srun python3 -u plot_1D_PDFs_fromNPZ.py -i ../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/fiducialnovrel/bin/ -n1 0 -n2 125 -F 1

echo "1D plotted"
deactivate
