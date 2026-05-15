import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import bin_convert
import analyse_bin
import math
import matplotlib.cm as cm
import gc  
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

from save_2D_arrays_3D import ATOMIC_MASS, LENGTH, TIME, MASS, VELOCITY, DENSITY, ENERGY, POWER, PRESSURE, TEMPERATURE, MU, N_UNIT, COOLING_UNIT

# De- dimensionalising facotrs
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
NX = 256
NY = 1024
NZ = 256
max_level = 0
DY = 40./NY
T_inflection = (T_h + T_c)/2
# cell centre for ith cell is at i*DY + DY/2 + YMIN

global dir
#dir = r"../../my_outputs/noSMR_2_3_cutoffISMcoolfn/fid3D_32_cool/bin/"
#dir = r"../../my_outputs/fiducial1040_cool2D/bin/"
#dir = r"../../my_outputs/fid3D_2xlessvel_1040_cool/bin/"
#dir = r"../../my_outputs/noise_tests/noSMR_2_3_cutoffISMcoolfn/fiducialno_cooling/bin/"
#dir = r"../../my_outputs/fid3D_halfbox_1040_cool/bin/"
#dir = r"../../../Downloads/Chandra_data/snapsfiducial16cool/"
dir = r"../../../Downloads/Niagara3Dfidnpz/"
#dir = r"../../my_outputs/noSMR_2_3_cutoffISMcoolfn/fid3D_5xmoredens_1040_cool/bin/"

from save_2D_arrays_3D import ISMCoolFn

#make an array of 5 colours from matplotlib.cm
cmap = cm.get_cmap('viridis')
colours = [cmap(i) for i in np.linspace(0, 1, 8)]

colours = ["#076c4e", "#FFEE00", "#8b85ea", "#e80178", "#82ed08", "#8D0000", '#a6761d', "#00fff7"]

custom_lines = [
    Line2D([0], [0], color=colours[0], lw=3),
    Line2D([0], [0], color=colours[1], lw=3),
    Line2D([0], [0], color=colours[2], lw=3),
    Line2D([0], [0], color=colours[3], lw=3),
    Line2D([0], [0], color=colours[4], lw=3),
    Line2D([0], [0], color=colours[5], lw=3),
    Line2D([0], [0], color=colours[6], lw=3),
    Line2D([0], [0], color=colours[7], lw=3)
]
# list out the possible marker shapes in plt.errorbar
marker_shapes = ['o', 's', 'D', '^', 'v', '<', '>', 'p']

custom_markers = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='black', markersize=10),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='black', markersize=10),
    Line2D([0], [0], marker='^', color='w', markerfacecolor='black', markersize=10),
    Line2D([0], [0], marker='v', color='w', markerfacecolor='black', markersize=10),
    Line2D([0], [0], marker='<', color='w', markerfacecolor='black', markersize=10),
    Line2D([0], [0], marker='>', color='w', markerfacecolor='black', markersize=10),
    Line2D([0], [0], marker='p', color='w', markerfacecolor='black', markersize=10)
]
# list the possible linestyles for errorbars
linestyles = ['-', ':', '-.', '--']

custom_linestyles = [
    Line2D([0], [0], color='black', linestyle='-'),
    Line2D([0], [0], color=colours[5], linestyle=':'),
    Line2D([0], [0], color=colours[6], linestyle='-.'),
    Line2D([0], [0], color=colours[7], linestyle='--'),
]

marker_faces = ['black', 'none']

custom_marker_faces = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markersize=10),
]

# defining a global array that holds the run name, P_0, rho_h,rho_c, L_perp, v_rel, T_0, and observed Sigma_cool and its 1-sigma error and color
runs_info = [
    ["fid_1040_3D", 14.02645, 0.001, 0.1, 10.0, 31., 1.0e5, 1.43, 0.21, colours[0], 1, marker_shapes[0], linestyles[0], marker_faces[0]],
    [r"$5\rho_h$", 14.02645*5., 0.005, 0.5, 10.0, 31., 1.0e5, 1.57, 0.15, colours[1], 1, marker_shapes[0], linestyles[0], marker_faces[0]],
    [r"$\rho_h/5$", 14.02645/5., 0.001/5., 0.1/5., 10.0, 31., 1.0e5,0.69, 0.1, colours[2], 1, marker_shapes[0], linestyles[0], marker_faces[0]],
    [r"$2\Delta u$", 14.02645, 0.001, 0.1, 10.0, 31.*2., 1.0e5, 0.93, 0.13, colours[5], 1, marker_shapes[0], linestyles[1], marker_faces[1]],
    [r"$\Delta u/2$", 14.02645, 0.001, 0.1, 10.0, 31./2., 1.0e5 ,1.81, 0.4, colours[6], 1, marker_shapes[0], linestyles[2], marker_faces[1]],
    ["res-hlf", 14.02645, 0.001, 0.1, 10.0, 31., 1.0e5, 1.39, 0.18, colours[0], 2, marker_shapes[0], linestyles[0], marker_faces[0]],
    [r"$L_{perp}-hlf$", 14.02645, 0.001, 0.1, 5.0, 31., 1.0e5, 0.97, 0.11, colours[0], 1, marker_shapes[1], linestyles[0], marker_faces[0]],
    [r"$res-hlf(5\rho_h)$", 14.02645*5., 0.005, 0.5, 10.0, 31., 1.0e5, 1.59, 0.09, colours[1], 2, marker_shapes[0], linestyles[0], marker_faces[0]],
    [r"$res-hlf(5\rho_h),\ L_{perp}-hlf$", 14.02645*5., 0.005, 0.5, 5.0, 31., 1.0e5, 1.61, 0.09, colours[1], 2, marker_shapes[1], linestyles[0], marker_faces[0]],
    [r"$res-hlf(5\rho_h),\ L_{perp}-2x$", 14.02645*5., 0.005, 0.5, 20.0, 31., 1.0e5, 1.60, 0.07, colours[1], 2, marker_shapes[2], linestyles[0], marker_faces[0]],
    ["fid(2D)", 14.02645, 0.001, 0.1, 10.0, 31., 1.0e5, 1.66, 0.36, colours[7], 1, marker_shapes[0], linestyles[3], marker_faces[0]],
    [r"$res-1/4$x (fid)", 14.02645, 0.001, 0.1, 10.0, 31., 1.0e5, 1.21, 0.14, colours[0], 3, marker_shapes[0], linestyles[0], marker_faces[0]],
    [r"$res-1/8$x (fid)", 14.02645, 0.001, 0.1, 10.0, 31., 1.0e5, 1.25, 0.19, colours[0], 4, marker_shapes[0], linestyles[0], marker_faces[0]],
    [r"$res\approx 1/8$x fid", 14.02645, 0.001, 0.1, 10.0, 31., 1.0e5, 1.18, 0.18, colours[0], 4, marker_shapes[0], linestyles[0], marker_faces[0]],
    [r"$res\approx 1/16$x fid", 14.02645, 0.001, 0.1, 10.0, 31., 1.0e5, 0.97, 0.54, colours[0], 5, marker_shapes[0], linestyles[0], marker_faces[0]],
    [r"$res\approx 1/32$x fid", 14.02645, 0.001, 0.1, 10.0, 31., 1.0e5, 1.09, 0.42, colours[0], 6, marker_shapes[0], linestyles[0], marker_faces[0]],
    [r"$res\approx 1/64$x fid", 14.02645, 0.001, 0.1, 10.0, 31., 1.0e5, 1.93, 0.89, colours[0], 7, marker_shapes[0], linestyles[0], marker_faces[0]],
    [r"$res\approx 1/2$x (fid) $L_{perp}-1x$", 14.02645, 0.001, 0.1, 10.0, 31., 1.0e5, 1.36, 0.18, colours[0], 2, marker_shapes[0], linestyles[0], marker_faces[0]],
    [r"$res\approx 1/2$x (fid) $L_{perp}-2x$", 14.02645, 0.001, 0.1, 20.0, 31., 1.0e5, 1.48, 0.27, colours[0], 2, marker_shapes[2], linestyles[0], marker_faces[0]],
    [r"$res\approx 1/2$x (fid) $L_{perp}-4x$", 14.02645, 0.001, 0.1, 40.0, 31., 1.0e5, 1.46, 0.23, colours[0], 2, marker_shapes[3], linestyles[0], marker_faces[0]],
    [r"$res\approx 1/2$x (fid) $L_{perp}-8x$", 14.02645, 0.001, 0.1, 80.0, 31., 1.0e5, 1.41, 0.18, colours[0], 2, marker_shapes[4], linestyles[0], marker_faces[0]],
    [r"fid\ res-2x", 14.02645, 0.001, 0.1, 10.0, 31., 1.0e5, 1.39, 0.17, colours[0], 0, marker_shapes[0], linestyles[0], marker_faces[0]],
    [r"$5\rho_h$ res-2x", 14.02645*5., 0.005, 0.5, 10.0, 31., 1.0e5, 1.52, 0.15, colours[1], 0, marker_shapes[0], linestyles[0], marker_faces[0]],
    [r"3\rho_h-L_{\perp}-2x", 14.02645*3., 0.003, 0.3, 20.0, 31., 1.0e5, 1.45, 0.16, colours[3], 1, marker_shapes[2], linestyles[0], marker_faces[0]],
    [r"4\rho_h-L_{\perp}-2x", 14.02645*4., 0.004, 0.4, 20.0, 31., 1.0e5, 1.45, 0.12, colours[4], 1, marker_shapes[2], linestyles[0], marker_faces[0]],
]

def give_chi(rhoc,rhoh):
    return rhoc/rhoh

def give_xi(L_perp, v_rel, cooling_time):
    return L_perp / (v_rel * cooling_time)

def give_t0(P0, rho0):
    time_0 = 3.*P0/(2.*rho0*rho0*ISMCoolFn(T_0)/COOLING_UNIT)
    return time_0

def give_pred_cooling_rate1_2(chi,xi):
    return 0.025*2.5*math.pow(chi,3./8.)*math.pow(xi, 0.5)


def give_pred_cooling_rate2(chi, xi):
    return 0.043*2.5*math.pow(chi,3./8.)*math.pow(xi, 0.25)


def main():

    fig, ax = plt.subplots(figsize=(8, 6))

    for run in runs_info:
        run_name = run[0]
        Pres = run[1]
        rhoh = run[2]
        rhoc = run[3]
        Lperp = run[4]
        vrel = run[5]
        T0 = run[6]
        Sigma_cool_obs = run[7]
        Sigma_cool_err = run[8]

        rho0 = Pres/(T0/TEMPERATURE)
        chi = give_chi(rhoc, rhoh)
        t0 = give_t0(Pres, rho0)
        xi = give_xi(Lperp, vrel, t0)
        Sigma_cool_pred = give_pred_cooling_rate1_2(chi, xi)
        Sigma_cool_pred2 = give_pred_cooling_rate2(chi, xi)
        print(f"Check: Sigma_cool_pred: {Sigma_cool_pred}, Sigma_cool_pred2: {Sigma_cool_pred2}")

        if not (run[9] == colours[0] and run[10] > 2):
            err = ax.errorbar(xi, Sigma_cool_obs, yerr=Sigma_cool_err, fmt='o',markersize=10, capsize = 5, capthick=2, label=run_name, alpha=1./(math.pow(1.25, run[10])), marker=run[11], markeredgecolor='black' ,markerfacecolor=run[9], ecolor=run[9])
            print(f"Run: {run_name}, xi: {xi}, xi^-0.25: {math.pow(xi, -0.25)}, chi: {chi}, chi^3/8:{math.pow(chi,3./8.)} Sigma_cool_obs: {Sigma_cool_obs}, Sigma_cool_pred: {Sigma_cool_pred}")
            for bar in err[2]:
                bar.set_linestyle(run[12])
                bar.set_linewidth(2)


    ax.plot(np.linspace(0.01, 200, 100), np.vectorize(give_pred_cooling_rate2)(100,np.linspace(0.01, 200, 100)), label=r"$\propto \xi^1/4$", color='black', linestyle='--')
    #ax.plot(np.linspace(0.01, 200, 100), np.vectorize(give_pred_cooling_rate1_2)(100,np.linspace(0.01, 200, 100)), label=r"$\propto \xi^1/2$", color='black', linestyle='-')
    ax.set_ylim(0.5,2.5)
    ax.set_xlim(1,200)
    ax.set_xlabel(r'$\xi = L_\perp/(\Delta ut_0)$', fontsize=22)
    ax.set_ylabel(r'$\dot{\Sigma}_{cool}/(p_0\Delta u)$', fontsize=22)
    legend1 = ax.legend(custom_lines, [r"$fiducial$", r"$5\rho_h$", r"$1/5\rho_h$", r"$3\rho_h$", r"$4\rho_h$"], loc='lower right', fontsize=13)
    legend2 = ax.legend(custom_markers, [r"$L_\perp=1$x", r"$L_\perp=1/2$x", r"$L_\perp=2$x", r"$L_\perp=4$x", r"$L_\perp=8$x"], loc='lower center', bbox_to_anchor=(0.65, 0.0), fontsize=13)
    legend3 = ax.legend(custom_linestyles, [r"$fiducial$", r"$2\Delta u$", r"$\Delta u/2$", r"$fid(2D)$"], loc='lower center', bbox_to_anchor=(0.42, 0.0), fontsize=13)
    ax.add_artist(legend1)
    ax.add_artist(legend2)
    ax.add_artist(legend3)
    ax.grid(axis='both', which='both', linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=17)
    #ax.title(r"Scalings for $\dot{\Sigma}_{cool}/p_0\Delta u$ vs $\xi$ for all the runs", fontsize=24, x=0.43, y=-0.15)
    plt.tight_layout()
    plt.savefig('scalings_figurecoloured.png', dpi=300)


    fig2, ax = plt.subplots(figsize=(8, 6))

    for run in runs_info:
        run_name = run[0]
        Pres = run[1]
        rhoh = run[2]
        rhoc = run[3]
        Lperp = run[4]
        vrel = run[5]
        T0 = run[6]
        Sigma_cool_obs = run[7]
        Sigma_cool_err = run[8]

        rho0 = Pres/(T0/TEMPERATURE)
        chi = give_chi(rhoc, rhoh)
        t0 = give_t0(Pres, rho0)
        xi = give_xi(Lperp, vrel, t0)
        Sigma_cool_pred = give_pred_cooling_rate1_2(chi, xi)
        Sigma_cool_pred2 = give_pred_cooling_rate2(chi, xi)
        print(f"Check: Sigma_cool_pred: {Sigma_cool_pred}, Sigma_cool_pred2: {Sigma_cool_pred2}")

        if (run[9] == colours[0] and xi > 16 and xi < 17):
            global XI
            XI = xi
            ax.errorbar(run[10]-1, Sigma_cool_obs, yerr=Sigma_cool_err, fmt='o',markersize=10, capsize = 5, capthick=2, label=run_name, color = run[9])
        
    ax.plot(np.linspace(-2, 8, 100), np.full(100,give_pred_cooling_rate2(100, XI)), label='Predicted Scaling', color='black', linestyle='--')
    ax.set_ylim(0,3.0)
    ax.set_xlim(-2, 7)
    ax.set_xlabel(r'$\alpha$', fontsize=22)
    ax.set_ylabel(r'$\dot{\Sigma}_{cool}/(p_0\Delta u)$', fontsize=22)
    ax.grid(axis='both', which='both', linestyle='--', alpha=0.5)
    ax.tick_params(axis='x', labelsize=17)
    ax.tick_params(axis='y', labelsize=17)
    plt.tight_layout()
    plt.savefig('scalings_figure_res.png')


if __name__ == "__main__":
    main()


