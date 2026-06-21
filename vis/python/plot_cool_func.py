import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate

from save_2D_arrays_3D import PRESSURE, COOLING_UNIT, TEMPERATURE
from make_figure_3_4 import P_0

global k_B
k_B = 1.380649e-16  # Boltzmann constant in erg/K

def ISMCoolFn_stock(temp):
    # original data from Shure et al. paper, covers 4.12 < logt < 8.16
    norm22k = 3.5350949795856343/21.866519671987078
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
        return norm22k * (2.0e-19*math.exp(-1.184e5/(temp + 1.0e3)) + 2.8e-28*math.sqrt(temp)*math.exp(-92.0/temp))
  
    if (temp > 0.95e6):
      return 0.0
    # for temperatures above 10^8.15 use CGOLS fit
    if (logt > 8.15):
      return norm22k * pow(10.0, (0.45*logt - 26.065))
  
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
    return norm22k * pow(10.0,logcool)

def calculate_norm_factors():
    global norm_factor_normal_lambda, norm_factor_const_lambda, norm_factor_inverted_lambda
    temp_grid = np.linspace(1.05e4, 0.95e6, 1000)
    area_stock = integrate.simpson([ISMCoolFn_stock(t) for t in temp_grid], temp_grid)

    area_lambda_func = integrate.simpson([(math.exp(-0.5*((math.log10(t) - 5.0)/0.2)**2)) for t in temp_grid], temp_grid)
    norm_factor_normal_lambda = area_stock / area_lambda_func

    area_const_lambda = integrate.simpson([1.0 for t in temp_grid], temp_grid)
    norm_factor_const_lambda = area_stock / area_const_lambda

    area_inverted_lambda = integrate.simpson([math.exp(0.5*((math.log10(t) - 5.0)/0.2)**2) for t in temp_grid], temp_grid)
    #area_inverted_lambda = integrate.simpson([2*constant_cool_fn(t) - log_norm_cool_fn(t) for t in temp_grid], temp_grid)
    norm_factor_inverted_lambda = area_stock / area_inverted_lambda
   
def log_norm_cool_fn(temp):
    # generate a log-normal distribution centered around 10^5 K with a width such that area under curve is same as area under ISM cooling function 
    # from 10^4 K to 10^6 K
    norm22k = 4.841586997370533/21.866519671987078

    mean  = 5.0
    sigma = 0.2

    if (temp < 1.05e4 or temp > 0.95e6):
        return 0.0
    
    logt = math.log10(temp)
    lambda_T = math.exp(-0.5 * ((logt - mean)/sigma)*((logt - mean)/sigma))

    norm = 5.4591716620684276e-21
    return norm22k * norm * lambda_T

def constant_cool_fn100less(temp):
    # return a constant cooling rate 
    norm22k = 239.77557073519606/(4.0*21.866519671987078)

    # return a constant cooling rate 
    if (temp < 1.05e4) or (temp > 0.95e6):
        return 0.0

    norm_factor_const_lambda = 7.457876781873938e-24

    return norm22k * norm_factor_const_lambda

def inverted_log_norm_cool_fn(temp):
    # generate an inverted log-normal distribution centered around 10^5 K with
    # a width such that area under curve is same as area under ISM cooling function 
    # from 10^4 K to 10^6 K

    norm22k = 1.0

    mean  = 5.0
    sigma = 0.2

    if (temp < 1.05e4 or temp > 0.95e6):
        return 0.0
    
    logt = math.log10(temp)
    lambda_T = math.exp(0.5 * ((logt - mean)/sigma)*((logt - mean)/sigma))

    norm = 5.253246378625558e-28
    return (norm22k * norm/4.0) * lambda_T

def hot_peaked_cool_fn(temp):

    norm22k = 13.226398492299235/21.866519671987078

    if (temp < 1.05e4) or (temp > 0.75e6):
        return 0.0
       
    norm_hotpeak = 5.4591716620684276e-43
    return norm22k * norm_hotpeak * (temp**4)

def plot_cooling_fn_paper():

    temps = np.logspace(4, 6, 1000)
    calculate_norm_factors()

    cooling_rates_stock = [ISMCoolFn_stock(temp) for temp in temps]
    cooling_rates_norm = [log_norm_cool_fn(temp) for temp in temps]
    cooling_rates_const = [constant_cool_fn100less(temp) for temp in temps]
    cooling_rates_inverted = [inverted_log_norm_cool_fn(temp) for temp in temps]
    cooling_rates_hotpeak = [hot_peaked_cool_fn(temp) for temp in temps]

    Lambda_stock = np.vectorize(ISMCoolFn_stock)(temps)/COOLING_UNIT
    Lambda_norm = np.vectorize(log_norm_cool_fn)(temps)/COOLING_UNIT
    Lambda_const = np.vectorize(constant_cool_fn100less)(temps)/COOLING_UNIT
    Lambda_inverted = np.vectorize(inverted_log_norm_cool_fn)(temps)/COOLING_UNIT
    Lambda_hotpeak = np.vectorize(hot_peaked_cool_fn)(temps)/COOLING_UNIT

    P_0_array = P_0 * np.ones_like(temps)
    cooling_time_inverted_lambda = np.divide(3. * (temps/TEMPERATURE)**2 , 2*P_0_array * Lambda_inverted, out=np.full_like(temps, 1e30, dtype=float), 
                    where=Lambda_inverted != 0)
    cooling_time_stock = np.divide(3. * (temps/TEMPERATURE)**2 , 2*P_0_array * Lambda_stock, out=np.full_like(temps, 1e30, dtype=float),
                    where=Lambda_stock != 0)
    cooling_time_norm_lambda = np.divide(3. * (temps/TEMPERATURE)**2 , 2*P_0_array * Lambda_norm, out=np.full_like(temps, 1e30, dtype=float),
                    where=Lambda_norm != 0)
    cooling_time_const_lambda = np.divide(3. * (temps/TEMPERATURE)**2 , 2*P_0_array * Lambda_const, out=np.full_like(temps, 1e30, dtype=float),
                    where=Lambda_const != 0)
    cooling_time_hotpeak = np.divide(3. * (temps/TEMPERATURE)**2 , 2*P_0_array * Lambda_hotpeak, out=np.full_like(temps, 1e30, dtype=float),
                    where=Lambda_hotpeak != 0)


    area_stock = integrate.simpson(cooling_rates_stock, temps)
    print(f"Area under the cooling function curve (stock): {area_stock:.2e} erg cm^3 s^-1 K")
    area_norm = integrate.simpson(cooling_rates_norm, temps)
    print(f"Area under the cooling function curve (log-normal): {area_norm:.2e} erg cm^3 s^-1 K with normalization factor {norm_factor_normal_lambda}")
    area_const = integrate.simpson(cooling_rates_const, temps)
    print(f"Area under the cooling function curve (constant): {area_const:.2e} erg cm^3 s^-1 K with normalization factor {norm_factor_const_lambda}")
    area_inverted = integrate.simpson(cooling_rates_inverted, temps)
    print(f"Area under the cooling function curve (inverted log-normal): {area_inverted:.2e} erg cm^3 s^-1 K with normalization factor {norm_factor_inverted_lambda}")
    area_hotpeak = integrate.simpson(cooling_rates_hotpeak, temps)
    print(f"Area under the cooling function curve (hot-peaked): {area_hotpeak:.2e} erg cm^3 s^-1 K")
    
    plt.figure(figsize=(12, 7))

    plt.subplot(1,2,1)
    plt.plot(temps, cooling_rates_norm, label=r"$\log(\Lambda) \propto \exp (-\frac{(\log T_0 - \log T)^2}{2\sigma^2})$", color='red')
    plt.plot(temps, cooling_rates_inverted, label=r"$\log(\Lambda) \propto \exp (\frac{(\log T_0 - \log T)^2}{2\sigma^2})$", color='orange')
    plt.plot(temps, cooling_rates_const, label=r"$\Lambda \propto T^0$", color='green')
    plt.plot(temps, cooling_rates_hotpeak, label=r"$\Lambda \propto T^4$", color='purple')
    plt.plot(temps, cooling_rates_stock, label='Schure et al. (2009)', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-28, 2e-18)
    plt.xlabel(r"T (K)", fontsize=14)
    plt.ylabel(r"Cooling Rate $ \Lambda \ (erg\ cm^3\ s^{-1})$", fontsize=14)
    plt.grid(which="both", axis='both', ls="--")
    plt.legend(loc='upper left', fontsize=12)

    plt.subplot(1,2,2)
    plt.plot(temps, cooling_time_norm_lambda, label=r"$\log(\Lambda) \propto \exp (-\frac{(\log T_0 - \log T)^2}{2\sigma^2})$", color='red')
    plt.plot(temps, cooling_time_inverted_lambda, label=r"$\log(\Lambda) \propto \exp (\frac{(\log T_0 - \log T)^2}{2\sigma^2})$", color='orange')
    plt.plot(temps, cooling_time_const_lambda,label=r"$\Lambda \propto T^0$", color='green')
    plt.plot(temps, cooling_time_hotpeak, label=r"$\Lambda \propto T^4$", color='purple')
    plt.plot(temps, cooling_time_stock, label='Schure et al. (2009)', color='blue')
    plt.xlabel(r"T (K)", fontsize=14)
    plt.ylabel(r'$t_{cool} = \frac{3/2 \, P}{\left(\frac{P}{k_B T}\right)^2 \Lambda (T)}$ (Myr)', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-6,1e6)
    plt.grid(which="both", axis='both', ls="--")    
    plt.legend(loc='lower right', fontsize=12)

    print(f"min cooling time for log-normal lambda function: {np.min(cooling_time_norm_lambda):.2e} Myr")
    print(f"min cooling time for constant lambda function: {np.min(cooling_time_const_lambda):.2e} Myr")
    print(f"min cooling time for inverted log-normal lambda function: {np.min(cooling_time_inverted_lambda):.2e} Myr")
    print(f"min cooling time for hot-peaked lambda function: {np.min(cooling_time_hotpeak):.2e} Myr")
    print(f"min cooling time for stock lambda function: {np.min(cooling_time_stock):.2e} Myr")

    plt.tight_layout()
    plt.savefig("cooling_function_and_times_comparison.png", dpi=300)

def plot_cooling_rate_comparison():

    with np.load(path + f'KH_1D_arrays_time_averaged{n1}to{n2}with{jump}.npz', 'r') as f:
        Sigma_dot_cool_hist_1D = f['Sigma_dot_cool_hist']

    with np.load(path + f'KH_fluxes_time_averaged{n1}to{n2}with{jump}.npz', 'r') as f:
        Sigma_dot_cool_flux = f['Sigma_dot_cool_hist']

    import athena_read

    Sigma_dot_coolfromhstdict = athena_read.hst(path + 'KH.hydro.hst')

    plt.figure(figsize=(10, 6))
    print(Sigma_dot_coolfromhstdict['time'].shape)
    plt.plot(Sigma_dot_coolfromhstdict['time'], Sigma_dot_coolfromhstdict['Edot_total']/100, label='Cooling Rate directly from history file', color='blue')
    plt.plot(Sigma_dot_coolfromhstdict['time'][::2], Sigma_dot_cool_hist_1D, label=r'Cooling Rate by average and integrating $\int_{bottom}^{top} \left<n^2 \Lambda (T)\right>\, dz$ using python scripts', color='black')
    plt.plot(Sigma_dot_coolfromhstdict['time'][::2], Sigma_dot_cool_flux, label=r'Cooling Rate by average and integrating $\int_{bottom}^{top} \left<n^2 \Lambda (T)\right>\, dz$ using fluxes from python scripts', color='red', linestyle='--')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Time (Myr)')
    plt.yscale('log')
    plt.xlabel("Temperature (K)")
    plt.ylabel('Cooling Rate (code units)')
    plt.legend()
    plt.savefig(path + "cooling_rate_comparison.png", dpi = 150)

def plot_vturb_profiles(filename1, filename2, filename3, filename4, filename5, filename6):

    dir = r"../../../Downloads/Trillium_data/corrected_1_3_runs/"

    with np.load(dir + "snaps" + f'{filename1}22k_256_1024_1_3_' + f'/KH_1D_arrays_time_averaged{n1}to125with{jump}.npz', 'r') as f:
        v_turb_mw_whole_box_file1 = f['v_turb_mw_whole_box']/delU
        v_turb_mw_whole_box1_file1 = f['v_turb_mw_whole_box1']/delU
        v_turb_mw_whole_box2_file1 = f['v_turb_mw_whole_box2']/delU
        v_turb_mw_whole_box3_file1 = f['v_turb_mw_whole_box3']/delU
        v_turb_volw_whole_box_file1 = f['v_turb_volw_whole_box']/delU
        v_turb_volw_whole_box1_file1 = f['v_turb_volw_whole_box1']/delU
        v_turb_volw_whole_box2_file1 = f['v_turb_volw_whole_box2']/delU
        v_turb_volw_whole_box3_file1 = f['v_turb_volw_whole_box3']/delU
        v_turb_wholebox_file1 = f['v_turb_wholebox']/delU
        v_turb_wholebox1_file1 = f['v_turb_wholebox1']/delU
        v_turb_wholebox2_file1 = f['v_turb_wholebox2']/delU
        v_turb_wholebox3_file1 = f['v_turb_wholebox3']/delU

    times = np.linspace(0, 10.0, len(v_turb_mw_whole_box_file1))

    with np.load(dir + "snaps" + f'{filename1}22k_128_512_1_3_' + f'/KH_1D_arrays_time_averaged{n1}to125with{jump}.npz', 'r') as f:
        v_turb_mw_whole_box_file1half = f['v_turb_mw_whole_box']/delU
        v_turb_mw_whole_box1_file1half = f['v_turb_mw_whole_box1']/delU
        v_turb_mw_whole_box2_file1half = f['v_turb_mw_whole_box2']/delU
        v_turb_mw_whole_box3_file1half = f['v_turb_mw_whole_box3']/delU
        v_turb_volw_whole_box_file1half = f['v_turb_volw_whole_box']/delU
        v_turb_volw_whole_box1_file1half = f['v_turb_volw_whole_box1']/delU
        v_turb_volw_whole_box2_file1half = f['v_turb_volw_whole_box2']/delU
        v_turb_volw_whole_box3_file1half = f['v_turb_volw_whole_box3']/delU
        v_turb_wholebox_file1half = f['v_turb_wholebox']/delU
        v_turb_wholebox1_file1half = f['v_turb_wholebox1']/delU
        v_turb_wholebox2_file1half = f['v_turb_wholebox2']/delU
        v_turb_wholebox3_file1half = f['v_turb_wholebox3']/delU

    with np.load(dir + "snaps" + f'{filename2}22k_256_1024_1_3_' + f'/KH_1D_arrays_time_averaged{n1}to{n2}with{jump}.npz', 'r') as f:
        v_turb_mw_whole_box_file2 = f['v_turb_mw_whole_box']/delU
        v_turb_mw_whole_box1_file2 = f['v_turb_mw_whole_box1']/delU
        v_turb_mw_whole_box2_file2 = f['v_turb_mw_whole_box2']/delU
        v_turb_mw_whole_box3_file2 = f['v_turb_mw_whole_box3']/delU
        v_turb_volw_whole_box_file2 = f['v_turb_volw_whole_box']/delU
        v_turb_volw_whole_box1_file2 = f['v_turb_volw_whole_box1']/delU
        v_turb_volw_whole_box2_file2 = f['v_turb_volw_whole_box2']/delU
        v_turb_volw_whole_box3_file2 = f['v_turb_volw_whole_box3']/delU
        v_turb_wholebox_file2 = f['v_turb_wholebox']/delU
        v_turb_wholebox1_file2 = f['v_turb_wholebox1']/delU
        v_turb_wholebox2_file2 = f['v_turb_wholebox2']/delU
        v_turb_wholebox3_file2 = f['v_turb_wholebox3']/delU

    with np.load(dir + "snaps" + f'{filename2}22k_128_512_1_3_' + f'/KH_1D_arrays_time_averaged{n1}to{n2}with{jump}.npz', 'r') as f:
        v_turb_mw_whole_box_file2half = f['v_turb_mw_whole_box']/delU
        v_turb_mw_whole_box1_file2half = f['v_turb_mw_whole_box1']/delU
        v_turb_mw_whole_box2_file2half = f['v_turb_mw_whole_box2']/delU
        v_turb_mw_whole_box3_file2half = f['v_turb_mw_whole_box3']/delU
        v_turb_volw_whole_box_file2half = f['v_turb_volw_whole_box']/delU
        v_turb_volw_whole_box1_file2half = f['v_turb_volw_whole_box1']/delU
        v_turb_volw_whole_box2_file2half = f['v_turb_volw_whole_box2']/delU
        v_turb_volw_whole_box3_file2half = f['v_turb_volw_whole_box3']/delU
        v_turb_wholebox_file2half = f['v_turb_wholebox']/delU
        v_turb_wholebox1_file2half = f['v_turb_wholebox1']/delU
        v_turb_wholebox2_file2half = f['v_turb_wholebox2']/delU
        v_turb_wholebox3_file2half = f['v_turb_wholebox3']/delU

    with np.load(dir + "snaps" + f'{filename3}22k_256_1024_1_3_' + f'/KH_1D_arrays_time_averaged{n1}to{n2}with{jump}.npz', 'r') as f:
        v_turb_mw_whole_box_file3 = f['v_turb_mw_whole_box']/delU
        v_turb_mw_whole_box1_file3 = f['v_turb_mw_whole_box1']/delU
        v_turb_mw_whole_box2_file3 = f['v_turb_mw_whole_box2']/delU
        v_turb_mw_whole_box3_file3 = f['v_turb_mw_whole_box3']/delU
        v_turb_volw_whole_box_file3 = f['v_turb_volw_whole_box']/delU
        v_turb_volw_whole_box1_file3 = f['v_turb_volw_whole_box1']/delU
        v_turb_volw_whole_box2_file3 = f['v_turb_volw_whole_box2']/delU
        v_turb_volw_whole_box3_file3 = f['v_turb_volw_whole_box3']/delU
        v_turb_wholebox_file3 = f['v_turb_wholebox']/delU
        v_turb_wholebox1_file3 = f['v_turb_wholebox1']/delU
        v_turb_wholebox2_file3 = f['v_turb_wholebox2']/delU
        v_turb_wholebox3_file3 = f['v_turb_wholebox3']/delU

    with np.load(dir + "snaps" + f'{filename3}22k_128_512_1_3_' + f'/KH_1D_arrays_time_averaged{n1}to{n2}with{jump}.npz', 'r') as f:
        v_turb_mw_whole_box_file3half = f['v_turb_mw_whole_box']/delU
        v_turb_mw_whole_box1_file3half = f['v_turb_mw_whole_box1']/delU
        v_turb_mw_whole_box2_file3half = f['v_turb_mw_whole_box2']/delU
        v_turb_mw_whole_box3_file3half = f['v_turb_mw_whole_box3']/delU
        v_turb_volw_whole_box_file3half = f['v_turb_volw_whole_box']/delU
        v_turb_volw_whole_box1_file3half = f['v_turb_volw_whole_box1']/delU
        v_turb_volw_whole_box2_file3half = f['v_turb_volw_whole_box2']/delU
        v_turb_volw_whole_box3_file3half = f['v_turb_volw_whole_box3']/delU
        v_turb_wholebox_file3half = f['v_turb_wholebox']/delU
        v_turb_wholebox1_file3half = f['v_turb_wholebox1']/delU
        v_turb_wholebox2_file3half = f['v_turb_wholebox2']/delU
        v_turb_wholebox3_file3half = f['v_turb_wholebox3']/delU

    with np.load(dir + "snaps" + f'{filename4}22k_256_1024_1_3_' + f'/KH_1D_arrays_time_averaged{n1}to{n2}with{jump}.npz', 'r') as f:
        v_turb_mw_whole_box_file4 = f['v_turb_mw_whole_box']/delU
        v_turb_mw_whole_box1_file4 = f['v_turb_mw_whole_box1']/delU
        v_turb_mw_whole_box2_file4 = f['v_turb_mw_whole_box2']/delU
        v_turb_mw_whole_box3_file4 = f['v_turb_mw_whole_box3']/delU
        v_turb_volw_whole_box_file4 = f['v_turb_volw_whole_box']/delU
        v_turb_volw_whole_box1_file4 = f['v_turb_volw_whole_box1']/delU
        v_turb_volw_whole_box2_file4 = f['v_turb_volw_whole_box2']/delU
        v_turb_volw_whole_box3_file4 = f['v_turb_volw_whole_box3']/delU
        v_turb_wholebox_file4 = f['v_turb_wholebox']/delU
        v_turb_wholebox1_file4 = f['v_turb_wholebox1']/delU
        v_turb_wholebox2_file4 = f['v_turb_wholebox2']/delU
        v_turb_wholebox3_file4 = f['v_turb_wholebox3']/delU

    with np.load(dir + "snaps" + f'{filename4}22k_128_512_1_3_' + f'/KH_1D_arrays_time_averaged{n1}to{n2}with{jump}.npz', 'r') as f:
        v_turb_mw_whole_box_file4half = f['v_turb_mw_whole_box']/delU
        v_turb_mw_whole_box1_file4half = f['v_turb_mw_whole_box1']/delU
        v_turb_mw_whole_box2_file4half = f['v_turb_mw_whole_box2']/delU
        v_turb_mw_whole_box3_file4half = f['v_turb_mw_whole_box3']/delU
        v_turb_volw_whole_box_file4half = f['v_turb_volw_whole_box']/delU
        v_turb_volw_whole_box1_file4half = f['v_turb_volw_whole_box1']/delU
        v_turb_volw_whole_box2_file4half = f['v_turb_volw_whole_box2']/delU
        v_turb_volw_whole_box3_file4half = f['v_turb_volw_whole_box3']/delU
        v_turb_wholebox_file4half = f['v_turb_wholebox']/delU
        v_turb_wholebox1_file4half = f['v_turb_wholebox1']/delU
        v_turb_wholebox2_file4half = f['v_turb_wholebox2']/delU
        v_turb_wholebox3_file4half = f['v_turb_wholebox3']/delU
    
    with np.load(dir + "snaps" + f'{filename5}22k_256_1024_1_3_' + f'/KH_1D_arrays_time_averaged{n1}to{n2}with{jump}.npz', 'r') as f:
        v_turb_mw_whole_box_file5 = f['v_turb_mw_whole_box']/delU
        v_turb_mw_whole_box1_file5 = f['v_turb_mw_whole_box1']/delU
        v_turb_mw_whole_box2_file5 = f['v_turb_mw_whole_box2']/delU
        v_turb_mw_whole_box3_file5 = f['v_turb_mw_whole_box3']/delU
        v_turb_volw_whole_box_file5 = f['v_turb_volw_whole_box']/delU
        v_turb_volw_whole_box1_file5 = f['v_turb_volw_whole_box1']/delU
        v_turb_volw_whole_box2_file5 = f['v_turb_volw_whole_box2']/delU
        v_turb_volw_whole_box3_file5 = f['v_turb_volw_whole_box3']/delU
        v_turb_wholebox_file5 = f['v_turb_wholebox']/delU
        v_turb_wholebox1_file5 = f['v_turb_wholebox1']/delU
        v_turb_wholebox2_file5 = f['v_turb_wholebox2']/delU
        v_turb_wholebox3_file5 = f['v_turb_wholebox3']/delU

    with np.load(dir + "snaps" + f'{filename5}22k_128_512_1_3_' + f'/KH_1D_arrays_time_averaged{n1}to{n2}with{jump}.npz', 'r') as f:
        v_turb_mw_whole_box_file5half = f['v_turb_mw_whole_box']/delU
        v_turb_mw_whole_box1_file5half = f['v_turb_mw_whole_box1']/delU
        v_turb_mw_whole_box2_file5half = f['v_turb_mw_whole_box2']/delU
        v_turb_mw_whole_box3_file5half = f['v_turb_mw_whole_box3']/delU
        v_turb_volw_whole_box_file5half = f['v_turb_volw_whole_box']/delU
        v_turb_volw_whole_box1_file5half = f['v_turb_volw_whole_box1']/delU
        v_turb_volw_whole_box2_file5half = f['v_turb_volw_whole_box2']/delU
        v_turb_volw_whole_box3_file5half = f['v_turb_volw_whole_box3']/delU
        v_turb_wholebox_file5half = f['v_turb_wholebox']/delU
        v_turb_wholebox1_file5half = f['v_turb_wholebox1']/delU
        v_turb_wholebox2_file5half = f['v_turb_wholebox2']/delU
        v_turb_wholebox3_file5half = f['v_turb_wholebox3']/delU

    with np.load(dir + "snaps" + f'{filename6}22k_256_1024_1_3_' + f'/KH_1D_arrays_time_averaged{n1}to{n2}with{jump}.npz', 'r') as f:
        v_turb_mw_whole_box_file6 = f['v_turb_mw_whole_box']/delU
        v_turb_mw_whole_box1_file6 = f['v_turb_mw_whole_box1']/delU
        v_turb_mw_whole_box2_file6 = f['v_turb_mw_whole_box2']/delU
        v_turb_mw_whole_box3_file6 = f['v_turb_mw_whole_box3']/delU
        v_turb_volw_whole_box_file6 = f['v_turb_volw_whole_box']/delU
        v_turb_volw_whole_box1_file6 = f['v_turb_volw_whole_box1']/delU
        v_turb_volw_whole_box2_file6 = f['v_turb_volw_whole_box2']/delU
        v_turb_volw_whole_box3_file6 = f['v_turb_volw_whole_box3']/delU
        v_turb_wholebox_file6 = f['v_turb_wholebox']/delU
        v_turb_wholebox1_file6 = f['v_turb_wholebox1']/delU
        v_turb_wholebox2_file6 = f['v_turb_wholebox2']/delU
        v_turb_wholebox3_file6 = f['v_turb_wholebox3']/delU

    plt.figure(figsize=(14, 14))

    plt.subplot(2,1,1)
    plt.plot(times, v_turb_mw_whole_box_file1, label=r"$Shure\ et\ al.$", color='blue', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box_file1half, color='blue', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box_file2, label=r"inverted log-normal", color='yellow', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box_file2half, color='yellow', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box_file3, label=r"constant", color='green', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box_file3half, color='green', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box_file4, label=r"hot-peaked", color='purple', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box_file4half, color='purple', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box_file5, label=r"normlog", color='red', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box_file5half, color='red', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box_file6, label=r"non radiative", color='orange', linestyle='-')
    #plt.plot(times, v_turb_mw_whole_box_file6half, color='orange', linestyle=':')
    plt.yscale('log')
    plt.ylim(bottom=1e-2)
    plt.xlabel(r"time (My)", fontsize=14)
    plt.ylabel(r"$u_{turb}/\Delta u$", fontsize=14)
    plt.title(r"Mass-weighted turbulent velocity", fontsize=16)
    plt.legend(loc='best', fontsize=12, frameon=False)
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')


    plt.subplot(2,1,2)
    plt.plot(times, v_turb_volw_whole_box_file1, label=r"$Shure\ et\ al.$", color='blue', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box_file1half, color='blue', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box_file2, label=r"inverted log-normal", color='yellow', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box_file2half, color='yellow', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box_file3, label=r"constant", color='green', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box_file3half, color='green', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box_file4, label=r"hot-peaked", color='purple', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box_file4half, color='purple', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box_file5, label=r"normlog", color='red', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box_file5half, color='red', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box_file6, label=r"non radiative", color='orange', linestyle='-')
    #plt.plot(times, v_turb_volw_whole_box_file6half, color='orange', linestyle=':')
    plt.yscale('log')
    plt.ylim(bottom=3e-2)
    plt.xlabel(r"time (My)", fontsize=14)
    plt.ylabel(r"$u_{turb}/\Delta u$", fontsize=14)
    plt.title(r"Volume-weighted turbulent velocity", fontsize=16)
    plt.legend(loc='best', fontsize=12, frameon=False)
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')

    plt.savefig(dir + 'turbulent_velocity_profiles_compare.png', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(14, 14))

    plt.subplot(2,1,1)
    plt.plot(times, v_turb_mw_whole_box1_file1, label=r"$Shure\ et\ al.$", color='blue', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box1_file1half, color='blue', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box1_file2, label=r"inverted log-normal", color='yellow', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box1_file2half, color='yellow', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box1_file3, label=r"constant", color='green', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box1_file3half, color='green', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box1_file4, label=r"hot-peaked", color='purple', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box1_file4half, color='purple', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box1_file5, label=r"normlog", color='red', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box1_file5half, color='red', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box1_file6, label=r"non radiative", color='orange', linestyle='-')
    plt.yscale('log')
    plt.ylim(bottom=8e-3)
    plt.xlabel(r"time (My)", fontsize=14)
    plt.ylabel(r"$u_{turb,x}/\Delta u$", fontsize=14)
    plt.title(r"Mass-weighted turbulent velocity", fontsize=16)
    plt.legend(loc='best', fontsize=12, frameon=False)
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')


    plt.subplot(2,1,2)
    plt.plot(times, v_turb_volw_whole_box1_file1, label=r"$Shure\ et\ al.$", color='blue', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box1_file1half, color='blue', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box1_file2, label=r"inverted log-normal", color='yellow', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box1_file2half, color='yellow', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box1_file3, label=r"constant", color='green', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box1_file3half, color='green', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box1_file4, label=r"hot-peaked", color='purple', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box1_file4half, color='purple', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box1_file5, label=r"normlog", color='red', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box1_file5half, color='red', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box1_file6, label=r"non radiative", color='orange', linestyle='-')
    plt.yscale('log')
    plt.ylim(bottom=2e-2)
    plt.xlabel(r"time (My)", fontsize=14)
    plt.ylabel(r"$u_{turb,x}/\Delta u$", fontsize=14)
    plt.title(r"Volume-weighted turbulent velocity", fontsize=16)
    plt.legend(loc='best', fontsize=12, frameon=False)
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')

    plt.savefig(dir + 'turbulent_velocity_profiles_x_compare.png', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(14, 14))

    plt.subplot(2,1,1)
    plt.plot(times, v_turb_mw_whole_box2_file1, label=r"$Shure\ et\ al.$", color='blue', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box2_file1half, color='blue', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box2_file2, label=r"inverted log-normal", color='yellow', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box2_file2half, color='yellow', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box2_file3, label=r"constant", color='green', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box2_file3half, color='green', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box2_file4, label=r"hot-peaked", color='purple', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box2_file4half, color='purple', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box2_file5, label=r"normlog", color='red', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box2_file5half, color='red', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box2_file6, label=r"non radiative", color='orange', linestyle='-')
    plt.yscale('log')
    plt.ylim(bottom=8e-3)
    plt.xlabel(r"time (My)", fontsize=14)
    plt.ylabel(r"$u_{turb,z}/\Delta u$", fontsize=14)
    plt.title(r"Mass-weighted turbulent velocity", fontsize=16)
    plt.legend(loc='best', fontsize=12, frameon=False)
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')


    plt.subplot(2,1,2)
    plt.plot(times, v_turb_volw_whole_box2_file1, label=r"$Shure\ et\ al.$", color='blue', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box2_file1half, color='blue', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box2_file2, label=r"inverted log-normal", color='yellow', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box2_file2half, color='yellow', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box2_file3, label=r"constant", color='green', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box2_file3half, color='green', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box2_file4, label=r"hot-peaked", color='purple', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box2_file4half, color='purple', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box2_file5, label=r"normlog", color='red', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box2_file5half, color='red', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box2_file6, label=r"non radiative", color='orange', linestyle='-')
    plt.yscale('log')
    plt.ylim(bottom=1e-2)
    plt.xlabel(r"time (My)", fontsize=14)
    plt.ylabel(r"$u_{turb,z}/\Delta u$", fontsize=14)
    plt.title(r"Volume-weighted turbulent velocity", fontsize=16)
    plt.legend(loc='best', fontsize=12, frameon=False)
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')

    plt.savefig(dir + 'turbulent_velocity_profiles_z_compare.png', bbox_inches='tight', dpi=300)

    plt.figure(figsize=(14, 14))

    plt.subplot(2,1,1)
    plt.plot(times, v_turb_mw_whole_box3_file1, label=r"$Shure\ et\ al.$", color='blue', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box3_file1half, color='blue', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box3_file2, label=r"inverted log-normal", color='yellow', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box3_file2half, color='yellow', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box3_file3, label=r"constant", color='green', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box3_file3half, color='green', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box3_file4, label=r"hot-peaked", color='purple', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box3_file4half, color='purple', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box3_file5, label=r"normlog", color='red', linestyle='-')
    plt.plot(times, v_turb_mw_whole_box3_file5half, color='red', linestyle=':')
    plt.plot(times, v_turb_mw_whole_box3_file6, label=r"non radiative", color='orange', linestyle='-')
    plt.yscale('log')
    plt.ylim(bottom=8e-3)
    plt.xlabel(r"time (My)", fontsize=14)
    plt.ylabel(r"$u_{turb,y}/\Delta u$", fontsize=14)
    plt.title(r"Mass-weighted turbulent velocity", fontsize=16)
    plt.legend(loc='best', fontsize=12, frameon=False)
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')


    plt.subplot(2,1,2)
    plt.plot(times, v_turb_volw_whole_box3_file1, label=r"$Shure\ et\ al.$", color='blue', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box3_file1half, color='blue', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box3_file2, label=r"inverted log-normal", color='yellow', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box3_file2half, color='yellow', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box3_file3, label=r"constant", color='green', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box3_file3half, color='green', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box3_file4, label=r"hot-peaked", color='purple', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box3_file4half, color='purple', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box3_file5, label=r"normlog", color='red', linestyle='-')
    plt.plot(times, v_turb_volw_whole_box3_file5half, color='red', linestyle=':')
    plt.plot(times, v_turb_volw_whole_box3_file6, label=r"non radiative", color='orange', linestyle='-')
    plt.yscale('log')
    plt.ylim(bottom=1e-2)
    plt.xlabel(r"time (My)", fontsize=14)
    plt.ylabel(r"$u_{turb,y}/\Delta u$", fontsize=14)
    plt.title(r"Volume-weighted turbulent velocity", fontsize=16)
    plt.legend(loc='best', fontsize=12, frameon=False)
    plt.grid(which='both', axis='both', linestyle='--', linewidth=0.5, color='gray')

    plt.savefig(dir + 'turbulent_velocity_profiles_y_compare.png', bbox_inches='tight', dpi=300)

def plot_prsvs_temp(filename1, filename2, filename3, filename4, filename5, filename6, ni, nf, W, F):

    dir = r"../../../Downloads/Trillium_data/corrected_1_3_runs/"

    with np.load(dir + "snaps" + f'{filename1}22k_256_1024_1_3_' + f'/KH_jointPDFprstemp_percentiles{W}_snapshot_' + f'{ni}to{nf}' + f'C{F}' + '.npz', 'r') as f:
        avg_P1 = f['avg_P']
        median_P1 = f['median_P']
        p16_P1 = f['p16_P']
        p84_P1 = f['p84_P']
        bin_centers_x = f['bin_centers_x']
        bin_centers_y = f['bin_centers_y']

    with np.load(dir + "snaps" + f'{filename2}22k_256_1024_1_3_' + f'/KH_jointPDFprstemp_percentiles{W}_snapshot_' + f'{ni}to{nf}' + f'C{F}' + '.npz', 'r') as f:
        avg_P2 = f['avg_P']
        median_P2 = f['median_P']
        p16_P2 = f['p16_P']
        p84_P2 = f['p84_P']

    with np.load(dir + "snaps" + f'{filename3}22k_256_1024_1_3_' + f'/KH_jointPDFprstemp_percentiles{W}_snapshot_' + f'{ni}to{nf}' + f'C{F}' + '.npz', 'r') as f:
        avg_P3 = f['avg_P']
        median_P3 = f['median_P']
        p16_P3 = f['p16_P']
        p84_P3 = f['p84_P']

    with np.load(dir + "snaps" + f'{filename4}22k_256_1024_1_3_' + f'/KH_jointPDFprstemp_percentiles{W}_snapshot_' + f'{ni}to{nf}' + f'C{F}' + '.npz', 'r') as f:
        avg_P4 = f['avg_P']
        median_P4 = f['median_P']
        p16_P4 = f['p16_P']
        p84_P4 = f['p84_P']

    with np.load(dir + "snaps" + f'{filename5}22k_256_1024_1_3_' + f'/KH_jointPDFprstemp_percentiles{W}_snapshot_' + f'{ni}to{nf}' + f'C{F}' + '.npz', 'r') as f:
        avg_P5 = f['avg_P']
        median_P5 = f['median_P']
        p16_P5 = f['p16_P']
        p84_P5 = f['p84_P']

    with np.load(dir + "snaps" + f'{filename6}22k_256_1024_1_3_' + f'/KH_jointPDFprstemp_percentiles{W}_snapshot_' + f'{ni}to{nf}' + f'C{F}' + '.npz', 'r') as f:
        avg_P6 = f['avg_P']
        median_P6 = f['median_P']
        p16_P6 = f['p16_P']
        p84_P6 = f['p84_P']

    plt.figure(figsize=(14, 14))


if __name__ == "__main__":

    global path, n1, n2, jump
    #path = r"../../../Downloads/Trillium_data/snapsinvertedfiducial256_1024_longer2corrected/"
    path = r"../../../Downloads/Trillium_data/snapshotpeak_cutoff75_128_512/"
    n1 = 0
    n2 = 125
    jump = 1
    delU = 31.

    calculate_norm_factors()
    print(f"Normalization factor for log-normal lambda function: {norm_factor_normal_lambda} erg cm^3 s^-1")
    print(f"Normalization factor for constant lambda function: {norm_factor_const_lambda} erg cm^3 s^-1")
    print(f"Normalization factor for inverted log-normal lambda function: {norm_factor_inverted_lambda} erg cm^3 s^-1")
    #plot_cooling_functions()
    #plot_cooling_rate_comparison()


    plot_cooling_fn_paper()
    #plot_vturb_profiles("fidcool", "invnorm", "normlog", "const", "hotpeak", "nocool")



