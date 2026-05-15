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
    mean = 5.0  # log10(10^5 K)
    sigma = 0.2  # width of the distribution
    logt = math.log10(temp)
    if (temp < 1.05e4) or (temp > 0.95e6):
        return 0.0
    lambda_func = math.exp(-0.5*((logt - mean)/sigma)**2)
    
    return norm_factor_normal_lambda * lambda_func

def constant_cool_fn100less(temp):
    # return a constant cooling rate 
    if (temp < 1.05e4) or (temp > 0.95e6):
        return 0.0

    return norm_factor_const_lambda/100.

def inverted_log_norm_cool_fn(temp):
    # generate an inverted log-normal distribution centered around 10^5 K with
    # a width such that area under curve is same as area under ISM cooling function 
    # from 10^4 K to 10^6 K

    # just reflect the log-normal distribution about y = const_lambda_func
    if (temp < 1.05e4) or (temp > 0.95e6):
        return 0.0
    new_lambda = math.exp(0.5*((math.log10(temp) - 5.0)/0.2)**2)
    #new_lambda = 2*constant_cool_fn(temp) - log_norm_cool_fn(temp)

    return new_lambda * norm_factor_inverted_lambda

def plot_cooling_functions():
    temps = np.linspace(1e4, 1e6, 1000)
    calculate_norm_factors()
    cooling_rates_stock = [ISMCoolFn_stock(temp) for temp in temps]
    cooling_rates_norm = [log_norm_cool_fn(temp) for temp in temps]
    cooling_rates_const = [constant_cool_fn100less(temp) for temp in temps]
    cooling_rates_inverted = [inverted_log_norm_cool_fn(temp) for temp in temps]

    area_stock = integrate.simpson(cooling_rates_stock, temps)
    print(f"Area under the cooling function curve (stock): {area_stock:.2e} erg cm^3 s^-1 K")
    area_norm = integrate.simpson(cooling_rates_norm, temps)
    print(f"Area under the cooling function curve (log-normal): {area_norm:.2e} erg cm^3 s^-1 K with normalization factor {norm_factor_normal_lambda}")
    area_const = integrate.simpson(cooling_rates_const, temps)
    print(f"Area under the cooling function curve (constant): {area_const:.2e} erg cm^3 s^-1 K with normalization factor {norm_factor_const_lambda}")
    area_inverted = integrate.simpson(cooling_rates_inverted, temps)
    print(f"Area under the cooling function curve (inverted log-normal): {area_inverted:.2e} erg cm^3 s^-1 K with normalization factor {norm_factor_inverted_lambda}")

    plt.figure(figsize=(10, 6))
    plt.plot(temps, cooling_rates_stock, label='ISM Cooling Function', color='blue')
    plt.plot(temps, cooling_rates_norm, label='Log-Normal Approximation', color='red',)
    plt.plot(temps, cooling_rates_const, label='Constant Approximation', color='green',)
    plt.plot(temps, cooling_rates_inverted, label='Inverted Log-Normal', color='orange',)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Cooling Rate (erg cm^3 s^-1)')
    plt.title('ISM Cooling Function vs Temperature')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("cooling_function_comparison.png")

def plot_cooling_fn_paper():

    temps = np.logspace(4, 6, 1000)
    calculate_norm_factors()

    cooling_rates_stock = [ISMCoolFn_stock(temp) for temp in temps]
    cooling_rates_norm = [log_norm_cool_fn(temp) for temp in temps]
    cooling_rates_const = [constant_cool_fn100less(temp) for temp in temps]
    cooling_rates_inverted = [inverted_log_norm_cool_fn(temp) for temp in temps]

    Lambda_stock = np.vectorize(ISMCoolFn_stock)(temps)/COOLING_UNIT
    Lambda_norm = np.vectorize(log_norm_cool_fn)(temps)/COOLING_UNIT
    Lambda_const = np.vectorize(constant_cool_fn100less)(temps)/COOLING_UNIT
    Lambda_inverted = np.vectorize(inverted_log_norm_cool_fn)(temps)/COOLING_UNIT

    P_0_array = P_0 * np.ones_like(temps)
    cooling_time_inverted_lambda = np.divide(5. * (temps/TEMPERATURE)**2 , 2*P_0_array * Lambda_inverted, out=np.full_like(temps, math.inf, dtype=float), 
                    where=Lambda_inverted != 0)
    cooling_time_stock = np.divide(5. * (temps/TEMPERATURE)**2 , 2*P_0_array * Lambda_stock, out=np.full_like(temps, math.inf, dtype=float),
                    where=Lambda_stock != 0)
    cooling_time_norm_lambda = np.divide(5. * (temps/TEMPERATURE)**2 , 2*P_0_array * Lambda_norm, out=np.full_like(temps, math.inf, dtype=float),
                    where=Lambda_norm != 0)
    cooling_time_const_lambda = np.divide(5. * (temps/TEMPERATURE)**2 , 2*P_0_array * Lambda_const, out=np.full_like(temps, math.inf, dtype=float),
                    where=Lambda_const != 0)


    area_stock = integrate.simpson(cooling_rates_stock, temps)
    print(f"Area under the cooling function curve (stock): {area_stock:.2e} erg cm^3 s^-1 K")
    area_norm = integrate.simpson(cooling_rates_norm, temps)
    print(f"Area under the cooling function curve (log-normal): {area_norm:.2e} erg cm^3 s^-1 K with normalization factor {norm_factor_normal_lambda}")
    area_const = integrate.simpson(cooling_rates_const, temps)
    print(f"Area under the cooling function curve (constant): {area_const:.2e} erg cm^3 s^-1 K with normalization factor {norm_factor_const_lambda}")
    area_inverted = integrate.simpson(cooling_rates_inverted, temps)
    print(f"Area under the cooling function curve (inverted log-normal): {area_inverted:.2e} erg cm^3 s^-1 K with normalization factor {norm_factor_inverted_lambda}")

    plt.figure(figsize=(6, 11))

    plt.subplot(2,1,1)
    plt.plot(temps, cooling_rates_norm, label=r"$\log(\Lambda) \propto \exp (-\frac{(\log T_0 - \log T)^2}{2\sigma^2})$", color='red')
    plt.plot(temps, cooling_rates_const, label=r"$\Lambda = const$", color='green')
    plt.plot(temps, cooling_rates_inverted, label=r"$\log(\Lambda) \propto \exp (\frac{(\log T_0 - \log T)^2}{2\sigma^2})$*", color='orange')
    plt.plot(temps, cooling_rates_stock, label='Schure et al. (2009)', color='blue')
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(1e-28, 2e-20)
    plt.xlabel(r"T (K)", fontsize=14)
    plt.ylabel(r"Cooling Rate $ \Lambda \ (erg\ cm^3\ s^{-1})$", fontsize=14)
    plt.grid(which="both", axis='both', ls="--")
    plt.legend(loc='lower right', fontsize=12)

    plt.subplot(2,1,2)
    plt.ylim(1e-5, 1e4)
    plt.plot(temps, cooling_time_norm_lambda, label=r"$\log(\Lambda) \propto \exp (-\frac{(\log T_0 - \log T)^2}{2\sigma^2})$", color='red')
    plt.plot(temps, cooling_time_const_lambda,label=r"$\Lambda = const$", color='green')
    plt.plot(temps, cooling_time_inverted_lambda, label=r"$\log(\Lambda) \propto \exp (\frac{(\log T_0 - \log T)^2}{2\sigma^2})$*", color='orange')
    plt.plot(temps, cooling_time_stock, label='Schure et al. (2009)', color='blue')
    plt.xlabel(r"T (K)", fontsize=14)
    plt.ylim(1e-5,1e4)
    plt.ylabel(r'$t_{cool} = \frac{5/2 \, P}{\left(\frac{P}{k_B T}\right)^2 \Lambda (T)}$ (Myr)', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which="both", axis='both', ls="--")    
    plt.legend(loc='lower right', fontsize=12)

    plt.tight_layout()
    plt.savefig("cooling_function_and_times_comparison.png", dpi=300)

def plot_cooling_times_tempspace():
    temperatures = np.logspace(4, 6, 1000)

    Lambda_stock = np.vectorize(ISMCoolFn_stock)(temperatures)/COOLING_UNIT
    Lambda_norm = np.vectorize(log_norm_cool_fn)(temperatures)/COOLING_UNIT
    Lambda_const = np.vectorize(constant_cool_fn100less)(temperatures)/COOLING_UNIT
    Lambda_inverted = np.vectorize(inverted_log_norm_cool_fn)(temperatures)/COOLING_UNIT

    P_0_array = P_0 * np.ones_like(temperatures)
    cooling_time_inverted_lambda = np.divide(5. * (temperatures/TEMPERATURE)**2 , 2*P_0_array * Lambda_inverted, out=np.full_like(temperatures, math.inf, dtype=float), 
                    where=Lambda_inverted != 0)
    cooling_time_stock = np.divide(5. * (temperatures/TEMPERATURE)**2 , 2*P_0_array * Lambda_stock, out=np.full_like(temperatures, math.inf, dtype=float),
                    where=Lambda_stock != 0)
    cooling_time_norm_lambda = np.divide(5. * (temperatures/TEMPERATURE)**2 , 2*P_0_array * Lambda_norm, out=np.full_like(temperatures, math.inf, dtype=float),
                    where=Lambda_norm != 0)
    cooling_time_const_lambda = np.divide(5. * (temperatures/TEMPERATURE)**2 , 2*P_0_array * Lambda_const, out=np.full_like(temperatures, math.inf, dtype=float),
                    where=Lambda_const != 0)
    plt.figure(figsize=(10, 6))
    plt.ylim(1e-5, 1e4)
    plt.plot(temperatures, cooling_time_inverted_lambda, label='Cooling Time with Inverted Log-Normal Lambda', color='purple')
    plt.plot(temperatures, cooling_time_stock, label='Cooling Time with Stock ISM Cooling Function', color='blue')
    plt.plot(temperatures, cooling_time_norm_lambda, label='Cooling Time with Log-Normal Lambda', color='red')
    plt.plot(temperatures, cooling_time_const_lambda, label='Cooling Time with Constant Lambda', color='green')
    plt.xlabel('Temperature (K)')
    plt.xlim(1e4, 1e6)
    plt.ylabel(r'Cooling Time (Myr)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title(r'Isobaric Cooling time = $\frac{5/2 \, P}{\left(\frac{P}{k_B T}\right)^2 \Lambda (T)}$ profiles across temeprature space for different cooling functions')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.savefig("cooling_times_tempspace.png", dpi=300)

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
    plt.plot(Sigma_dot_coolfromhstdict['time'][::2], Sigma_dot_cool_flux, label=r'Cooling Rate by average and integrating $\int_{bottom}^{top} \left<n^2 \Lambda (T)\right>\, dz$ using fluxes from python scripts', color='red')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Time (Myr)')
    plt.yscale('log')
    plt.xlabel("Temperature (K)")
    plt.ylabel('Cooling Rate (code units)')
    plt.legend()
    plt.savefig(path + "cooling_rate_comparison.png", dpi = 150)

def plot_quantity_from_hst(quantity_name):
    import athena_read

    hst_dict = athena_read.hst(path + 'KH.hydro.hst')
    plt.figure(figsize=(10, 6))
    plt.plot(hst_dict['time'], hst_dict[quantity_name]/100., label=f'{quantity_name} from history file', color='blue')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Time (Myr)')
    plt.ylabel(f'{quantity_name} (code units)')
    plt.legend()
    plt.savefig(path + f"{quantity_name}.png", dpi = 150)

def plot_quantity_from_hst():
    import athena_read

    hst_dict = athena_read.hst(path + 'KH.hydro.hst')
    tot_E = hst_dict['tot-E']/100.
    times = hst_dict['time']
    grad_totE = np.gradient(tot_E, times)
    E_dot = hst_dict['Edot_total']/100.
    E_dot_integ = np.cumsum(E_dot)*np.mean(np.diff(times))
    plt.figure(figsize=(10, 6))
    plt.plot(times, grad_totE, label=f'Total Energy from history file', color='blue')
    plt.plot(times, E_dot, label=f'E_dot_cool_integrated', color='red')
    plt.plot(times, grad_totE + E_dot, label=f'Energy balance check', color='green')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Time (Myr)')
    plt.ylabel(f'Energy balance check (code units)')
    plt.xlim(0.0,20.0)
    plt.ylim(-1000,1000)
    plt.legend()
    plt.savefig(path + f"energy_balance.png", dpi = 150)

def plot_tot_E():
   
    with np.load(path + f'KH_1D_arrays_time_averaged{n1}to{n2}with{jump}.npz', 'r') as f:
        energy_density_integrated = f['energy_density_integrated']

    import athena_read

    hst_dict = athena_read.hst(path + 'KH.hydro.hst')
    tot_E = hst_dict['tot-E']/100.
    times = hst_dict['time']

    # smoothen the total energy curve to get a better estimate of its derivative using ftt
    import scipy.fft
    E_dct = scipy.fft.dct(tot_E)
    # set the high frequency components to zero to smoothen the curve
    cutoff = 30  # adjust this cutoff to control the amount of smoothing
    E_dct[cutoff:] = np.zeros_like(E_dct[cutoff:])
    smoothened_totE = scipy.fft.idct(E_dct)

    grad = np.gradient(smoothened_totE, times)

    plt.figure(figsize=(10, 6))
    #plt.plot(times, tot_E/100., label=f'Total Energy from history file', color='blue')
    #plt.plot(times[::2], energy_density_integrated, label=f'Total Energy by integrating energy density across z-axis using python scripts', color='black')
    plt.plot(times, grad, label=f'dE/dt (smoothed) from history file', color='red')
    plt.grid(True, which="both", ls="--")
    plt.xlabel('Time (Myr)')
    plt.ylabel(f'Total Energy (code units)')
    plt.xlim(2.5,20.0)
    plt.legend()
    plt.savefig(path + f"tot_E_grad.png", dpi = 150)
    print(np.mean(grad))

if __name__ == "__main__":

    global path, n1, n2, jump
    #path = r"../../../Downloads/Trillium_data/snapsinvertedfiducial256_1024_longer2corrected/"
    path = r"../../../Downloads/Trillium_data/snapsinvfid4000less256_1024/"
    n1 = 0
    n2 = 125
    jump = 1

    calculate_norm_factors()
    print(f"Normalization factor for log-normal lambda function: {norm_factor_normal_lambda} erg cm^3 s^-1")
    print(f"Normalization factor for constant lambda function: {norm_factor_const_lambda} erg cm^3 s^-1")
    print(f"Normalization factor for inverted log-normal lambda function: {norm_factor_inverted_lambda} erg cm^3 s^-1")
    plot_cooling_functions()
    plot_cooling_times_tempspace()
    plot_cooling_rate_comparison()


    plot_cooling_fn_paper()



