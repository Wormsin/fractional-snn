import utils
import func
import model as m
import numpy as np
import os
from shutil import copyfile
import matplotlib.pyplot as plt
from scipy.special import gamma 




pars = {'V_th': -50.0, 'V_reset': -70.0, 'g_L': 25.0, 'C_m': 500,  'V_init': -70.0, 'E_L': -70.0, 'tref': 5.0, 'dt': 0.1,
        'tm':5, 'tp':3, 'Ap':0.6, 'Am':-0.3, 'range_t': np.arange(0, 500, 0.1), 'Iinj': 3, 'lr': 3, 'stdp_rate': 0.0625}

V_th, V_reset = pars['V_th'], pars['V_reset']
g_L, C_m = pars['g_L'], pars['C_m']
V_init, E_L = pars['V_init'], pars['E_L']
dt, range_t = pars['dt'], pars['range_t']
tref = pars['tref']
Lt = range_t.size
Iinj = pars['Iinj']
tm = pars['tm']
tp = pars['tp']
lr = pars['lr']
Am = pars['Am']
Ap = pars['Ap']
stdp_rate = pars['stdp_rate']

start_V = E_L
time_step = pars['dt']*30
num_inputs = 2
nu = np.array([0.5, 1])
alfa = 0.5
N = Lt//(20)
input = np.array([1, 1])
neuron = m.Fractional_LIF(V_th, V_reset, E_L, Iinj, g_L, C_m, dt, alfa, tref, stdp_rate, tm, tp, Am, Ap)
proc = func.Process(V_th, E_L, dt, Lt, range_t, N, nu, time_step, Iinj, alfa, num_inputs, input, neuron)

#proc.instant_view()

#proc.make_new_data(3)
#proc.add_data(5, 3)

#proc.plot_file("output_data/data_1.csv", False)
#proc.plot_folder("output_data", False)
#proc.plot_files('output_data')

'''
ax = utils.a4_plot()
#ax = utils.ISI_plot(ax, "output_data", 4, 3000)
#ax = utils.ISI_plot(ax, "08_02_1", 4, 1500, 0.5)
utils.ISI_plot(ax, "plot", 8, 1500, 1)
plt.legend()
plt.show()
'''