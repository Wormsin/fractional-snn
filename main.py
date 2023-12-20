import utils
import model as m
import numpy as np
import matplotlib.pyplot as plt


pars = {'V_th': -50.0, 'V_reset': -20, 'tau_m': 10, 'g_L': 10.0, 'V_init': -70.0, 'E_L': -70.0, 'tref': 5.0, 'T': 400.0, 
        'dt': 0.1, 'tm':5, 'tp':3, 'Ap':0.6, 'Am':-0.3, 'range_t': np.arange(0, 400, 0.1), 'Vinj': 2, 'lr': 1}

V_th, V_reset = pars['V_th'], pars['V_reset']
tau_m, g_L = pars['tau_m'], pars['g_L']
V_init, E_L = pars['V_init'], pars['E_L']
dt, range_t = pars['dt'], pars['range_t']
tref = pars['tref']
Lt = range_t.size
Vinj = pars['Vinj']
tm = pars['tm']
tp = pars['tp']
lr = pars['lr']
Am = pars['Am']
Ap = pars['Ap']

nu = 0.6
rate = 2
alfa = 1
np.random.seed(12)
input = np.array([1, 1, 1])
neuron = m.Fractional_LIF(V_th, V_reset, E_L, Vinj, tau_m, dt, alfa, tref, 0.0625, tm, tp, Am, Ap)
layer1 = m.Layer(3, 1, neuron)
snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, train=False)
in_spikes, out_spikes, out_rate, V = snn.forward()

utils.plot_spikes(3, 1, in_spikes, out_spikes.T, V, range_t, V_th, E_L)
plt.show()