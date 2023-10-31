import utils
import model as m
import numpy as np
import matplotlib.pyplot as plt

pars = {'V_th': -69.0, 'V_reset': -20, 'tau_m': 10.0, 'g_L': 10.0, 'V_init': -75.0, 'E_L': -75.0, 'tref': 5.0, 'T': 400.0, 
        'dt': 0.1, 'tm':5, 'tp':3, 'Ap':0.6, 'Am':-0.3, 'range_t': np.arange(0, 400, 0.1), 'Vinj': 3, 'lr': 1}

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
alfa_minus = np.round(np.exp(-dt/tm), 3)
alfa_plus = np.round(np.exp(-dt/tp), 3)

b = np.round(np.exp(-dt/tref), 3)
neuron = m.Fractional_LIF(V_th, V_reset, E_L, b, Vinj, tau_m, dt, 0.8)
#neuron = m.LIF(V_th, V_reset, E_L, b, Vinj)

input = np.random.rand(2)
layer1 = m.Layer(2, 1, neuron, 0.0625, Am, Ap, alfa_minus, alfa_plus)
#print(layer1.weights)
#layer2 = m.Layer(2, 1, neuron, 0.0625, Am, Ap, alfa_minus, alfa_plus)
snn = m.SNN([layer1], input, Lt, 1, range_t, rate=500)
in_spikes, out_spikes, out_rate, V = snn.inference()

print(f"rates of the output neurons: {out_rate}")
#print(snn.layers[0].weights)
utils.plot_spikes(2, 1, in_spikes, out_spikes, V, range_t, V_th, E_L)
plt.show()