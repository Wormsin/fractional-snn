import numpy as np
import matplotlib.pyplot as plt

from model import neuron, layer, fc_model, train, eval
from data_proc import encoding

pars = {'V_th': -50.0, 'V_reset': -70.0, 'g_L': 25.0, 'C_m': 500,  'V_init': -70.0, 'E_L': -70.0, 'tref': 5.0, 'dt': 0.1,
        'tm':5, 'tp':3, 'Ap':0.6, 'Am':-0.3, 'T': 1000, 'Iinj': 3, 'stdp_rate': 0.0625}

V_th, V_reset = pars['V_th'], pars['V_reset']
g_L, C_m = pars['g_L'], pars['C_m']
V_init, E_L = pars['V_init'], pars['E_L']
dt, T = pars['dt'], pars['T']
tref = pars['tref']
Iinj = pars['Iinj']
tm = pars['tm']
tp = pars['tp']
Am = pars['Am']
Ap = pars['Ap']
stdp_rate = pars['stdp_rate']

start_V = E_L
time_step = 0.1 # ms
alfa = 0.7
N = T//(time_step)


neuron_flif = neuron.Fractional_LIF(V_th, V_reset, start_V, Iinj, g_L, C_m, dt, alfa, tref)
layer1 = layer.Layer(9, 2, start_V, neuron_flif)
layer2 = layer.Layer(3, 2, start_V, neuron_flif)
snn = fc_model.FC([layer1], T, dt)

data_train= 'data_proc/dataset/train/'
data_test = 'data_proc/dataset/test/'
train.train(model=snn, epochs=1, data_dir=data_train, N_spk=N, dt=dt, t_step=time_step)
#eval.eval(model=snn, weights_dir='weights', data_dir=data_test, N_spk=N, dt=dt, t_step=time_step, V_th=V_th, V_rest=V_reset, alfa=alfa)

