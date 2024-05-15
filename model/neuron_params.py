import numpy as np

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
nu = np.array([0.8])
alfa = 0.4
N = int(T*nu//((time_step)))
inputs = int(len(nu))