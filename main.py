import utils
import model as m
import numpy as np
import os


pars = {'V_th': -50.0, 'V_reset': -70.0, 'g_L': 25.0, 'C_m': 500,  'V_init': -70.0, 'E_L': -70.0, 'tref': 5.0, 'dt': 0.1,
        'tm':5, 'tp':3, 'Ap':0.6, 'Am':-0.3, 'range_t': np.arange(0, 500, 0.1), 'Iinj': 3.5, 'lr': 1, 'stdp_rate': 0.0625}

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

#V, out_spikes, in_spikes, dV = utils.get_csv_data("output_data")
#start_V = V[-1, 0]

start_V = E_L
time_step = pars['dt']*600
nu = 1
rate = 0.005
alfa = 0.2
'''
#dV
input = np.array([1])
neuron = m.Fractional_LIF(V_th, V_reset, E_L, Iinj, g_L, C_m, dt, alfa, tref, stdp_rate, tm, tp, Am, Ap)

V20, out_spikes20, _, _ = utils.get_csv_data("20")
out_spikes = np.copy(out_spikes20)
c = 2
while c<38:
        os.remove(f"plot/data_500_{c}.csv")
        V, _, _, dV = utils.get_csv_data("plot")
        start_V = V[-1, 0]
        np.random.seed(12)
        layer1 = m.Layer(1, 1, start_V, neuron)
        snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, time_step=time_step, train=False, dVs=dV, check=True, file_name = f'output_data/data_500_20_{c}.csv')
        in_spikes, out_spikes, out_rate, V = snn.forward()
        c+=1
        print(c)
'''
'''
#adaptation time test
i=0
snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, time_step=time_step, train=False, dVs=0, check=True, file_name = f'output_data/data_{int(Lt*dt)}_{i}.csv')
in_spikes, out_spikes, out_rate, V = snn.forward()
acc = utils.count_acc(in_spikes[0], out_spikes[:, 0], time_step/dt)
print(0, acc)
#utils.plot_spikes(1, 1, in_spikes*Iinj, out_spikes.T, V, range_t, V_th, E_L, legend = [f"{nu}", f"{time_step}", f"{rate}", f"{alfa}"])
i+=1
while acc < 0.85 and i<20:
        V, _, _, dV = utils.get_csv_data("output_data")
        start_V = V[-1, 0]
        layer1 = m.Layer(1, 1, start_V, neuron)
        snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, time_step=time_step, train=False, dVs=dV, check=True, file_name = f'output_data/data_{int(Lt*dt)}_{i}.csv')
        in_spikes, out_spikes, out_rate, V = snn.forward()
        acc = utils.count_acc(in_spikes[0], out_spikes[:, 0], time_step/dt)
        print(i, acc)
'''


#plot files
V, out_spikes, in_spikes, dV = utils.get_csv_data("plot")
print(utils.count_acc(in_spikes[0], out_spikes[:, 0], time_step/dt))
range_t = np.arange(0, 4500, 0.1)
utils.plot_spikes(1, 1, in_spikes*Iinj, out_spikes.T, V, range_t, V_th, E_L, legend = [f"{nu}", f"{time_step}", f"{rate}", f"{alfa}"])
