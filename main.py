import utils
import model as m
import numpy as np
import matplotlib.pyplot as plt


pars = {'V_th': -50.0, 'V_reset': -70.0, 'g_L': 25.0, 'C_m': 500,  'V_init': -70.0, 'E_L': -70.0, 'tref': 5.0, 'dt': 0.1,
        'tm':5, 'tp':3, 'Ap':0.6, 'Am':-0.3, 'range_t': np.arange(0, 1000, 0.1), 'Iinj': 2, 'lr': 1, 'stdp_rate': 0.0625}

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

time_step = 200
nu = 1
rate = 0.01
alfa = 0.2
np.random.seed(3)
input = np.array([1])
neuron = m.Fractional_LIF(V_th, V_reset, E_L, Iinj, g_L, C_m, dt, alfa, tref, stdp_rate, tm, tp, Am, Ap)
layer1 = m.Layer(1, 1, neuron)
snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, time_step=time_step, train=False)
in_spikes, out_spikes, out_rate, V1 = snn.forward()


utils.plot_spikes(1, 1, in_spikes*Iinj, out_spikes.T, V, range_t, V_th, E_L, legend = [f"{nu}", f"{time_step}", f"{rate}", f"{alfa}"])
plt.show()


'''
alfa_arr = [1, 0.6, 0.4, 0.2]
nu = 1
rate = 2
ISI = []

for i, alfa in enumerate(alfa_arr):
        input = np.array([1])
        neuron = m.Fractional_LIF(V_th, V_reset, E_L, Iinj, g_L, C_m, dt, alfa, tref, stdp_rate, tm, tp, Am, Ap)
        layer1 = m.Layer(1, 1, neuron)
        snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, train=False)
        in_spikes, out_spikes, out_rate, V = snn.forward()

        spikes_time = range_t[out_spikes.T[0]==1]
        isi = utils.count_ISI(spikes_time)
        ISI.append(isi)

for j in range(len(alfa_arr)):
        hist, bins = np.histogram(np.log10(ISI[j]), bins = 50)
        x = np.array([(bins[i+1]+bins[i])/2 for i in range(len(bins)-1)])
        widths = np.array([bins[i+1]-bins[i] for i in range(len(bins)-1)])
        widths = widths[hist!=0]
        y = np.log10(hist[hist!=0]/(widths))
        x = x[hist!=0]
        plt.bar(x, y, label = f"alfa= {np.round(alfa_arr[j], 1)}", width =widths)
        break

plt.legend()
plt.show()   
'''