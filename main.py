import utils
import model as m
import numpy as np
import matplotlib.pyplot as plt


pars = {'V_th': -50.0, 'V_reset': -70.0, 'g_L': 25.0, 'C_m': 500,  'V_init': -70.0, 'E_L': -70.0, 'tref': 5.0, 'dt': 0.1,
        'tm':5, 'tp':3, 'Ap':0.6, 'Am':-0.3, 'range_t': np.arange(0, 1000, 0.1), 'Iinj': 3, 'lr': 1, 'stdp_rate': 0.0625}

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


V, out_spikes, in_spikes, dV = utils.get_csv_data("output_data")

#start_V = V[-1, 0]
start_V = E_L
time_step = pars['dt']*1
nu = 1
rate = 3
alfa = 0.2
'''
np.random.seed(12)
input = np.array([1])
neuron = m.Fractional_LIF(V_th, V_reset, E_L, Iinj, g_L, C_m, dt, alfa, tref, stdp_rate, tm, tp, Am, Ap)
layer1 = m.Layer(1, 1, start_V, neuron)
snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, time_step=time_step, train=False, dVs=0, check=True)
in_spikes, out_spikes, out_rate, V = snn.forward()
'''

utils.plot_spikes(1, 1, in_spikes*Iinj, out_spikes.T, V, range_t, V_th, E_L, legend = [f"{nu}", f"{time_step}", f"{rate}", f"{alfa}"])

#plot from csv file
#V, out_spikes, in_spikes, dV = utils.get_csv_data("data_500.csv")
#utils.plot_spikes(1, 1, in_spikes*Iinj, out_spikes.T, V, range_t, V_th, E_L, legend = [f"{nu}", f"{time_step}", f"{rate}", f"{alfa}"])

'''
time_step = 0.1
nu = 1
rate = 0.02
alfa = 1
Iinj = 0.2
np.random.seed(3)
input = np.array([1])
neuron = m.Fractional_LIF(V_th, V_reset, E_L, Iinj, g_L, C_m, dt, alfa, tref, stdp_rate, tm, tp, Am, Ap)
layer1 = m.Layer(1, 1, neuron)
snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, time_step=time_step, train=False)
in_spikes, out_spikes, out_rate, V2 = snn.forward()


#utils.plot_spikes(1, 1, in_spikes*Iinj, out_spikes.T, V, range_t, V_th, E_L, legend = [f"{nu}", f"{time_step}", f"{rate}", f"{alfa}"])
V1 = V1.T[0]
plt.plot(range_t[V1<=-69.9], V1[V1<=-69.9])
plt.plot(range_t, V2)
plt.legend(["alfa=0.2", 'alfa=1'])
'''

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