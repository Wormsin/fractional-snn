import utils
import model as m
import numpy as np
import matplotlib.pyplot as plt
import time 


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
#plt.plot(range_t, V2)
#plt.legend()
plt.show()


'''
alfa_arr = [1, 0.3]
nu_arr = np.round(np.arange(0.2, 1, 0.2), 1)
rate = 2
isi_mean = np.empty((2, 4))
K=150
total_spikes = np.array([])
num_spikes = np.empty((2, 4))

for i, alfa in enumerate(alfa_arr):
        for j, nu in enumerate(nu_arr):
                sum_spikes = 0
                total_spikes = np.array([])
                k=0
                while k<K:
                        input = np.random.rand(1)
                        neuron = m.Fractional_LIF(V_th, V_reset, E_L, Vinj, tau_m, dt, alfa, tref, 0.0625, tm, tp, Am, Ap)
                        layer1 = m.Layer(1, 1, neuron)
                        snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, train=False)
                        in_spikes, out_spikes, out_rate, V = snn.forward()
                        spikes = range_t[out_spikes.T[0]==1] + 400*k
                        total_spikes = np.concatenate((total_spikes, spikes))
                        sum_spikes+=np.sum(out_spikes)
                        k+=1
                        print(k)
                isi = utils.count_ISI(total_spikes)        
                mean = np.sum(isi)/len(isi)
                isi_mean[i, j] = mean
                num_spikes[i, j] = sum_spikes
       
utils.plot_means(isi_mean, ['alpha = 1', 'alpha = 0.3'], nu_arr.astype('str'), 'nu', 'ISI mean value')
plt.figure(1)
utils.plot_means(num_spikes, ['alpha = 1', 'alpha = 0.3'], nu_arr.astype('str'), 'nu', 'number of spikes per minut')
plt.show()    

'''
'''
rate = 2
N = 2000
alfa_arr = np.array([1, 0.3])
nu = 0.6
ISI = []
np.random.seed(12)
for i, alfa in enumerate(alfa_arr):
        sum_spikes = 0
        isi = np.array([])
        k=0
        while sum_spikes<N:
                input = np.random.rand(1)
                neuron = m.Fractional_LIF(V_th, V_reset, E_L, Vinj, tau_m, dt, alfa, tref, 0.0625, tm, tp, Am, Ap)
                layer1 = m.Layer(1, 1, neuron)
                snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, train=False)
                in_spikes, out_spikes, out_rate, V = snn.forward()
                sum_spikes+=np.sum(out_spikes)
                spikes = range_t[out_spikes.T[0]==1] + 400*k
                isi = np.concatenate((isi, spikes))
                k+=1
                print(sum_spikes, alfa)
        ISI.append(utils.count_ISI(isi))

for i in range(len(alfa_arr)):
        hist, bins = np.histogram(np.log10(ISI[i]), bins=100, density=True)
        x = np.array([(bins[i+1]+bins[i])/2 for i in range(len(bins)-1)])
        xerrors = np.array([(bins[i+1]-bins[i])/2 for i in range(len(bins)-1)])
        yerrors = np.sqrt(hist)/2
        #y = np.log10(hist[hist!=0])
        plt.errorbar(x, hist, xerr = xerrors, yerr=yerrors,  fmt = 'o',
         elinewidth = 3, capsize=5, label = f"alfa= {np.round(alfa_arr[i], 1)}")
        #plt.plot(x, y)
        #plt.plot(x, y)
        print(np.sum(ISI[i])/len(ISI[i]))
'''
'''
hist, bins = np.histogram(ISI[0])
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.hist(ISI[0], bins=logbins, rwidth=0.8, label="alfa = 1")
hist1, bins1 = np.histogram(ISI[1])
logbins1 = np.logspace(np.log10(bins1[0]),np.log10(bins1[-1]),len(bins1))
plt.hist(ISI[1], bins=logbins1, rwidth=0.8, label="alfa = 0.3", alpha=0.5)
plt.xscale('log')
'''
'''
plt.title(f'nu = {nu}, rate = {rate}')
plt.xlabel('log(ISI) ms', fontweight ='bold', fontsize = 15) 
plt.ylabel('density of ISI\'s number', fontweight ='bold', fontsize = 15) 
plt.legend()
plt.show()
'''
