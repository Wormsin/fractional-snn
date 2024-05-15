import numpy as np
import matplotlib.pyplot as plt

from model import neuron, layer, fc_model, train, eval
from data_proc.encoding import get_mnist, flick_mnist
from features import utils
from scipy.optimize import curve_fit

pars = {'V_th': -50.0, 'V_reset': -70.0, 'g_L': 25.0, 'C_m': 500,  'V_init': -70.0, 'E_L': -70.0, 'tref': 5.0, 'dt': 0.1,
        'tm':5, 'tp':3, 'Ap':0.6, 'Am':-0.3, 'T': 2000, 'Iinj': 3, 'stdp_rate': 0.0625}

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


neuron_flif = neuron.Fractional_LIF(V_th, V_reset, start_V, Iinj, g_L, C_m, dt, alfa, tref)
neuron_lif  = neuron.Fractional_LIF(V_th, V_reset, start_V, Iinj, g_L, C_m, dt, 1, tref)
layer1 = layer.Layer(14*14, 85, start_V, neuron_lif)
layer2 = layer.Layer(85, 1, start_V, neuron_lif)
snn = fc_model.FC([layer1, layer2], T, dt, True)

data_train= 'data_proc/dataset/train/'
data_test = 'data_proc/dataset/test/'
#train.train(model=snn, epochs=1, data_dir=data_train, dt=dt, t_step=time_step, V_th=V_th, V_rest=V_reset, alfa=alfa, plot=True)
#eval.eval(model=snn, weights_dir='weights1', data_dir=data_test, dt=dt, t_step=time_step, V_th=V_th, V_rest=V_reset, alfa=alfa)


#data = 'data_proc/img0.png'
#flick_mnist(data, int(T/dt), dt, time_step)

'''
ax = utils.a4_plot()
y1 =  utils.ISI_plot('lif/9/plot19.csv', 'r',  ax, 2000, 0.5, 0, 0.5)
y1 = utils.ISI_plot('lif/9/plot29.csv','r',  ax, 2000, 0.5, 0, 0.5) +y1
y1 = utils.ISI_plot('lif/9/plot39.csv','r',  ax, 2000, 0.5, 0, 0.5) +y1
y2 = utils.ISI_plot('lif/5/plot15.csv', 'b',  ax, 2000, 0.5, 0, 0.5)
y2  = utils.ISI_plot('lif/5/plot25.csv','b',  ax, 2000, 0.5, 0, 0.5) +y2
y2  = utils.ISI_plot('lif/5/plot35.csv','b',  ax, 2000, 0.5, 0, 0.5) +y2
x1 = np.arange(0, 2000, 0.5)
x2 = np.arange(0, 2000,0.5)
y1 = y1[1:]
x1, y1 = x1[y1>0], y1[y1>0]
y2 = y2[1:]
x2, y2 = x2[y2>0], y2[y2>0]
ax.plot(x1, y1, marker ='o',markersize = 8, linewidth = 0, color = 'r', markeredgewidth=0, alpha=0.5, label = f'{9}')
ax.plot(x2, y2, marker ='o',markersize = 8, linewidth = 0, color = 'b', markeredgewidth=0, alpha=0.5,  label = f'{5}')
params, covariance = curve_fit(lambda x, s, nu: s*(x**(-1-nu)), x1, y1, [2000, 0.5])
fit = 250*(x1**(-1-0.7))
#ax.plot(x1, fit, linewidth = 3, label = f'{9} beta = {0.7}', color = 'r')
params, covariance = curve_fit(lambda x, s, nu: s*(x**(-1-nu)), np.concatenate((x2, x1)), np.concatenate((y2, y1)), [250, 0.9])
fit = 40*(x2**(-1-0.9))
ax.plot(x2, fit, linewidth = 3, label = f'{5} beta = {0.9}', color = 'b')
ax.set_title('LIF',fontweight = 'bold', size = 12 )
ax.set_yscale('log')
#ax.set_xscale('log')
ax.set_xlabel('ISI, ms', fontweight = 'bold', size = 10)
ax.set_ylabel('ISI number', fontweight = 'bold', size = 10)
ax.grid(True,which='major',axis='both',alpha=0.3)
plt.legend()
plt.show()
'''