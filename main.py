import utils
import model as m
import numpy as np
import os
from shutil import copyfile
import matplotlib.pyplot as plt


def plot_files(folder_name, time):
        arr_acc = []
        lst = os.listdir(folder_name)
        lst = [int(file.split('_')[-1][:-4]) for file in lst]
        indxes = np.argsort(lst)
        file_names =  os.listdir(folder_name)
        for i, indx in enumerate(indxes):
                file_name = os.path.join(folder_name,file_names[indx])
                V, out_spikes, in_spikes, _ = utils.get_csv_file_data(file_name)
                acc = utils.count_acc(in_spikes[0], out_spikes[:, 0], time_step/dt)
                print(acc)
                arr_acc.append(acc)
                range_t = np.arange(0, time[i], 0.1)
                utils.plot_spikes(1, 1, in_spikes*Iinj, out_spikes.T, V, range_t, V_th, E_L, legend = [f"{nu}", f"{time_step}", f"{rate}", f"{alfa}"])
        return arr_acc

def plot_file(file_name, time, acc:bool):
        V, out_spikes, in_spikes, _ = utils.get_csv_file_data(file_name)
        range_t = np.arange(0, time, 0.1)
        utils.plot_spikes(1, 1, in_spikes*Iinj, out_spikes.T, V, range_t, V_th, E_L, legend = [f"{nu}", f"{time_step}", f"{rate}", f"{alfa}"])
        if acc:
                accuracy = utils.count_acc(in_spikes[0], out_spikes[:, 0], time_step/dt)
                print(accuracy)
                return accuracy
        return None

def plot_folder(folder_name, time, acc:bool):
        V, out_spikes, in_spikes, _ = utils.get_csv_dir_data(folder_name)
        range_t = np.arange(0, time, 0.1)
        utils.plot_spikes(1, 1, in_spikes*Iinj, out_spikes.T, V, range_t, V_th, E_L, legend = [f"{nu}", f"{time_step}", f"{rate}", f"{alfa}"])
        if acc:
                accuracy = utils.count_acc(in_spikes[0], out_spikes[:, 0], time_step/dt)
                print(accuracy)
                return accuracy
        return None

def make_new_data(epochs):
        start_V = E_L
        i = 0 
        layer1 = m.Layer(1, 1, start_V, neuron)
        snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, time_step=time_step, train=False, dVs=0, 
                    check=True, file_name = f'output_data/data_{int(Lt*dt)}_{i}.csv', period=i)
        in_spikes, out_spikes, out_rate, V = snn.forward()
        acc = utils.count_acc(in_spikes[0], out_spikes[:, 0], time_step/dt)
        #acc = 0
        print(f"{i} accuracy: {acc}")
        i+=1
        while i < epochs and acc < 0.83:
                V, _, _, dV = utils.get_csv_dir_data("output_data")
                start_V = V[-1, 0]
                layer1 = m.Layer(1, 1, start_V, neuron)
                #np.random.seed(12)
                snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, time_step=time_step, 
                        train=False, dVs=dV, check=True, file_name = f'output_data/data_{int(Lt*dt)}_{i}.csv', period=i)
                in_spikes, out_spikes, out_rate, V = snn.forward()
                acc = utils.count_acc(in_spikes[0], out_spikes[:, 0], time_step/dt)
                print(f"{i} accuracy: {acc}")
                i+=1

def add_data(epochs, i):
        acc = 0
        while i < epochs and acc < 0.8:
                V, _, _, dV = utils.get_csv_dir_data("output_data")
                start_V = V[-1, 0]
                layer1 = m.Layer(1, 1, start_V, neuron)
                #np.random.seed(12)
                snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, time_step=time_step, 
                        train=False, dVs=dV, check=True, file_name = f'output_data/data_{int(Lt*dt)}_{i}.csv', period = i)
                in_spikes, out_spikes, out_rate, V = snn.forward()
                acc = utils.count_acc(in_spikes[0], out_spikes[:, 0], time_step/dt)
                print(f"{i} accuracy: {acc}")
                i+=1

def instant_view():
        #np.random.seed(1)
        layer1 = m.Layer(1, 1, start_V, neuron)
        snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, time_step=time_step, train=False, 
                    dVs=0, check=False, file_name = f'output_data/data_{int(Lt*dt)}_{0}.csv', period=0)
        in_spikes, out_spikes, out_rate, V = snn.forward()
        utils.plot_spikes(1, 1, in_spikes*Iinj, out_spikes.T, V, range_t, V_th, E_L, legend = [f"{nu}", f"{time_step}", f"{rate}", f"{alfa}"])
        #accuracy = utils.count_acc(in_spikes[0], out_spikes[:, 0], time_step/dt)
        #print(accuracy)

pars = {'V_th': -50.0, 'V_reset': -70.0, 'g_L': 25.0, 'C_m': 500,  'V_init': -70.0, 'E_L': -70.0, 'tref': 5.0, 'dt': 0.1,
        'tm':5, 'tp':3, 'Ap':0.6, 'Am':-0.3, 'range_t': np.arange(0, 1000, 0.1), 'Iinj': 1.5, 'lr': 3, 'stdp_rate': 0.0625}

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
time_step = pars['dt']*60
nu = 1
rate = 0.2
alfa = 0.6
input = np.array([1])
neuron = m.Fractional_LIF(V_th, V_reset, E_L, Iinj, g_L, C_m, dt, alfa, tref, stdp_rate, tm, tp, Am, Ap)



#make_new_data(20)
#add_data(11, 10)
plot_file("output_data/data_1000_10.csv", 1000, True)
#plot_folder("plot", 2000, True)
#instant_view()



'''
#dV
c = 0
while c<2:
        V, _, _, dV = utils.get_csv_data("plot")
        start_V = V[-1, 0]
        np.random.seed(12)
        layer1 = m.Layer(1, 1, start_V, neuron)
        snn = m.SNN([layer1], input, Lt, 1, range_t, rate=rate, nu=nu, time_step=time_step, 
                    train=False, dVs=dV, check=True, file_name = f'exp3/data_{int(Lt*dt)}_1_{c}.csv')
        in_spikes, out_spikes, out_rate, V = snn.forward()
        os.remove(f"plot/data_200_{c}.csv")
        c+=1
        print(c)

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


