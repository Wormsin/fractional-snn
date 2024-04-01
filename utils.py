import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import os
from scipy.optimize import curve_fit

def get_csv_dir_data(dir):
    lst = os.listdir(dir)
    lst = [int(name.split('_')[-1][:-4]) for name in lst]
    indxes = np.argsort(lst)
    file_names = os.listdir(dir)
    for i, indx in enumerate(indxes):
        file_name = file_names[indx]
        with open(os.path.join(dir, file_name), 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
        data_array = np.array(data)
        prop = data_array[1:6, 0].astype(float)
        num_inputs = prop[-1]
        nu = data_array[1:int(num_inputs+1), 1].astype(float)
        weights = data_array[1:int(num_inputs+1), 2].astype(float)
        v, out_s, dv = data_array[1:, 3].astype(float), data_array[1:, 4].astype(float), data_array[1:, 5].astype(float)
        in_s = data_array[1:, 6:].astype(float)
        in_s = in_s.T
        if i ==0:
            V = v
            out_spikes = out_s
            in_spikes = in_s
            dV = dv[:-1]
        else:
            V = np.concatenate((V, v))
            out_spikes = np.concatenate((out_spikes, out_s))
            in_spikes = np.concatenate((in_spikes, in_s), axis = 1)
            dV = np.concatenate((dV, dv[:-1]))
    out_spikes = np.expand_dims(out_spikes, 1)
    V = np.expand_dims(V, 1)
    weights = np.expand_dims(weights, 1)
    prop[0]*=len(file_names)
    prop[2]*=len(file_names)
    return V, out_spikes, in_spikes, [dV], prop, nu, weights

def get_csv_file_data(file_name):
    with open(file_name, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
    data_array = np.array(data)
    prop = data_array[1:6, 0].astype(float)
    num_inputs = prop[-1]
    nu = data_array[1:int(num_inputs+1), 1].astype(float)
    weights = data_array[1:int(num_inputs+1), 2].astype(float)
    V, out_spikes, dV = data_array[1:, 3].astype(float), data_array[1:, 4].astype(float), data_array[1:, 5].astype(float)
    in_spikes = data_array[1:, 6:].astype(float)
    in_spikes = in_spikes.T
    out_spikes = np.expand_dims(out_spikes, 1)
    V = np.expand_dims(V, 1)
    weights = np.expand_dims(weights, 1)
    return V, out_spikes, in_spikes, [dV[:-1]], prop, nu, weights

def a4_plot():
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.1,0.1,0.5,0.8])
    return ax

def plot_spikes(in_features, out_features, in_spikes, out_spikes, V, range_t, V_th, V_rest, legend, title = ""):
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    fig, ax = plt.subplots(3, figsize=(10/1.5,6/1.5), gridspec_kw={'hspace':1})
    fig.subplots_adjust(right=0.87, left=0.07)
    for i in range(in_features):
        #ax[0].scatter(range_t[in_spikes[i]!=0], in_spikes[i][in_spikes[i]!=0]*(i+1),  s=0.8)
        #ax[0].vlines(range_t[in_spikes[i]!=0], 0+i, 1+i, color = (0.1, 0.2, 0.5*i))
        ax[0].plot(range_t, in_spikes[i]+i*np.max(in_spikes[i]))
    for i in range(out_features):
        #ax[1].scatter(range_t[out_spikes[i]!=0], out_spikes[i][out_spikes[i]!=0]*(i+1),  s=5)
        ax[1].vlines(range_t[out_spikes[i]!=0], 0+i, 1+i, color = (0.1, 0.5, 0.5*i))
    for i in range(out_features):
        ax[2].plot(range_t, V[:, i])
    ax[2].hlines(V_th, 0, range_t[-1], color = 'r')
    ax[2].hlines(V_rest, 0, range_t[-1], color = 'g')
    nu, step, rate, alfa, Iinj = legend
    leg_in = nu
    leg_out = [str(x) for x in range(1,out_features+1)]
    leg_out.append('Vth')
    leg_out.append('Vrest')
    if(in_features < 20):
        ax[0].legend(leg_in, loc='upper left', bbox_to_anchor=(0.99, 1.15),
          frameon = False)
        
    #ax[1].legend(leg_out, loc='upper left', bbox_to_anchor=(1, 0),  fancybox=True, shadow=True)
    ax[2].legend(leg_out, loc='upper left',bbox_to_anchor=(0.99, 1.15), frameon = False, )
    ax[0].set_xlim([-1, range_t[-1]+1])
    ax[1].set_xlim([-1, range_t[-1]+1])
    ax[2].set_xlim([-1, range_t[-1]+1])

    ax[0].get_yaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    plt.title(title)
    ax[0].set_title(f'input current = {Iinj}  nA, step = {step} ms, mean val. of spikes = {rate}', fontweight ='bold', fontsize = 11, loc = 'left')
    ax[1].set_title('output spikes', fontweight ='bold', fontsize = 11, loc = 'left')
    ax[2].set_title(f'output neuron membrane potential, alfa = {alfa}', fontweight ='bold', fontsize = 11, loc='left')
    ax[2].set_xlabel('time, ms', fontweight ='bold', fontsize = 11)
    plt.show()

def weight_dynamic(dir):
    lst = os.listdir(dir)
    lst = [int(name.split('_')[-1][:-4]) for name in lst]
    indxes = np.argsort(lst)
    file_names = os.listdir(dir)
    for i, indx in enumerate(indxes):
        file_name = file_names[indx]
        with open(os.path.join(dir, file_name), 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
        data_array = np.array(data)
        prop = data_array[1:6, 0].astype(float)
        num_inputs = prop[-1]
        weights = data_array[1:int(num_inputs+1), 2].astype(float)
        if i ==0:
            W = weights.reshape((2, 1))
        else:
            W= np.concatenate((W, weights.reshape((2, 1))), axis=1)
    return W

def plot_weights(dir, L, nu, alfa):
    weights = weight_dynamic(dir=dir)
    print(weights)
    ax = a4_plot()
    ax.plot(np.arange(L), weights[0], linewidth = 3, label = f'nu = {nu[0]}')
    ax.plot(np.arange(L), weights[1], linewidth = 3, label = f'nu = {nu[1]}')
    ax.grid(True,which='major',axis='both',alpha=0.3)
    ax.set_title(f'альфа = {alfa}', fontweight = 'bold', size = 11)
    ax.set_xlabel('время, с', fontweight = 'bold', size = 11)
    ax.set_ylabel('синаптический вес', fontweight = 'bold', size = 11)
    plt.legend()
    plt.show()

def make_plot_voltage_memory_trace_mean(ax, dir):
    _, _, _, dV, prop, nu = get_csv_dir_data(dir)
    dV = dV[0]
    time, _, _, alfa, _ = prop
    WMT = []
    range_t = np.arange(2, int(len(dV)+2))
    voltage_memory_trace = 0
    total = 0
    for N in np.arange(2, int(len(dV)+2)):
        if N%(time) !=0:
            k = np.arange(0, N-1)
            W = (N-k)**(1-alfa)-(N-1-k)**(1-alfa)
            voltage_memory_trace += dV[:N-1]@W.T
            total+=1
        else:
            WMT.append(voltage_memory_trace/total)
            voltage_memory_trace = 0
            total = 0
    ax.plot(range_t[range_t%(time)==0]/10000, WMT, linewidth = 3, label = f'alpha = {alfa}')
    ax.grid(True,which='major',axis='both',alpha=0.3)
    ax.set_title(f'ню = {nu}', fontweight = 'bold', size = 11)
    ax.set_xlabel('время, с', fontweight = 'bold', size = 11)
    ax.set_ylabel('траектория пямяти, мВ', fontweight = 'bold', size = 11)
    return ax

def make_plot_voltage_memory_trace(ax, dir):
    _, _, _, dV, prop, nu = get_csv_dir_data(dir)
    dV = dV[0]
    time, _, _, alfa, _ = prop
    WMT = []
    range_t = np.arange(2, int(len(dV)+2))
    for N in np.arange(2, int(len(dV)+2)):
        k = np.arange(0, N-1)
        W = (N-k)**(1-alfa)-(N-1-k)**(1-alfa)
        voltage_memory_trace = dV[:N-1]@W.T
        WMT.append(voltage_memory_trace)
    ax.plot(range_t/10000, WMT, linewidth = 3, label = f'nu = {nu}')
    ax.grid(True,which='major',axis='both',alpha=0.3)
    ax.set_title(f'альфа = {alfa}', fontweight = 'bold', size = 10)
    ax.set_xlabel('время, с', fontweight = 'bold', size = 10)
    ax.set_ylabel('траектория пямяти, мВ', fontweight = 'bold', size = 10)
    return ax


def plot_voltage_memory_trace(dirs):
    ax = a4_plot()
    for dir in dirs:
        ax = make_plot_voltage_memory_trace_mean(ax, dir)
    plt.legend()
    plt.show()

def ISI_plot(ax, folders, t0, size_fac, nu_extr):
    #ax = a4_plot()
    for f, folder in enumerate(folders):
        _, out_spikes, _, _, prop, nu = get_csv_dir_data(folder)
        time, _, _, alfa, _  = prop
        out_spikes, _ = np.where(out_spikes!=0)
        out_isi = count_ISI(out_spikes/10)
        x = np.arange(0, time, t0)
        y = np.array([0])
        for i in x:
            c = np.sum([out_isi<=i+t0]) - np.sum(y)
            y = np.append(y, c)
        y = y[1:]
        if f==0:
            total = np.array(y)
        else:
            total +=y
    if nu_extr!=0:
        nu = [nu_extr]
    ax.plot(x[total>1], total[total>1],  marker ='o',markersize = 8, linewidth = 0, markeredgewidth=0, label = f'nu = {nu}, alpha = {alfa}')
    if size_fac!=0:
        for i in range(len(nu)):
            params, covariance = curve_fit(lambda x, s: s*(x**(-1-nu[i])), x[total>1], total[total>1], size_fac)
            fit = (params[0])*(x[total>1]**(-1-nu[i]))
            ax.plot(x[total>1], fit, linewidth = 3, label = f'nu = {nu[i]}, size_fact = {np.round(params[0]+np.sqrt(covariance[0, 0]), 3)}')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('ISI, ms', fontweight = 'bold', size = 10)
    ax.set_ylabel('ISI number', fontweight = 'bold', size = 10)
    #ax.legend(labels = [f'nu = {nu}, alpha = {alfa}', 'approximation'])
    ax.grid(True,which='major',axis='both',alpha=0.3)
    #plt.show()
    return ax

def count_ISI(spikes):
    isi = spikes[1:]-spikes[0:-1]
    return isi

def count_acc(input, output, time_step):
    input-=np.min(input)
    i=0
    ones_time_in = []
    while i<len(input):
        if input[i]!=0:
            ones_time_in.append(i)
            i +=int(time_step)
        else:
            i+=1
    ones_time_out = np.where(output!=0)[0]/10
    ones_time_in = np.array(ones_time_in)/10
    acc = 0
    for t in ones_time_in:
        arr = ones_time_out[ones_time_out>=t]
        if len(arr) != 0:
            acc+=np.max((arr-t <= time_step/10)*1)
    print(acc,len(ones_time_in))
    #return acc/len(ones_time_in)
    return acc