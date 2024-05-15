import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import os
import cv2
from scipy.optimize import curve_fit



def a4_plot():
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.1,0.1,0.5,0.8])
    return ax

def plot_neuron_exp(in_features, out_features, in_spikes, out_spikes, V, range_t, V_th, V_rest, legend, title = ""):
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


def ISI_plot(filen,color,ax, time, t0, size_fac, nu):
    out_spikes = []
    with open(filen, 'r') as file:
        csvreader = csv.reader(file)
        for c, row in enumerate(csvreader):
            if c>0:
                out_spikes.append(float(row[0]))
    #ax = a4_plot()
    out_spikes = np.array(out_spikes)
    out_isi = count_ISI(out_spikes/10)
    x = np.arange(0, time, t0)
    y = np.array([0])
    for i in x:
        c = np.sum([out_isi<=i+t0]) - np.sum(y)
        y = np.append(y, c)
    return y

def count_ISI(spikes):
    isi = spikes[1:]-spikes[0:-1]
    return isi


def plot_inspks(out_spikes, T, dt, nu, img_file):
    range_t = np.arange(0, T*dt, dt)
    class_name = img_file[-5]
    ax = a4_plot()
    ax2 = a4_plot()
    for i in range(len(nu)):
        #ax.vlines(range_t[out_spikes[i]!=0], 0+i, 1+i, color = (0.1, 0.5, 0.5), label = f"neuron: {i+1}")
        ax.scatter(range_t[out_spikes[i]!=0], out_spikes[i][out_spikes[i]!=0]*(i+1),  s=3, label = f"neuron: {i+1}, nu = {nu[i]}")
    ax.set_xlim([-1, range_t[-1]+1])
    ax.get_yaxis().set_visible(False)
    ax.set_title(f'input spikes for image of class: {class_name}', fontweight ='bold', fontsize = 11, loc = 'left')
    ax.set_xlabel('t, с')
    img = cv2.imread(img_file)
    ax2.imshow(img, cmap='gray')
    ax.legend( loc='upper left',bbox_to_anchor=(0.99, 1), frameon = False, )
    plt.show()
    
def plot_outspks(out_spikes, T, dt, classes, img_file):
    range_t = np.arange(0, T*dt, dt)
    class_name = img_file[-5]
    ax = a4_plot()
    ax2 = a4_plot()
    for i in range(classes):
        ax.vlines(range_t[out_spikes[i]!=0], 0+i, 1+i, color = (0.1, 0.5, 0.5*i), label = f"neuron: {i+1}")
        #ax.scatter(range_t[out_spikes[i]!=0], out_spikes[i][out_spikes[i]!=0]*(i+1),  s=3, label = f"neuron: {i+1}")
    ax.set_xlim([-1, range_t[-1]+1])
    ax.get_yaxis().set_visible(False)
    ax.set_title(f'output spikes for image of class: {class_name}', fontweight ='bold', fontsize = 11, loc = 'left')
    ax.set_xlabel('t, с')
    img = cv2.imread(img_file)
    ax2.imshow(img, cmap='gray')
    ax.legend( loc='upper left',bbox_to_anchor=(0.99, 1), frameon = False, )
    plt.show()

def plot_v_out(model, T, dt, V_th, V_rest, alfa):
    ax = a4_plot()
    range_t = np.arange(0, T*dt, dt)
    ax.plot(range_t, model.V_out)
    ax.hlines(V_th, 0, range_t[-1], color = 'r')
    ax.hlines(V_rest, 0, range_t[-1], color = 'g')
    ax.set_xlim([-1, range_t[-1]+1])
    ax.set_title(f'output neuron membrane potential, alfa = {alfa}', fontweight ='bold', fontsize = 11, loc='left')
    ax.set_xlabel('time, ms', fontweight ='bold', fontsize = 11)
    plt.show()

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