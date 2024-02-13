import numpy as np
import matplotlib.pyplot as plt
import csv
import os 

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
        v, out_s, in_s, dv = data_array[1:, 0].astype(float), data_array[1:, 1].astype(float), data_array[1:, 2].astype(float), data_array[1:, 3].astype(float)
        if i ==0:
            V = v
            out_spikes = out_s
            in_spikes = in_s
            dV = dv[:-1]
        else:
            V = np.concatenate((V, v))
            out_spikes = np.concatenate((out_spikes, out_s))
            in_spikes = np.concatenate((in_spikes, in_s))
            dV = np.concatenate((dV, dv[:-1]))
    in_spikes = np.expand_dims(in_spikes, 0)
    out_spikes = np.expand_dims(out_spikes, 1)
    V = np.expand_dims(V, 1)
    return V, out_spikes, in_spikes, [dV]

def get_csv_file_data(file_name):
    with open(file_name, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
    data_array = np.array(data)
    V, out_spikes, in_spikes, dV = data_array[1:, 0].astype(float), data_array[1:, 1].astype(float), data_array[1:, 2].astype(float), data_array[1:, 3].astype(float)
    in_spikes = np.expand_dims(in_spikes, 0)
    out_spikes = np.expand_dims(out_spikes, 1)
    V = np.expand_dims(V, 1)
    return V, out_spikes, in_spikes, [dV[:-1]]

def plot_spikes(in_features, out_features, in_spikes, out_spikes, V, range_t, V_th, V_rest, legend, title = ""):
    fig, ax = plt.subplots(3)
    for i in range(in_features):
        #ax[0].scatter(range_t[in_spikes[i]!=0], in_spikes[i][in_spikes[i]!=0]*(i+1),  s=0.8)
        #ax[0].vlines(range_t[in_spikes[i]!=0], 0+i, 1+i, color = (0.1, 0.2, 0.5*i))
        ax[0].plot(range_t, in_spikes[i], color = 'orange')
    for i in range(out_features):
        #ax[1].scatter(range_t[out_spikes[i]!=0], out_spikes[i][out_spikes[i]!=0]*(i+1),  s=5)
        ax[1].vlines(range_t[out_spikes[i]!=0], 0+i, 1+i, color = (0.1, 0.5, 0.5*i))
    for i in range(out_features):
        ax[2].plot(range_t, V[:, i])
    ax[2].hlines(V_th, 0, range_t[-1], color = 'r')
    ax[2].hlines(V_rest, 0, range_t[-1], color = 'g')
    leg_in = [str(x) for x in range(1,in_features+1)]
    leg_out = [str(x) for x in range(1,out_features+1)]
    leg_out.append('Vth')
    leg_out.append('Vrest')
    if(in_features < 20):
        ax[0].legend(leg_in, loc='upper center', bbox_to_anchor=(0.5, 1.2),
          fancybox=True, shadow=True, ncol=in_features)
        
    ax[1].legend(leg_out)
    ax[2].legend(leg_out, loc='upper right',fancybox=True, ncol=in_features)
    ax[0].set_xlim([-1, range_t[-1]+1])
    ax[1].set_xlim([-1, range_t[-1]+1])
    ax[2].set_xlim([-1, range_t[-1]+1])

    #ax[0].get_yaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    plt.title(title)
    nu, step, rate, alfa = legend
    ax[0].set_title(f'input current nA, nu = {nu}, step = {step} ms, rate = {rate}', fontweight ='bold', fontsize = 10, loc = 'left')
    ax[1].set_title('output spikes', fontweight ='bold', fontsize = 10, loc = 'left')
    ax[2].set_title(f'output layer membrane potential, alfa = {alfa}', fontweight ='bold', fontsize = 10, loc='left')
    ax[2].set_xlabel('time, ms', fontweight ='bold', fontsize = 10)
    plt.show()

def count_ISI(spikes):
    isi = spikes[1:]-spikes[0:-1]
    return isi

def count_acc(input, output, time_step):
    #input-=np.min(input)
    i=0
    ones_time_in = []
    while i<len(input):
        if input[i]==1:
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
    #print(acc,len(ones_time_in))
    return acc/len(ones_time_in)

def plot_acc(arr, time_interval):
    plt.plot(np.arange(1, len(arr)+1)*time_interval, arr, 'o')
    plt.plot([0], [arr[0]], 'o')
    plt.xlabel("unaccounted periods of time", fontweight ='bold')
    plt.ylabel('accuracy', fontweight ='bold')
    plt.title("influence unaccounted dV on accuracy", fontweight ='bold')
    plt.legend(['partial memory', 'total memory'])
    plt.show()

def plot_means(means, br_labels, x_labels, xname, yname):
    r, c = means.shape
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8)) 
    for i, br in enumerate(means):
        plt.bar(np.arange(0, c, 1)+barWidth*i, br, color =(0.1, 0.5, 0.5*i), width = barWidth, 
        edgecolor ='grey', label = br_labels[i])
    plt.xlabel(xname, fontweight ='bold', fontsize = 15) 
    plt.ylabel(yname, fontweight ='bold', fontsize = 15) 
    plt.xticks(np.arange(0, c, 1)+barWidth, 
            x_labels)
    plt.legend()
