import numpy as np
import matplotlib as mpl
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
        v, out_s, in_s, dv, prop = data_array[1:, 0].astype(float), data_array[1:, 1].astype(float), data_array[1:, 2].astype(float), data_array[1:, 3].astype(float), data_array[1:6, 4].astype(float)
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
    prop[0]*=len(file_names)
    prop[3]*=len(file_names)
    return V, out_spikes, in_spikes, [dV], prop

def get_csv_file_data(file_name):
    with open(file_name, 'r') as f:
            reader = csv.reader(f)
            data = list(reader)
    data_array = np.array(data)
    V, out_spikes, in_spikes, dV, prop = data_array[1:, 0].astype(float), data_array[1:, 1].astype(float), data_array[1:, 2].astype(float), data_array[1:, 3].astype(float), data_array[1:6, 4].astype(float)
    in_spikes = np.expand_dims(in_spikes, 0)
    out_spikes = np.expand_dims(out_spikes, 1)
    V = np.expand_dims(V, 1)
    return V, out_spikes, in_spikes, [dV[:-1]], prop

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
        ax[0].plot(range_t, in_spikes[i], color = 'orange')
    for i in range(out_features):
        #ax[1].scatter(range_t[out_spikes[i]!=0], out_spikes[i][out_spikes[i]!=0]*(i+1),  s=5)
        ax[1].vlines(range_t[out_spikes[i]!=0], 0+i, 1+i, color = (0.1, 0.5, 0.5*i))
    for i in range(out_features):
        ax[2].plot(range_t, V[:, i])
    ax[2].hlines(V_th, 0, range_t[-1], color = 'r')
    ax[2].hlines(V_rest, 0, range_t[-1], color = 'g')
    #leg_in = [str(x) for x in range(1,in_features+1)]
    leg_out = [str(x) for x in range(1,out_features+1)]
    leg_out.append('Vth')
    leg_out.append('Vrest')
    #if(in_features < 20):
    #    ax[0].legend(leg_in, loc='upper left', bbox_to_anchor=(1, 1),
    #      fancybox=True, shadow=True, ncol=in_features)
        
    #ax[1].legend(leg_out, loc='upper left', bbox_to_anchor=(1, 0),  fancybox=True, shadow=True)
    ax[2].legend(leg_out, loc='upper left',bbox_to_anchor=(0.99, 1.15), frameon = False, )
    ax[0].set_xlim([-1, range_t[-1]+1])
    ax[1].set_xlim([-1, range_t[-1]+1])
    ax[2].set_xlim([-1, range_t[-1]+1])

    #ax[0].get_yaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    plt.title(title)
    nu, step, rate, alfa = legend
    ax[0].set_title(f'input current nA, nu = {nu}, step = {step} ms, mean val. of spikes = {rate}', fontweight ='bold', fontsize = 11, loc = 'left')
    ax[1].set_title('output spikes', fontweight ='bold', fontsize = 11, loc = 'left')
    ax[2].set_title(f'output neuron membrane potential, alfa = {alfa}', fontweight ='bold', fontsize = 11, loc='left')
    ax[2].set_xlabel('time, ms', fontweight ='bold', fontsize = 11)
    plt.show()

def ISI_plot(ax, folder, t0, size_fac, nu_extr):
    file_names = os.listdir(folder)
    #ax = a4_plot()
    for f, file_name in enumerate(file_names):
        _, out_spikes, _, _, prop = get_csv_file_data(folder+'/'+file_name)
        time, nu, _, _, alfa = prop
        out_isi = count_ISI(out_spikes[out_spikes!=0])
        x = np.arange(t0//2, time, t0)
        for i in range(len(x)):
            if i!=0:
                c = np.sum([out_isi<=x[i]+t0//2]) - np.sum(y[:i])
                y = np.append(y, c)
            else:
                c = np.sum([out_isi<=x[i]+t0//2])
                y = np.array([c])
        if f==0:
            total = np.array(y)
        else:
            total +=y
    if nu_extr!=0:
        nu = nu_extr
    fit = size_fac*x[total>1]**(-1-nu)
    ax.plot(x[total>1], total[total>1],  marker ='o',markersize = 8, linewidth = 0, markeredgewidth=0, label = f'nu = {nu}, alpha = {alfa}')
    if size_fac!=0:
        ax.plot(x[total>1], fit, linewidth = 3, label = f'nu = {nu}')
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

