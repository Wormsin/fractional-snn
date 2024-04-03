import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import csv
import os
from scipy.optimize import curve_fit



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


def ISI_plot(ax, alfa, time, out_spikes,  t0, size_fac, nu_extr):
    #ax = a4_plot()
    out_spikes, _ = np.where(out_spikes!=0)
    out_isi = count_ISI(out_spikes/10)
    x = np.arange(0, time, t0)
    y = np.array([0])
    for i in x:
        c = np.sum([out_isi<=i+t0]) - np.sum(y)
        y = np.append(y, c)
    y = y[1:]
    if nu_extr!=0:
        nu = [nu_extr]
    ax.plot(x[y>1], y[y>1],  marker ='o',markersize = 8, linewidth = 0, markeredgewidth=0, label = f'nu = {nu}, alpha = {alfa}')
    if size_fac!=0:
        for i in range(len(nu)):
            params, covariance = curve_fit(lambda x, s: s*(x**(-1-nu[i])), x[y>1], y[y>1], size_fac)
            fit = (params[0])*(x[y>1]**(-1-nu[i]))
            ax.plot(x[y>1], fit, linewidth = 3, label = f'nu = {nu[i]}, size_fact = {np.round(params[0]+np.sqrt(covariance[0, 0]), 3)}')
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


def plot_spks(out_spikes, range_t, classes, img_file):
    class_name = img_file[-5]
    ax = a4_plot()
    for i in range(classes):
        ax.vlines(range_t[out_spikes[i]!=0], 0+i, 1+i, color = (0.1, 0.5, 0.5*i))
    ax.set_xlim([-1, range_t[-1]+1])
    ax.get_yaxis().set_visible(False)
    ax.set_title(f'output spikes for image of class: {class_name}', fontweight ='bold', fontsize = 11, loc = 'left')
    ax.set_xlabel('t, с')
    plt.imshow(img_file, cmap='gray')
    plt.show()
    