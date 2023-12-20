import numpy as np
import matplotlib.pyplot as plt

def plot_spikes(in_features, out_features, in_spikes, out_spikes, V, range_t, V_th, V_rest):
    fig, ax = plt.subplots(3)
    for i in range(in_features):
        #ax[0].scatter(range_t[in_spikes[i]!=0], in_spikes[i][in_spikes[i]!=0]*(i+1),  s=0.8)
        ax[0].vlines(range_t[in_spikes[i]!=0], 0+i, 1+i, color = (0.1, 0.2, 0.5*i))
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

    ax[0].get_yaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)

    ax[0].set_title('input spikes, nu = 0.6', fontweight ='bold', fontsize = 10, loc = 'left')
    ax[1].set_title('output spikes', fontweight ='bold', fontsize = 10, loc = 'left')
    ax[2].set_title('output layer membrane potential, alfa = 1', fontweight ='bold', fontsize = 10, loc='left')
    ax[2].set_xlabel('time, ms', fontweight ='bold', fontsize = 10)

def count_ISI(spikes):
    isi = spikes[1:]-spikes[0:-1]
    return isi

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



