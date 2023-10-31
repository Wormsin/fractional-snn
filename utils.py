import numpy as np
import matplotlib.pyplot as plt

def plot_spikes(in_features, out_features, in_spikes, out_spikes, V, range_t, V_th, V_rest):
    fig, ax = plt.subplots(3)
    for i in range(in_features):
        ax[0].scatter(range_t, in_spikes[i]*(i+1),  s=0.8)
    for i in range(out_features):
        ax[1].scatter(range_t, out_spikes[:, i]*(i+1),  s=5)
    for i in range(out_features):
        ax[2].plot(range_t, V[:, i])
        ax[2].hlines(V_th, 0, range_t[-1], color = 'r')
        ax[2].hlines(V_rest, 0, range_t[-1], color = 'g')
    leg_in = [str(x) for x in range(1,in_features+1)]
    leg_out = [str(x) for x in range(1,out_features+1)]
    if(in_features < 20):
        ax[0].legend(leg_in, loc='upper center', bbox_to_anchor=(0.5, 1.2),
          fancybox=True, shadow=True, ncol=in_features)
    else:
        ax[0].set_ylabel(f"{in_features} input neurons")
    ax[1].legend(leg_out)
    ax[2].legend(leg_out)