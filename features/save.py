import numpy as np
import pandas as pd
import os


def save_weights(model, epoch):
    if not os.path.isdir(f'weights{epoch}'):
        os.mkdir(f'weights{epoch}')
    for c, layer in enumerate(model.layers):
        np.savetxt(f'weights{epoch}/layer{c}.csv', layer.weights,  
              delimiter = ",")
         
def save_prediction(file_name, V, out_spikes):
    V = V.T[0]
    out_spikes = out_spikes.T[0]
    data_main = {
            'V': V,
            'out_spikes': out_spikes,
        }
    df_main = pd.DataFrame(data_main)
    df_main.to_csv(file_name, index=False)


def save_one_neuron_exp(file_name, dV, V, out_spikes, in_spikes, inputs, spk_amp, L_time, dt, t_step, N_spk, alfa, nu, weights):
        V = V.T[0]
        out_spikes = out_spikes.T[0]
        dV = np.append(dV, 0)
        weights = weights.reshape((inputs))
        data_main = {
            'V': V,
            'out_spikes': out_spikes,
            'dV': dV
        }
        for i in range(inputs):
            data_main[f'in_spikes{i}'] =  in_spikes[i]*spk_amp
        data_prop = {
            'prop': [
            L_time*dt,
            t_step,
            N_spk,
            alfa,
            inputs]
        }
        data_input = {
            'nu' : nu,
            'weights' : weights
        }
        df_input = pd.DataFrame(data_input)
        df_prop = pd.DataFrame(data_prop)
        df_main = pd.DataFrame(data_main)
        merged_data = pd.concat([df_prop, df_input, df_main], ignore_index=True, axis=1)
        merged_data.to_csv(file_name, index=False)

