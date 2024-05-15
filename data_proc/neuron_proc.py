import os
import numpy as np
import features.utils as utils
import model as m 
import data_proc.encoding as code
from features.save import save_one_neuron_exp

from model.neuron_params import *


def instant_view(snn, train:bool):
    snn.train = train
    np.random.seed(5)
    input_spikes = code.fractional_poisson_signal(nu, N, T, dt, time_step, inputs)
    out_spikes = snn.forward(input_spikes)
    V = snn.V_out
    utils.plot_neuron_exp(inputs, 1, input_spikes*Iinj, out_spikes.T, V, np.arange(0, T, dt), V_th, E_L, 
                        legend = [nu, f"{time_step}", f"{N}", f"{alfa}", f"{Iinj}"])

def make_new_data(snn, epochs, train):
    snn.train = train
    i = 0
    if not os.path.isdir("output_data"):
            os.mkdir('output_data')
    np.random.seed(5)
    input_spikes = code.fractional_poisson_signal(nu, N, T, dt, time_step, inputs)
    out_spikes = snn.forward(input_spikes)
    weights = snn.layers[0].weights
    dV = snn.layers[0].dV[0]
    V = snn.V_out
    save_one_neuron_exp(file_name=f'output_data/data_{i}.csv', dV=dV, V=V, out_spikes=out_spikes, in_spikes=input_spikes, inputs=inputs,
                        spk_amp=Iinj, L_time=T/dt, dt = dt, t_step=time_step, N_spk=N, alfa=alfa, nu = nu, weights=weights)
    i+=1
    while i < epochs:
        snn.layers[0].start_V = V[-1, 0]
        np.random.seed(5)
        input_spikes = code.fractional_poisson_signal(nu, N, T, dt, time_step, inputs)
        out_spikes = snn.forward(input_spikes)
        weights = snn.layers[0].weights
        dV = np.copy(snn.layers[0].dV[0][int(i*T/dt)-i:])
        V = snn.V_out
        save_one_neuron_exp(file_name=f'output_data/data_{i}.csv', dV=dV, V=V, out_spikes=out_spikes, in_spikes=input_spikes, inputs=inputs,
                    spk_amp=Iinj, L_time=T/dt, dt = dt, t_step=time_step, N_spk=N, alfa=alfa, nu = nu, weights=weights)
        i+=1

def add_data(snn, epochs, i, folder, train):
    snn.train = train
    V, _, _, dV, _, _, weights = utils.get_csv_dir_data(folder)
    snn.layers[0].weights = weights
    snn.layers[0].start_V = V[-1, 0]
    snn.layers[0].dV = dV
    snn.layers[0].N = np.array([len(dV[0])], dtype =np.int32)
    snn.layers[0].V_mem = np.array([V[-1, 0]])
    while i < epochs:
        np.random.seed(5)
        input_spikes = code.fractional_poisson_signal(nu, N, T, dt, time_step, inputs)
        out_spikes = snn.forward(input_spikes)
        weights = snn.layers[0].weights
        dV = np.copy(snn.layers[0].dV[0][int(i*T/dt)-i:])
        V = snn.V_out
        save_one_neuron_exp(file_name=f'output_data/data_{i}.csv', dV=dV, V=V, out_spikes=out_spikes, in_spikes=input_spikes, inputs=inputs,
                    spk_amp=Iinj, L_time=T/dt, dt = dt, t_step=time_step, N_spk=N, alfa=alfa, nu = nu, weights=weights)
        snn.layers[0].start_V = V[-1, 0]
        i+=1


def plot_folder(folder_name):
        V, out_spikes, in_spikes, dV, prop, nu, weights = utils.get_csv_dir_data(folder_name)
        print(weights)
        time, time_step, N, alfa, in_features = prop
        range_t = np.arange(0, time, 0.1)
        Iinj = np.max(in_spikes)
        utils.plot_neuron_exp(int(in_features), 1, in_spikes, out_spikes.T, V, range_t, V_th, E_L, 
                            legend = [nu, f"{time_step}", f"{N}", f"{alfa}", f"{Iinj}"])
        return None

def plot_file(file_name):
    V, out_spikes, in_spikes, _, prop, nu, weights = utils.get_csv_file_data(file_name)
    print(weights)
    time, time_step, N, alfa, in_features = prop
    range_t = np.arange(0, time, 0.1)
    Iinj = np.max(in_spikes)
    utils.plot_neuron_exp(int(in_features), 1, in_spikes, out_spikes.T, V, range_t, V_th, E_L, 
                        legend = [nu, f"{time_step}", f"{N}", f"{alfa}",  f"{Iinj}"])
    return None