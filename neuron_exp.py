import numpy as np
import matplotlib.pyplot as plt

from model import neuron, layer, fc_model
from data_proc.neuron_proc import instant_view, make_new_data, plot_folder, plot_file, add_data
from features import utils

from model.neuron_params import *

neuron_flif = neuron.Fractional_LIF(V_th, V_reset, start_V, Iinj, g_L, C_m, dt, alfa, tref)
layer = layer.Layer(inputs, 1, start_V, neuron_flif)
layer.weights = np.ones((inputs, 1))
snn = fc_model.FC([layer], T, dt, False)


#instant_view(snn, train=False)
#make_new_data(snn, 4, False)
#add_data(snn, 4, 2, "output_data", False)

#plot_file("output_data/data_2.csv")
#plot_folder("output_data")


