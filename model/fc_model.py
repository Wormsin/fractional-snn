import numpy as np
from scipy.special import gamma 
from math import ceil
from numpy.random import uniform
from model import learning
from model import layer
import csv
import os
    

class FC():
    def __init__(self, layers:layer.Layer, L_time) -> None:
        self.layers = layers
        self.L_time = L_time 
        self.dt = 0.1
        self.time = np.arange(0, L_time, self.dt)
        self.train = True
        self.classes =  layers[-1].out_features
        self.V_out = np.ones((1, self.classes))*self.layers[-1].start_V
        self.spks_out = np.zeros((1, self.classes))

    def learn_step(self, layer, spikes, in_spks):
        layer.P, layer.M, layer.weights=learning.stdp(layer.P, layer.M, layer.weights, spikes, in_spks)

    def load_weights(self, dir):
        file_names = os.listdir(dir)
        for i, layer in enumerate(self.layers):
            file_name = file_names[i]
            with open(file_name, 'r') as f:
                reader = csv.reader(f)
                data = list(reader)
            data_array = np.array(data)
            layer.weights = data_array.astype(float)

    def forward(self, input_spikes):
        for i in range(self.L_time - 1):
            spikes = input_spikes[:, i]
            for c, layer in enumerate(self.layers):
                in_spks = np.copy(spikes)
                spikes, v = layer.feed(spikes)
                if self.train:
                    self.learn_step(layer, spikes, in_spks)
                if c==len(self.layers)-1:
                    V[-1]=np.where(spikes!=0, self.layers[-1].neuron.V_th, V[-1])
                    V = np.concatenate((V, v.reshape(1, self.classes)))
                    out_spikes = np.concatenate((out_spikes, spikes.reshape(1, self.classes)*self.time[i]))
        return (out_spikes!=0)*1



