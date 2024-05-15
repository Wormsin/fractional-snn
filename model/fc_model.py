import numpy as np
from scipy.special import gamma 
from math import ceil
from numpy.random import uniform
from model import learning
from model import layer
import csv
import os
    

class FC():
    def __init__(self, layers:layer.Layer, time, dt, reset:bool) -> None:
        self.reset_2_initial = reset
        self.layers = layers
        self.L_time = int(time/dt)
        self.dt = dt
        self.train = True
        self.classes =  layers[-1].out_features
        self.V_out = np.ones((1, self.classes))*self.layers[-1].start_V
        self.spks_out = np.zeros((1, self.classes))

    def learn_step(self, layer, spikes, in_spks):
        layer.P, layer.M, layer.weights=learning.stdp(layer.P, layer.M, layer.weights, spikes, in_spks)

    def load_weights(self, dir):
        lst = os.listdir(dir)
        lst = [int(name[-5]) for name in lst]
        indxes = np.argsort(lst)
        file_names = os.listdir(dir)
        for i, indx in enumerate(indxes):
            file_name = file_names[indx]
            layer = self.layers[i]
            with open(dir+'/'+file_name, 'r') as f:
                reader = csv.reader(f, delimiter=",")
                data = list(reader)
            data_array = np.array(data)
            layer.weights = data_array.astype(float)
    
    def reset(self):
        for layer in self.layers:
            layer.M = np.zeros(layer.out_features)
            layer.P = np.zeros((layer.out_features,layer.in_features))
            layer.V_mem = np.ones(layer.out_features)*layer.start_V
            layer.dV = list(np.ones((layer.out_features, 1))*[])
            layer.N = np.zeros((layer.out_features), dtype =np.int32)
            layer.tr = np.ones((layer.out_features, 1))*layer.neuron.tref
            layer.out_spikes = np.zeros((layer.out_features))


    def forward(self, input_spikes):
        if self.reset_2_initial:
            self.reset()
        self.V_out = np.ones((1, self.classes))*self.layers[-1].start_V
        self.spks_out = np.zeros((1, self.classes))
        for i in range(self.L_time - 1):
            spikes = input_spikes[:, i]
            for c, layer in enumerate(self.layers):
                in_spks = np.copy(spikes)
                spikes, v = layer.feed(spikes)
                if self.train:
                    self.learn_step(layer, spikes, in_spks)
                if c==len(self.layers)-1:
                    self.V_out[-1]=np.where(spikes!=0, self.layers[-1].neuron.V_th, self.V_out[-1])
                    self.V_out = np.concatenate((self.V_out, v.reshape(1, self.classes)))
                    self.spks_out = np.concatenate((self.spks_out, spikes.reshape(1, self.classes)*i/self.dt))
        return (self.spks_out!=0)*1



