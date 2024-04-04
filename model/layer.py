import numpy as np
from scipy.special import gamma 
from math import ceil
from numpy.random import uniform
from model import neuron


class Layer():
    def __init__(self, in_features, out_features, start_V, neuron:neuron.Fractional_LIF) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.neuron = neuron
        self.start_V = start_V
        self.weights = np.random.rand(in_features, out_features)
        self.M = np.zeros(out_features)
        self.P = np.zeros((out_features,in_features))
        self.V_mem = np.ones(out_features)*start_V
        self.dV = list(np.ones((out_features, 1))*[])
        self.N = np.zeros((out_features), dtype =np.int32)
        self.tr = np.ones((out_features, 1))*neuron.tref
        self.out_spikes = np.zeros((out_features))
    
    def calc_dV(self, v_old, v_new, i):
        self.dV[i] = np.append(self.dV[i], v_new-v_old)

    def feed(self, spikes):
        self.N+=1
        for i in range(self.out_features):
            i_pulse = max((spikes*self.neuron.spk_amp)*self.weights[:, i].T)*1000
            v_old = self.V_mem[i]
            out_spk, self.V_mem[i], self.tr[i]= self.neuron.mem_dynamics(self.V_mem[i], self.dV[i][0:self.N[i]-1], self.tr[i], self.N[i], i_pulse)
            v_new = self.V_mem[i]
            self.calc_dV(v_old, v_new, i)
            self.out_spikes[i] = out_spk
        return self.out_spikes, self.V_mem