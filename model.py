import numpy as np
import matplotlib.pyplot as plt


class LIF():
    def __init__(self, V_th, V_reset, E_L, b, spk_amp) -> None:
        self.V_th = V_th
        self.V_reset = V_reset
        self.E_L = E_L
        self.b = b
        self.spk_amp = spk_amp
            
    def mem_dynamics(self, v, weights, spikes) -> bool:
        v = np.array([v])
        if v>=self.E_L*1.0001 and v<self.V_th:
            v = v + (spikes*self.spk_amp)@weights
        spk = (v>=self.V_th)
        v_next=v+spk*self.V_reset
        v = self.E_L+(v_next-self.E_L)*self.b
        return spk*1, v[0]


class Layer():
    def __init__(self, in_features, out_features, neuron:LIF, lr, A_minus, A_plus, alfa_minus, alfa_plus) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.weights= np.random.rand(in_features, out_features)
        self.neuron = neuron
        self.V_mem = np.ones(out_features)*neuron.E_L
        self.A_minus = A_minus
        self.A_plus = A_plus
        self.alfa_minus = alfa_minus
        self.alfa_plus = alfa_plus
        self.lr = lr
        self.M = np.zeros(out_features)
        self.P = np.zeros(in_features)
        self.w_min = 0
        self.w_max = 1

    def post_synaptic_dynamics(self, out_spk, i):
        self.M[i] = out_spk*self.A_minus+self.M[i]
        self.M[i] = self.M[i]*self.alfa_minus
        return self.M[i]
    
    def pre_synaptic_dynamics(self, spikes):
        self.P = self.A_plus*spikes+self.P
        self.P = self.P*self.alfa_plus
        return self.P
  
    def forward(self, spikes):
        out_spikes = np.empty(self.out_features)
        dw_plus = self.pre_synaptic_dynamics(spikes)
        for i in range(self.out_features):
            out_spk, self.V_mem[i] = self.neuron.mem_dynamics(self.V_mem[i], self.weights[:, i], spikes)
            out_spikes[i] = out_spk
            dw_minus = self.post_synaptic_dynamics(out_spk, i)
            self.weights[:, i] += self.lr*dw_minus*self.weights[:, i]  + self.lr*dw_plus*self.weights[:, i]*out_spk 
        return out_spikes, self.V_mem
        

class SNN():
    def __init__(self, layers:Layer, input, L_time, classes, input_time) -> None:
        self.layers = layers
        self.input = input
        self.L_time = L_time 
        self.classes = classes
        self.time = input_time

    def encoding(self):
        input_spikes = np.zeros((self.input.shape[0], self.L_time))
        for i in range(self.input.shape[0]):
            for it in range(self.L_time):
                if it<self.L_time*0.8 and it%20==0:
                    dt = int(-np.log(np.random.uniform(0, 1))*self.input[i]*100)
                    input_spikes[i, it+dt] = 1
        return input_spikes
    

    def inference(self):
        V = np.ones((1, self.classes))*self.layers[0].neuron.E_L
        spikes = []
        input_spikes = self.encoding()
        out_spikes = np.zeros((1, self.classes))
        for i in range(self.L_time - 1):
            spikes = input_spikes[:, i]
            for c, layer in enumerate(self.layers):
                spikes, v = layer.forward(spikes)
                if c==len(self.layers)-1:
                    V = np.concatenate((V, v.reshape(1, self.classes)))
                    out_spikes = np.concatenate((out_spikes, spikes.reshape(1, self.classes)*self.time[i]))
        num_spikes = np.sum(out_spikes!=0, axis=0)
        dT = np.max(out_spikes, axis=0)-np.min(out_spikes, axis=0)
        np.where(dT!=0, dT, 1)
        return input_spikes, (out_spikes!=0)*1, num_spikes/dT, V


