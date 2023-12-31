import numpy as np
from scipy.special import gamma 
from math import ceil
from numpy.random import uniform

class Fractional_LIF():
    def __init__(self, V_th, V_reset, E_L, spk_amp, g_L, C_m, dt, alfa, tref, stdp_rate, stdp_tm, stdp_tp, stdp_Aminus, stdp_Aplus) -> None:
        self.V_th = V_th
        self.V_reset = V_reset
        self.E_L = E_L
        self.g = g_L
        self.C = C_m
        self.spk_amp = spk_amp
        self.dt = dt
        self.alfa = alfa
        self.tref = tref
        self.lr = stdp_rate
        self.t_m = stdp_tm
        self.t_p = stdp_tp
        self.A_minus = stdp_Aminus
        self.A_plus = stdp_Aplus
            
    def mem_dynamics(self, v, weights, spikes, dV, tr, N):
        i_pulse = max((spikes*self.spk_amp)*weights.T)
        spk = (v>=self.V_th-0.01)
        if spk:
            tr = self.tref*2-self.dt
            v = self.V_reset
            return spk*1, v, tr
        if tr<=self.tref:
            if N>=2 and self.alfa!=1:
                markov_term = ((self.dt)**(self.alfa))*((-self.g*(v-self.E_L)+i_pulse)/self.C)*gamma(2-self.alfa)+v
                k = np.arange(0, N-1)
                W = (N-k)**(1-self.alfa)-(N-1-k)**(1-self.alfa)
                voltage_memory_trace = dV@W.T
            else :
                markov_term = self.dt*((-self.g*(v-self.E_L)+i_pulse)/self.C)+v
                voltage_memory_trace = 0
            v = markov_term - voltage_memory_trace
        else:
            tr = tr-self.dt
            v=self.V_reset
        return spk*1, v, tr
    
    def stdp(self, P, M, weights, out_spk, spikes):
        ind_w = np.argmax(weights*spikes)
        spike = spikes*0
        spike[ind_w] = 1
        spike = spike*spikes
        M = M + out_spk*self.A_minus - M*self.dt/self.t_m
        P = P + spike[ind_w]*self.A_plus - P*self.dt/self.t_p
        dw_m = self.lr*M*weights*spikes
        dw_p = self.lr*P*weights*out_spk
        weights += np.where(dw_m>0, dw_m, 0)  + np.where(dw_p<1, dw_p, 1)
        return P, M, weights


class Layer():
    def __init__(self, in_features, out_features, neuron:Fractional_LIF) -> None:
        self.in_features = in_features
        self.out_features = out_features
        self.neuron = neuron
        #self.weights= np.random.rand(in_features, out_features)
        self.weights = np.array([[1]])
        self.M = np.zeros(out_features)
        self.P = np.zeros(in_features)
        self.V_mem = np.ones(out_features)*neuron.E_L
        self.dV = list(np.ones((out_features, 1))*[])
        self.N = np.zeros((out_features), dtype =np.int32)
        self.tr = np.ones((out_features, 1))*neuron.tref
    
    def calc_dV(self, v_old, v_new, i):
        self.dV[i] = np.append(self.dV[i], v_new-v_old)

    def feed(self, spikes, train):
        out_spikes = np.empty(self.out_features)
        self.N+=1
        for i in range(self.out_features):
            v_old = self.V_mem[i]
            out_spk, self.V_mem[i], self.tr[i]= self.neuron.mem_dynamics(self.V_mem[i], self.weights[:, i], spikes, 
                                                                        self.dV[i][0:self.N[i]-1], self.tr[i], self.N[i])
            v_new = self.V_mem[i]
            self.calc_dV(v_old, v_new, i)
            if train:
                self.P[i], self.M[i], self.weights[:, i]=self.neuron.stdp(self.P[i], self.M[i], self.weights[:, i], out_spk, spikes)
            out_spikes[i] = out_spk
        return out_spikes, self.V_mem
    
class SNN():
    def __init__(self, layers:Layer, input, L_time, classes, time_interval, rate, nu, train: bool) -> None:
        self.layers = layers
        self.input = input
        self.L_time = L_time 
        self.classes = classes
        self.time = time_interval
        self.dt = 0.1
        self.rate = rate
        self.nu = nu
        self.train = train

    def encoding(self):
        chain = np.zeros((self.input.shape[0], self.L_time))
        input_rate = self.input*self.rate
        for it in range(self.input.shape[0]):
            t=0            
            for i in range(self.L_time):
                if t==0 or chain[it, i] == 1 :
                    U1 = uniform()
                    U2 = uniform()
                    U3 = uniform()
                    poisson_tau = ((-np.log(U1))**(1/self.nu))/(input_rate[it]**(1/self.nu))
                    levy_tau = np.sin(self.nu*np.pi*U2)*((np.sin((1-self.nu)*np.pi*U2))**(1/self.nu-1))/(((np.sin(np.pi*U2))**(1/self.nu))*((-np.log(U3))**(1/self.nu-1)))
                    tau = poisson_tau*levy_tau
                    t+=ceil(tau/self.dt)
                    t = min(self.L_time-1, t)
                    chain[it, t]=1
        return chain

    def forward(self):
        V = np.ones((1, self.classes))*self.layers[0].neuron.E_L
        spikes = []
        #input_spikes = self.encoding()
        input_spikes = np.ones((self.input.shape[0], self.L_time))
        out_spikes = np.zeros((1, self.classes))
        for i in range(self.L_time - 1):
            spikes = input_spikes[:, i]
            for c, layer in enumerate(self.layers):
                spikes, v = layer.feed(spikes, self.train)
                if c==len(self.layers)-1:
                    V[-1]=np.where(spikes!=0, self.layers[-1].neuron.V_th, V[-1])
                    V = np.concatenate((V, v.reshape(1, self.classes)))
                    out_spikes = np.concatenate((out_spikes, spikes.reshape(1, self.classes)*self.time[i]))
        #Firing rate
        num_spikes = np.sum(out_spikes!=0, axis=0)
        dT = np.max(out_spikes, axis=0)-np.min(out_spikes, axis=0)
        dT =np.where(dT!=0, dT, 1)
        return input_spikes, (out_spikes!=0)*1, num_spikes/dT, V


