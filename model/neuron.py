import numpy as np
from scipy.special import gamma 
from math import ceil
from numpy.random import uniform

class Fractional_LIF():
    def __init__(self, V_th, V_reset, E_L, spk_amp, g_L, C_m, dt, alfa, tref) -> None:
        self.V_th = V_th
        self.V_reset = V_reset
        self.E_L = E_L
        self.g = g_L
        self.C = C_m
        self.spk_amp = spk_amp
        self.dt = dt
        self.alfa = alfa
        self.tref = tref

            
    def mem_dynamics(self, v, dV, tr, N, i_pulse):
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
    
