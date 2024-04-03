import numpy as np



def stdp(self, P, M, weights, out_spk, in_spk):
    weights = weights[:, 0]
    ind_w = np.argmax(weights*in_spk)
    spike = np.zeros(weights.shape)
    spike[ind_w] = 1
    M = M + out_spk*self.A_minus - M*self.dt/self.t_m
    P = P +  spike*self.A_plus - P*self.dt/self.t_p
    dw_m = self.lr*M*weights*in_spk*spike
    dw_p = self.lr*P*weights*out_spk
    weights += np.where(weights+dw_m>-1, dw_m, -1)  + np.where(weights+dw_p<=1, dw_p, 0)
    weights = np.expand_dims(weights, 1)
    return P, M, weights