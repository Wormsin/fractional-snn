import numpy as np
from model.net_params import Am, Ap, dt, tm, tp, stdp_rate


def stdp(P, M, weights, out_spk, in_spk, A_minus = Am, dt=dt, t_m=tm, A_plus=Ap, t_p=tp, lr=stdp_rate):
    for i in range(len(M)):
        weight = weights[:, i]
        ind_w = np.argmax(weight*in_spk)
        spike = np.zeros(weight.shape)
        spike[ind_w] = 1
        M[i] = M[i] + out_spk[i]*A_minus - M[i]*dt/t_m
        P[i] = P[i] +  spike*A_plus - P[i]*dt/t_p
        dw_m = lr*M[i]*weight*in_spk*spike
        dw_p = lr*P[i]*weight*out_spk[i]
        weights[:, i] += np.where(weight+dw_m>-1, dw_m, -1)  + np.where(weight+dw_p<=1, dw_p, 0)
        #weights = np.expand_dims(weights, 1)
    return P, M, weights