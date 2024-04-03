import numpy as np
from scipy.special import gamma 
from math import ceil
from numpy.random import uniform
import os
import cv2

def cross2cross(v, amount, train):
    if train:
        dir = f'data_proc/dataset/train'
    else:
        dir = f'data_proc/dataset/test'
    if not os.path.isdir(dir):
        os.makedirs(dir)
    #v = np.where(uniform(0, 1)>0.5, 1, 0)
    for c in range(amount):
        mask0 = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        mask1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        if v ==0:
            data = np.random.randint(150, 255, size=mask0.shape)*mask0-np.random.randint(0, 80, size=mask0.shape)*(mask0-1)
        elif v==1:
            data = np.random.randint(150, 255, size=mask1.shape)*mask1-np.random.randint(0, 80, size=mask1.shape)*(mask1-1)
        cv2.imwrite(dir+'/'+f'img{c}_{v}.png', data) 


def encoding(data, N_spk, L_time, dt, t_step):
    data = data.flatten()
    t_step = int(t_step/dt)
    nu = np.round(data, 1)
    nu = np.where(nu<0.2, 0.2, nu)
    nu = np.where(nu==1, 0.9, nu)
    rate = gamma(nu+1)*N_spk/((L_time*dt)**nu)
    chain = np.zeros((data.shape[0], L_time))
    for it in range(data.shape[0]):
        t=0            
        for i in range(L_time):
            if i==t:
                U1 = uniform()
                U2 = uniform()
                U3 = uniform()
                poisson_tau = ((-np.log(U1))**(1/nu[it]))/(rate[it]**(1/nu[it]))
                levy_tau = np.sin(nu[it]*np.pi*U2)*((np.sin((1-nu[it])*np.pi*U2))**(1/nu[it]-1))/(((np.sin(np.pi*U2))**(1/nu[it]))*((-np.log(U3))**(1/nu[it]-1)))
                tau = poisson_tau*levy_tau
                t+=ceil(tau/dt)
                t = min(L_time-1*t_step, t)
                chain[it, t:t+t_step]=1
                t+=t_step
    return chain, nu


def periodic_signal(data, nu, N_spk, L_time, time, period, dt):
    rate = gamma(nu+1)*N_spk/(L_time**nu)
    chain = np.ones((data.shape[0], L_time))
    input_rate = data*rate
    chain = (np.sin((time+L_time*period*dt)/input_rate)+0.5)*chain
    return chain