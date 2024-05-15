import numpy as np
from scipy.special import gamma 
from math import ceil
from numpy.random import uniform
import os
import cv2
import gzip
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from features import utils

def read_ubyte(file, image_size, num, count):
    f = gzip.open(file,'r')
    f.read(count)
    buf = f.read(image_size * image_size * num)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num, image_size, image_size, 1)
    return data

def get_mnist(dir, num, number):
    if not os.path.isdir(dir):
        os.makedirs(dir)  
    images = read_ubyte('train-images-idx3-ubyte.gz', 28, num, 16)
    labels = read_ubyte('train-labels-idx1-ubyte.gz', 1, num, 8)
    labels = np.asarray(labels).squeeze()
    for c in range(num):
        if labels[c] == number:
            image = np.asarray(images[c]).squeeze()
            image = cv2.resize(image, (14, 14))
            cv2.imwrite(dir+'/'+f'img{c}.png', image)

def flick_mnist(data, L_time, dt, t_step):
    img =  np.asarray(cv2.imread(data, cv2.IMREAD_GRAYSCALE),  dtype="float")
    signals, nu = encoding(img/255, L_time, dt, t_step)
    #signals, rate =poisson_encode(img/255, L_time, dt, t_step)
    #utils.plot_inspks(signals, L_time, dt, nu, data)
    fig, ax = plt.subplots()
    ims = []
    for i in range(L_time):
        img = signals[:, i].reshape((28, 28))*255
        im = ax.imshow(img, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True, repeat=True)
    plt.show()

def encoding(data, L_time, dt, t_step):
    data = data.flatten()
    t_step = int(t_step/dt)
    nu = np.round(data, 1)
    nu = np.where(nu<0.1, 0.1, nu)
    nu = np.where(nu==1, 0.9, nu)
    rate = 2
    chain = np.zeros((len(data), L_time))
    for it in range(len(data)):
        t=0            
        for i in range(L_time):
            if i==t:
                U1 = uniform()
                U2 = uniform()
                U3 = uniform()
                poisson_tau = ((-np.log(U1))**(1/nu[it]))/(rate**(1/nu[it]))
                levy_tau = np.sin(nu[it]*np.pi*U2)*((np.sin((1-nu[it])*np.pi*U2))**(1/nu[it]-1))/(((np.sin(np.pi*U2))**(1/nu[it]))*((-np.log(U3))**(1/nu[it]-1)))
                tau = poisson_tau*levy_tau
                t+=ceil(tau/dt)
                t = min(L_time-1*t_step, t)
                chain[it, t:t+t_step]=1
                t+=t_step
    return chain, nu

def poisson_encode(data, L_time, dt, t_step):
    data = data.flatten()
    t_step = int(t_step/dt)
    rate = np.round(data, 1)
    rate = np.where(rate<0.01, 0.01, rate)
    rate = np.where(rate>0.4, rate*5, rate)
    chain = np.zeros((len(data), L_time))
    for it in range(len(data)):
        t=0            
        for i in range(L_time):
            if i==t:
                U1 = uniform()
                tau = (-np.log(U1))**(1/rate[it])
                t+=ceil(tau/dt)
                t = min(L_time-1*t_step, t)
                chain[it, t:t+t_step]=1
                t+=t_step
    return chain, rate


def periodic_signal(data, nu, N_spk, L_time, time, period, dt):
    rate = gamma(nu+1)*N_spk/(L_time**nu)
    chain = np.ones((data.shape[0], L_time))
    input_rate = data*rate
    chain = (np.sin((time+L_time*period*dt)/input_rate)+0.5)*chain
    return chain

def fractional_poisson_signal(nu, N_spk, T, dt, t_step, inputs):
    t_step = int(t_step/dt)
    L_time = int(T/dt)
    rate = gamma(nu+1)*N_spk/((T)**nu)
    chain = np.zeros((inputs, L_time))
    for it in range(inputs):
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
    return chain