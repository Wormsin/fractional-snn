import numpy as np 
import matplotlib.pyplot as plt 
import csv
from scipy.special import gamma 
import numfracpy as nfr
from scipy.optimize import curve_fit

def extract_data(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data_array = np.array(data)
    X, Y = data_array[1:, 0].astype(float), data_array[1:, 1].astype(float) 
    return X, Y

def plot_data(x, y, log):
    plt.plot(x, y, '.')
    if log:
        plt.yscale('log')
        plt.xscale('log')
    plt.xlabel('t, mcs')
    plt.ylabel('I(V)')

def plot_model(x, y, log):
    plt.plot(x, y)
    if log:
        plt.yscale('log')
        plt.xscale('log')
    plt.xlabel('t, mcs')
    plt.ylabel('I(V)')

def mittag_leffler(T, alpha, tau_m):
    MLF = []
    for t in T:
        mlf = 0.1*nfr.Mittag_Leffler_one(-((t)**(alpha))/tau_m, alpha)
        MLF.append(mlf)
    return np.array(MLF)


file_name = "data_tio2_n.csv"
xn, yn,  = extract_data(file_name)
file_name = "data_tio2.csv"
x, y,  = extract_data(file_name)

params, covariance = curve_fit(mittag_leffler, xn[xn>0], yn[xn>0], p0=[0.8, 2])
fit = mittag_leffler(xn[xn>0], *params)
print(*params, covariance)

plot_data(xn, yn, False, )
plot_data(x, y, False)
plot_model(xn[xn>0], fit, False)
plt.legend(['N-TiO2', 'TiO2', 'mlf'])
plt.show()