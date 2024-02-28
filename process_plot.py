import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import csv
from scipy.special import gamma 
import numfracpy as nfr
from scipy.optimize import curve_fit
from scipy.integrate import quad

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
#mpl.rcParams['font.family'] = 'Western'

def extract_data(file_name):
    with open(file_name, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
    data_array = np.array(data)
    X, Y = data_array[1:, 0].astype(float), data_array[1:, 1].astype(float) 
    return X, Y

def plot_data(ax, x, y, log):
    p = ax.plot(x, y, marker ='o',markerfacecolor='white', markersize = 8, linewidth = 0)
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')
    return p

def get_model_plot(ax, x, y, log):
    p = ax.plot(x, y, linewidth = 3)
    if log:
        ax.set_yscale('log')
        ax.set_xscale('log')
    return p

def mittag_leffler(T, A, alpha):
    mlf_arr = []
    for t in T:
        mlf = nfr.Mittag_Leffler_one(-(t/T[0])**(alpha), alpha)*A/(nfr.Mittag_Leffler_one(-(1)**(alpha), alpha))
        #mlf = A*nfr.Mittag_Leffler_one(-((t-T[0])/1e-6)**(alpha), alpha)/nfr.Mittag_Leffler_one(-((T[1]-T[0])/1e-6)**(alpha), alpha)
        mlf_arr.append(mlf)
    return np.array(mlf_arr)


def plot_model(ax, x, y, A, p0):
    #params, covariance = curve_fit(lambda x, alpha: mittag_leffler(x, A, alpha), x, y, p0)
    params = p0[0]
    fit = mittag_leffler(x, A, params)
    print(params)
    pm = get_model_plot(ax, x, fit, True)
    return pm, params

def creat_fig(data, names, fit):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_axes([0.1,0.1,0.5,0.8])
    ax.set_xlabel('t, mcs', fontweight = 'bold', size = 10)
    ax.set_ylabel('Normalized TRMC singnal', fontweight = 'bold', size = 10)
    p_arr = []
    alfa_arr = []
    pm_arr = []
    for file_name in data:
        x, y,  = extract_data(file_name)
        p = plot_data(ax, x, y, True)
        p_arr.append(p[0])
        if fit:
            pm, alfa = plot_model(ax,x[x>x[np.argmax(y)]], y[x>x[np.argmax(y)]], np.max(y), [0.35])
            #pm, alfa = plot_model(ax,x[x>9e-8], y[x>9e-8], np.max(y[x>9e-8]), [0.35])
            pm_arr.append(pm[0])
            alfa_arr.append(f'mlf, alpha = {np.round(alfa, 2)}')
    p_arr = p_arr+pm_arr      
    names = names + alfa_arr
    ax.legend(handles = p_arr, labels = names)
    ax.grid(True,which='major',axis='both',alpha=0.3)
    plt.show()


file_names = ["data_bare.csv"]
labels = ["bare film 20nm"]
creat_fig(file_names, labels, True)
