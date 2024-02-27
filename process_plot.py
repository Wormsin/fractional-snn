import numpy as np 
import matplotlib as mpl
import matplotlib.pyplot as plt 
import csv
from scipy.special import gamma 
import numfracpy as nfr
from scipy.optimize import curve_fit

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
    p = ax.plot(x, y, marker ='o', markersize = 8, linewidth = 0)
    if log:
        ax.yscale('log')
        ax.xscale('log')
    return p

def get_model_plot(ax, x, y, log):
    p = ax.plot(x, y, linewidth = 3, zorder = 3)
    if log:
        ax.yscale('log')
        ax.xscale('log')
    return p

def mittag_leffler(T, alpha, tau_m):
    MLF = []
    for t in T:
        mlf = 0.1*nfr.Mittag_Leffler_one(-((t)**(alpha))/tau_m, alpha)
        MLF.append(mlf)
    return np.array(MLF)

def plot_model(ax, x, y, p0):
    params, covariance = curve_fit(mittag_leffler, x, y, p0)
    fit = mittag_leffler(x, *params)
    print(*params, covariance)
    pm = get_model_plot(ax, x, fit, False)
    return pm, *params

file_name = "data_tio2_n.csv"
xn, yn,  = extract_data(file_name)
file_name = "data_tio2.csv"
x, y,  = extract_data(file_name)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_axes([0.1,0.1,0.5,0.8])
ax.set_xlabel('t, mcs', fontweight = 'bold', size = 10)
ax.set_ylabel('I(V)', fontweight = 'bold', size = 10)


pm, alfa, _  = plot_model(ax,x[x>x[np.argmax(y)]], y[x>x[np.argmax(y)]], [0.8, 2])
pmn, alfan, _ = plot_model(ax, xn[xn>0], yn[xn>0], [0.5, 2])

#fit = mittag_leffler(x[x>x[np.argmax(y)]], 0.079, 0.005)
#pm = get_model_plot(ax, x[x>x[np.argmax(y)]], fit, False)

pn = plot_data(ax, xn, yn, False)
p = plot_data(ax, x, y, False)
ax.legend(handles = [pn[0], p[0], pmn[0], pm[0]], labels = ['N-TiO2', 'TiO2', f'alpha = {np.round(alfan, 2)}', f'alpha = {np.round(alfa, 3)}'])
ax.grid(True,which='major',axis='both',alpha=0.3)
plt.show()