from model import fc_model
import data_proc.encoding as code
import features
import numpy as np
import cv2
import os
from features import utils


def eval(model:fc_model.FC, weights_dir, data_dir, N_spk, dt, t_step, V_th, V_rest, alfa):
    model.train = False
    model.load_weights(weights_dir)
    dir_classes = os.listdir(data_dir)
    file_names = []
    for i, dir in enumerate(dir_classes):
        file_names.append(np.array(os.listdir(data_dir+'/'+dir)))
        file_names[i] =[os.path.join(data_dir, dir,f) for f in file_names[i]]
    imgs = np.dstack(file_names).flatten()
    for file_name in imgs:
        data = np.asarray(cv2.imread(file_name, cv2.IMREAD_GRAYSCALE),  dtype="float")
        input_spikes, nu = code.encoding(data/255, N_spk, model.L_time, dt, t_step)
        out_spks = model.forward(input_spikes)
    #plot out spikes
        utils.plot_v_out(model, model.L_time, model.dt, V_th, V_rest, alfa)
        utils.plot_outspks(out_spks.T, model.L_time, model.dt, model.classes, file_name)
        utils.plot_inspks(input_spikes, model.L_time, model.dt, nu, file_name)
