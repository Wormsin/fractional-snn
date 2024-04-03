from model import fc_model
import data_proc.encoding as code
import features
import numpy as np
import cv2
import os
from features import utils


def eval(model:fc_model.FC, weights_dir, data_dir, N_spk, dt, t_step):
    model.train = False
    model.load_weights(weights_dir)
    file_names = os.listdir(data_dir)
    for file_name in file_names:
        data = cv2.imread(data_dir+file_name)
        input_spikes = code.encoding(data, N_spk, model.L_time, dt, t_step)
        out_spks = model.forward(input_spikes)
    #plot out spikes
        utils.plot_spks(out_spks, model.time, model.classes, data_dir+file_name)

