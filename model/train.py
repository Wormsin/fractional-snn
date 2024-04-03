from model import fc_model
import data_proc.encoding as code
from features import save
import numpy as np
import os
import cv2
from features import save


def train(model:fc_model.FC, epochs, data_dir, N_spk, dt, t_step):
    model.train = True
    for epoch in range(1, epochs+1):
        file_names = os.listdir(data_dir)
        for file_name in file_names:
            data = cv2.imread(data_dir+file_name)
            input_spikes = code.encoding(data, N_spk, model.L_time, dt, t_step)
            model.forward(input_spikes)
        print(f'epoch:{epoch}')
    # save weights
    save.save_weights(model)
    return model
