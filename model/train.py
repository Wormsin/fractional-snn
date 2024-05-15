from model import fc_model
import data_proc.encoding as code
from features import save
import numpy as np
import os
import cv2
from features import save, utils
from tqdm import tqdm


def train(model:fc_model.FC, epochs, data_dir, dt, t_step, V_th, V_rest, alfa, plot:bool):
    model.train = True
    dir_classes = os.listdir(data_dir)
    file_names = []
    for i, dir in enumerate(dir_classes):
        file_names.append(np.array(os.listdir(data_dir+'/'+dir)))
        file_names[i] =[os.path.join(data_dir, dir,f) for f in file_names[i]]
    imgs = np.dstack(file_names).flatten()
    for epoch in range(1, epochs+1):
        for c, file_name in enumerate(tqdm(imgs)):
            data =  np.asarray(cv2.imread(file_name, cv2.IMREAD_GRAYSCALE),  dtype="float")
            input_spikes, nu = code.encoding(data/255, model.L_time, dt, t_step)
            out_spks = model.forward(input_spikes)
            if plot:
                utils.plot_v_out(model, model.L_time, model.dt, V_th, V_rest, alfa)
                utils.plot_outspks(out_spks.T, model.L_time, model.dt, model.classes, file_name)
        print(f'epoch:{epoch}')
    # save weights
        save.save_weights(model, epoch)
    return model
