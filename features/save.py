import numpy as np
import pandas as pd
import os


def save_weights(model):
    if not os.path.isdir('weights'):
        os.mkdir('weights')
    for c, layer in enumerate(model.layers):
        np.savetxt(f'weights/layer{c}.csv', layer.weights,  
              delimiter = ",")
         
      

