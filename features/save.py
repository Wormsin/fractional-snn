import numpy as np
import pandas as pd
import os


def save_weights(model):
    if not os.path.isdir('weights'):
        os.mkdir('weights')
    for c, layer in enumerate(model.layers):
        data = {
             'weights':layer.weights
        }
        df = pd.DataFrame(data)
        df.to_csv(f'weights/layer{c}.csv', index = False)
         
      

