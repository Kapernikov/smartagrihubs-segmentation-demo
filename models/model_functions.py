import os
import yaml

from keras.models import model_from_json

#import configs.paths_config as paths
from models.architectures import dynamic_imported_unet as dynamic_imported_unet
from models.architectures import dynamic_unet as dynamic_unet

from omegaconf import OmegaConf


def create_model(shape, num_classes, filters, hyperspec=False, pca=False):
    """ Create a new model """

    if len(shape) == 2:
        if hyperspec and not pca:
            shape += (6,)
        else:            
            shape += (3,)

    implemented = True

    if implemented:
        model = dynamic_unet.UNet(shape, num_classes, filters=filters, dropout_rate=0.5).get_model()
    else:
        # Imported model ( tweaked )    
        model = dynamic_imported_unet.build_unet(shape,num_classes,filters=filters)
    
    return model

def load_model(model_name):
    """ Load model from h5 file """
     # read config paths
    cfg = OmegaConf.load('configs/env.yaml')

    print('Loading model', model_name)
    json_file = open(os.path.join(cfg.DIRS.history, model_name, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # Load weights into new model
    model.load_weights(os.path.join( cfg.DIRS.history, model_name, 'model.h5'))
    print("Loaded model and weights : ", model_name)
    return model