import os
import yaml

from keras.models import model_from_json

#import configs.paths_config as paths
from models.architectures import dynamic_imported_unet as dynamic_imported_unet
from models.architectures import dynamic_unet as dynamic_unet


def get_last_model(models_directory_path):
    '''
    Returns the last created model in the history folder.
    Will not work if model names are not in timestamp format.
    '''
    # Get all timestamps
    timestamps = os.listdir(models_directory_path)
    # Get all timestamps that are directories
    timestamps = [timestamp for timestamp in timestamps if os.path.isdir(
        os.path.join(models_directory_path, timestamp))]
    # Sort the timestamps
    timestamps.sort()
    # Get the last timestamp
    timestamp = timestamps[-1]
    return timestamp


def create_model(shape, num_classes, filters, hyperspec, pca):
    '''
    Create a new model.
    '''
    print('Creating model')

    if len(shape) == 2:
        if hyperspec and not pca:
            shape += (6,)
        else:            
            shape += (3,)

    # Imported model ( tweaked )    
    #model = dynamic_imported_unet.build_unet(shape,num_classes,filters=filters)
    
    # Implemented model
    model = dynamic_unet.UNet(shape, num_classes,filters=filters,dropout_rate=0.5).get_model()
    
    return model

def load_model(timestamp):
     # read config paths
    with open('configs/paths.yaml') as f:
        paths = yaml.load(f, Loader=yaml.FullLoader)

    print('Loading model', timestamp)
    json_file = open(os.path.join(paths['history_directory_path'],
                                  timestamp, 'model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # Load weights into new model
    model.load_weights(os.path.join(
        paths['history_directory_path'], timestamp, 'model.h5'))
    print("Loaded model and weights : ", timestamp)
    return model