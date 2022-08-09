import os
import cv2 as cv
import numpy as np
import namegenerator as namegen
import tensorflow as tf
import yaml

from models import evaluate
from models.model_functions import create_model, load_model
from models.saving import (load_params, save_losses, save_model, save_params, save_predictions)
from models.train_test.test import test_model
from models.train_test.train import train_model
from processing.postprocessing import encode_masks_to_rgb, create_color_map
from processing.preprocessing import (preprocess_data_from_images_dev, generate_categories_dict)
from utils.dir_processing import clean_folder, save_metadata

from utils.plotting import write_dataset
from sklearn.model_selection import train_test_split

from omegaconf import OmegaConf

def create_metadata(model_name):
    """ not necessary if we use mlflow """
    import datetime
    metadata = {}
    metadata['modelname'] = model_name
    metadata['timestamp'] = datetime.datetime.now().isoformat()
    return metadata

def main():

    # Check if laptop has a GPU available for computation
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Disabling logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # environment variables with paths
    cfg_path = OmegaConf.load('configs/paths.yaml')

    cfg = OmegaConf.load('configs/env.yaml')

    categories = cfg.MODEL.categories
    categories_dict = generate_categories_dict(categories)
    color_map = create_color_map(categories)

    newstr = cfg.DATA.img_dims.replace("(", "")
    newstr = newstr.replace(")","")
    desired_input_dimensions = tuple(map(int, newstr.split(', ')))

    # == MODEL == #
    if cfg.MODEL.model_name == '': 
        model_name = namegen.gen(separator='_')

        num_classes = len(categories)+1
        model = create_model(desired_input_dimensions, num_classes, cfg.TRAINING.filters, cfg.HYPERSPECTRAL.hyperspec, cfg.HYPERSPECTRAL.pca)
        model.summary()
    else:
        model_name = cfg.MODEL.model_name

        model = load_model(model_name)

        load_params(model_name)

        model.summary()

    PATH_LOG = os.path.join(cfg_path.DIRS.history, model_name)  # store model log
    PATH_RES = os.path.join(cfg_path.DIRS.results, model_name)  # store results

    metadata = create_metadata(model_name)

    # === DATASET LOADING AND PREPROCESSING === #
    X, y = preprocess_data_from_images_dev(data_path = cfg_path.DIRS.data, 
                                           shape = desired_input_dimensions,
                                           categories=categories,
                                           hspectral = [cfg.HYPERSPECTRAL.hyperspec, cfg.HYPERSPECTRAL.pca])

    #=== TRAIN/TEST SPLIT === #
    test_size = 1.- cfg.DATA.train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        shuffle=True,
                                                        random_state=cfg.DATA.seed)

    print(f'Number of TRAIN images: {len(X_train)}')
    print(f'Number of TEST images: {len(X_test)}')

    if cfg.MODEL.train:

        print('Training mode')

        os.makedirs(PATH_LOG, exist_ok=True)
        os.makedirs(os.path.join(PATH_RES, model_name), exist_ok=True)
        
        # == TRAINING == #
        history = train_model(model, PATH_RES, model_name, Xs=X_train, Ys=y_train)
        
        # == Saving model informations == #
        save_losses(history, PATH_RES)
        save_model(model, PATH_LOG)
        save_metadata(metadata, PATH_LOG)
        save_params(PATH_LOG)
    
    if cfg.MODEL.test:

        print('Inference mode')

        # == INFERENCE == #
        y_pred = test_model(model, X_test, prediction_threshold=0.8)
        #write_dataset(X_test, predictions)

        # == Confusion matrices == #
        confusion_classes, imgs_labels = evaluate.get_confusion_indices(y_test,
                                                                        y_pred,
                                                                        categories_dict=categories_dict,
                                                                        pixel_thres=10,
                                                                        meanIoU_threshold=0.7)            
            
        for class_name,confusion_matrix in confusion_classes.items():        
            evaluate.save_confusion_matrix(confusion_matrix, model_name, class_name, class_counter=None)

        clean_folder(PATH_RES)

        # encode ground truth and prediction masks
        y_test_en, y_pred_en = encode_masks_to_rgb(y_test, y_pred, color_map)
        save_predictions(X_test, 
                         y_test_en,
                         y_pred_en,
                         PATH_RES,
                         imgs_labels,
                         confusion_classes,
                         color_map)

if __name__ == "__main__":
    main()
