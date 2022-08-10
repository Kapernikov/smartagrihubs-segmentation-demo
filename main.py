import os
import tensorflow as tf
import namegenerator as namegen

from omegaconf import OmegaConf

from models.model_functions import create_model, load_model
from models.train_test.train import train_model
from models.saving import (load_params, save_losses, save_model, save_params)
from utils.utils import (create_metadata, save_metadata, create_color_map, create_category_dict)

from processing.preprocessing import preprocess_data_from_images
from processing.postprocessing import encode_masks_to_rgb
from sklearn.model_selection import train_test_split

from utils.dir_processing import clean_folder
from utils.utils import create_color_map, create_category_dict

from models import evaluate
from models.train_test.test import test_model
from models.saving import save_predictions

from sklearn.model_selection import train_test_split

def main():

    # Check if laptop has a GPU available for computation
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    # Disabling logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # load environment variables
    cfg = OmegaConf.load('configs/env.yaml')

    # used for inference only
    categ_dict = create_category_dict(cfg.MODEL.categories)
    color_map = create_color_map(cfg.MODEL.categories)

    print('Class categories', categ_dict)

    num_classes = len(cfg.MODEL.categories) + 1

    if cfg.MODEL.model_name == '': 
        # generate a new model name
        model_name = namegen.gen(separator='_')

        model = create_model(eval(cfg.DATA.img_dims), num_classes, cfg.TRAINING.filters)
    else:
        # use existing model with its proper name
        model_name = cfg.MODEL.model_name

        model = load_model(model_name)
        load_params(model_name)

    print(f'Model name is {model_name}')    
    
    metadata = create_metadata(model_name)

    model.summary()

    # === DATASET LOADING AND PREPROCESSING === #
    X, y = preprocess_data_from_images(data_path = cfg.DIRS.data, 
                                       shape = eval(cfg.DATA.img_dims),
                                       categories = cfg.MODEL.categories)

    #=== TRAIN/TEST SPLIT === #
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=cfg.DATA.test_split,
                                                        shuffle=cfg.TRAINING.shuffle,
                                                        random_state=cfg.DATA.seed)

    print(f'Number of TRAIN images: {len(X_train)}')
    print(f'Number of TEST images: {len(X_test)}')

    if cfg.MODEL.train:

        print('Training mode')

        # == TRAINING == #
        history = train_model(model, model_name, Xs=X_train, Ys=y_train, cfg=cfg)

        # == Saving model informations == #
        PATH_RESULTS = os.path.join(cfg.DIRS.results, model_name)
        PATH_LOG = os.path.join(cfg.DIRS.history, model_name)

        os.makedirs(PATH_LOG, exist_ok=True)

        save_losses(history, PATH_RESULTS)
        save_model(model, PATH_LOG)
        save_metadata(metadata, PATH_LOG)
        save_params(PATH_LOG, cfg)
    
    if cfg.MODEL.test:

        print('Inference mode')

        # == INFERENCE == #
        y_pred = test_model(model, X_test, prediction_threshold = cfg.TRAINING.prediction_threshold)

        # == Confusion matrices == #
        confusion_classes, imgs_labels = evaluate.get_confusion_indices(y_test,
                                                                        y_pred,
                                                                        categories_dict = categ_dict,
                                                                        pixel_thres = cfg.TRAINING.pixel_threshold,
                                                                        meanIoU_threshold = cfg.TRAINING.iou_threshold)            
            
        # encode ground truth and prediction masks
        y_test_en, y_pred_en = encode_masks_to_rgb(y_test, y_pred, color_map)

        PATH_RESULTS = os.path.join(cfg.DIRS.results, model_name)
        clean_folder(PATH_RESULTS)

        save_predictions(X_test, 
                         y_test_en,
                         y_pred_en,
                         PATH_RESULTS,
                         imgs_labels,
                         confusion_classes,
                         color_map)

if __name__ == "__main__":
    main()
