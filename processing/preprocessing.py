import os
import random
import yaml
import spectral
import cv2 as cv
import numpy as np
import tensorflow as tf

from glob import glob, iglob

from omegaconf import OmegaConf
from keras.preprocessing.image import ImageDataGenerator

def read_image_cv(path: str) -> np.array:
    """ basic opencv image reader """
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img

def read_image_spectral(path: str) -> np.array:
    """ read hyperspectral image and return selected bands """
    img = spectral.open_image(path)
    bands = tuple(range(img.shape[-1]))
    img_bands = img.read_bands(bands, use_memmap=True)  
    return img_bands  

def reduce_bands_image(data, nbands=3):
    """ We reduce the number of bands of the image using PCA """
    from sklearn.decomposition import PCA
    img_dims = data.shape[:2]
    pca = PCA(n_components=nbands)
    
    data = data.reshape(-1, data.shape[2])
    data = pca.fit_transform(data)
    return data.reshape(img_dims + (nbands,))

def process_mask(mask: np.array) -> np.array:
    """ mask processing: convert to gray, normalize and change type """
    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    mask = mask/255
    mask = mask.astype('uint8')
    # expand array shape: (X,X) -> (X,X,1)
    mask = np.expand_dims(mask, axis=2)
    return mask

def add_overlap_masks(masks: list) -> list:
    """ create an overlap mask with inverted background """
    mask_overlap = sum(masks)
    mask_overlap = 1 - mask_overlap

    masks.insert(0, mask_overlap)
    return masks

def write_image_cv(img: np.array, img_name: str) -> None:
    """ basic opencv image writer """
    FOLDER = 'tmp_images'
    os.makedirs(FOLDER, exist_ok=True)

    if img.max() <= 1:
        img = img*255

    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    cv.imwrite(os.path.join(FOLDER, img_name), img)

def replicate_dataset(X: list, Y: list, N_rep : int = 1, shuffle: bool=False):
    """ replicate dataset with N_rep copies """
    # Shuffling Xs and Ys ( zipping them together to keep the same shuffling order )
    SEED = 99
    training = list(zip(X * N_rep, Y * N_rep))
    if shuffle:
        random.seed(SEED)
        random.shuffle(training)
    X, Y = zip(*training)
    return X, Y 

def preprocess_data_from_images(data_path, categories, hspectral=[False, False], shape=None):
    '''
    Returns a list of images and their respective one-hot-encoded masks.
        Parameters:
            categories_dict: dictionary of categories
            shape: image dimensions that will be provided to the model
            option: loads the data from the train or test folder
            hspectral: list with hspectral and pca paramters
            data_path: path to the folder that contains the train and test folders    
    '''

    print('Preprocessing data')
    training_data_X = []
    training_data_Y = []

    # data format 
    hyperspec, pca = hspectral[0], hspectral[1] 

    if hyperspec:
        image_paths = glob(data_path + '/images/*.hdr')
    else:
        image_paths = glob(data_path + '/images/*.*')

    # Looping through each annotated image
    for image_path in image_paths:

        filename = image_path.split('/')[-1]

        if hyperspec:
            img = read_image_spectral(image_path)
            if pca:
                # reduce to 3 channels (default)
                img = reduce_bands_image(img) 
        else:
            img = read_image_cv(image_path)

        masks = []
        for category in categories:

            if hyperspec:
                mask_filename = filename.replace('.hdr', '_' + category + '.png')
            else:
                mask_filename = filename.replace('.png', '_' + category + '.png')

            mask_path = image_path.replace(filename, mask_filename).replace('images', 'masks')

            mask = read_image_cv(mask_path)
            mask = process_mask(mask)

            masks.append(mask)

        masks = add_overlap_masks(masks)
        # concatenating masks into a single array
        msk = np.concatenate(tuple(masks), axis=-1)
        
        # == transformation == #
        if shape != None:
            img = cv.resize(img, shape, interpolation=cv.INTER_NEAREST)
            msk = cv.resize(msk, shape, interpolation=cv.INTER_NEAREST)

        training_data_X.append(img) # (N_IMG, W,H, N_CHANNELS)
        training_data_Y.append(msk) # (N_IMG, W,H, N_CLASSES+1), background is not counted as a class

    assert len(training_data_X) == len(training_data_Y)
    print('Num of images loaded', len(training_data_X))

    training_data_X, training_data_Y = replicate_dataset(X=training_data_X, 
                                                         Y=training_data_Y, 
                                                         N_rep=1,
                                                         shuffle=True)

    return np.array(training_data_X, dtype=np.float32), np.array(training_data_Y, dtype=np.float32)


def samples_to_datagenerator(samples):
    '''
    # === NOT IN USE === #
    Creates and returns generators from an array of samples (X, Y)
        Args:
            samples: array of samples [X, Y]
        Returns:
            zip(X_generator,Y_generator)
    '''
    cfg = OmegaConf.load('configs/env.yaml')
    
    print('Creating data generators')
    images, masks = samples

    data_gen_args = dict(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=45,
        zoom_range=0.3,
        width_shift_range=0.3,
        height_shift_range=0.3,
        fill_mode='nearest'
    )

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # Provide the same seed and keyword arguments to the fit and flow methods seed = 1
    # compute quantities required for featurewise normalization
    image_datagen.fit(images, augment=True, seed=cfg.DATA.seed )
    mask_datagen.fit(masks, augment=True, seed=cfg.DATA.seed )

    image_generator = image_datagen.flow(images,
                                         batch_size=cfg.TRAINING.batch_size,
                                         seed=cfg.DATA.seed )

    mask_generator = mask_datagen.flow(masks, 
                                        batch_size=cfg.TRAINING.batch_size,
                                        seed=cfg.DATA.seed )

    # combine generators into one which yields image and masks
    return zip(image_generator, mask_generator)
