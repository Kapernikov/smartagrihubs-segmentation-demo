import math
import os
import random
import spectral
from glob import glob, iglob
import yaml

import cv2 as cv
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from pycocotools.coco import COCO
from skimage import io as io

def resize(input_image, input_mask, img_dimensions: tuple):
    '''
    Resize image and mask to same size.
        Parameters:
            input_image: image to be resized
            input_mask: mask to be resized
            img_dimensions: tuple of image dimensions
    '''

    input_image = tf.image.resize(
        input_image, img_dimensions, method="nearest")
    input_mask = tf.image.resize(
        input_mask, img_dimensions, method="nearest")
    return input_image, input_mask


def generate_data(annotations_dir_path, categories_dict, raw_dataset_folder, cleaned_dataset_path,
                  desired_dimensions=None, train_test_split_ratio=0.7):
    '''
    Generate data from CVAT Folder and splits it into a train and test folder.
            Parameters:
                annotations_dir_path: path to the CVAT folder
                categories_dict: dictionary of categories
                raw_dataset_folder: path to the raw dataset folder
                cleaned_dataset_path: path to the output folder that will receive the cleaned dataset
                desired_dimensions: tuple of desired image dimensions if resizing is needed
                train_test_split_ratio: ratio of train and test dataset 
    '''
    with open('configs/paths.yaml') as f:
        paths = yaml.load(f, Loader=yaml.FullLoader)

    # Check if created directories inside cleaned_dataset contain images
    if not len(list(iglob(cleaned_dataset_path + '/**/*.png', recursive=True))) == 0:
        print('Data already generated')

    else:
        print('Generating cleaned data')
        annotations_filenames = iglob(
            annotations_dir_path + '/**/*.json', recursive=True)
        coco_collections = []

        # Loading coco collections
        for filename in reversed(list(annotations_filenames)):
            coco_collections.append(COCO(filename))

        print('All categories merged from coco collections :', categories_dict)

        # Generate a mask for each class
        # Loop through each class
        for (k, coco_collection) in enumerate(coco_collections):
            print('Coco collection :', k)
            # Getting all images from collection
            CatIds = coco_collection.getCatIds()
            print('Number of categories : ', len(CatIds))
            ImgsIds = coco_collection.getImgIds(imgIds=[], catIds=[])
            print('Number of images ids : ', len(ImgsIds))
            # Loading imgs informations
            imgs = coco_collection.loadImgs(ids=ImgsIds)
            print("Total images loaded from collection", len(imgs))

            # Getting all image ids that have annotations
            ImgsIds_with_annotations = []
            for catId in CatIds:
                ImgsIds_with_annotations += coco_collection.getImgIds(
                    imgIds=[], catIds=catId)
            # Removing duplicate ids
            ImgsIds_with_annotations = list(set(ImgsIds_with_annotations))

            # Split ratio for each category
            print('Number of images with annotations : ',
                  len(ImgsIds_with_annotations))
            train_num = math.floor(
                len(ImgsIds_with_annotations) * train_test_split_ratio)
            test_num = len(ImgsIds_with_annotations) - train_num
            print("Splitting data to : \nTrain :",
                  train_num, "\nTest :", test_num)
            # input("Press Enter to continue...")
            counter = 0
            # Shuffling images
            random.shuffle(imgs)
            # Looping through each image)
            for (i, img) in enumerate(imgs):

                # If filename ends with .jpg ( not usable )
                if img['file_name'].endswith('.jpg'):
                    continue
                else:
                    # Loading image
                    image = io.imread(os.path.join(raw_dataset_folder, 'images',
                                                   img['file_name']))

                    # Loading annotations
                    img_annotations = coco_collection.getAnnIds(
                        imgIds=img['id'], catIds=[], iscrowd=None)

                    # If Image does not have any annotations, save in non annotated folder
                    if len(img_annotations) == 0:
                        filename = img['file_name'].split('/')[-1]
                        image_path = os.path.join(
                            cleaned_dataset_path, paths['non_annotated_dir_name'], filename)
                        io.imsave(image_path, image)

                    else:
                        # For each image, loop through each category and generate a mask if existing
                        for CatName, Ids in categories_dict.items():

                            if not isinstance(Ids, list):
                                Ids = [Ids]

                            img_annIds = coco_collection.getAnnIds(
                                imgIds=img['id'], catIds=Ids, iscrowd=None)

                            # If image has annotations for this category
                            if len(img_annIds) > 0:

                                # Loading annotations of image
                                anns = coco_collection.loadAnns(img_annIds)

                                # Looping through each annotation
                                for (j, ann) in enumerate(anns):

                                    # Generating a mask for each annotation
                                    mask = coco_collection.annToMask(ann)
                                    mask *= 255  # Because we have 0 and 1 --> 0 and 255
                                    # Saving mask
                                    if counter <= train_num:
                                        print('Saving train image :', i)
                                        option = 'train'
                                    else:
                                        print('Saving test image :', i)
                                        option = 'test'

                                    filename = 'ann_' + \
                                               str(i) + '_' + \
                                               img['file_name'].split('/')[-1]

                                    image_path = os.path.join(
                                        cleaned_dataset_path, option, 'images', filename)

                                    mask_path = os.path.join(
                                        cleaned_dataset_path, option, 'masks', CatName, filename)

                                    # Reshaping mask so that it'll have 3 dimensions
                                    mask = np.reshape(
                                        mask, (mask.shape[0], mask.shape[1], 1))

                                    # Resizing images so that they'll all have the same dimensions for performance purposes
                                    if desired_dimensions != None:
                                        # Resizing with distortion
                                        image, mask = resize(
                                            image, mask, desired_dimensions)

                                    io.imsave(mask_path, mask)
                            else:
                                pass
                        io.imsave(image_path, image)
                        counter += 1

def generate_categories_dict(predictable_categories):
    print('Generating categories dict')
    categories_dict = {}
    for i in range(len(predictable_categories) + 1):
        if i == 0:
            categories_dict['background'] = 0
        else:
            categories_dict[predictable_categories[i - 1]] = i
    return categories_dict


def preprocess_data_from_images_dev(data_path, categories, hspectral, shape=None):
    '''
    === SHOULD BE A CLASS (not a priority) === 
    Returns a list of images and their respective one-hot-encoded masks.
        Parameters:
            categories_dict: dictionary of categories
            shape: image dimensions that will be provided to the model
            option: loads the data from the train or test folder
            hspectral: list with hspectral and pca 
            data_path: path to the folder that contains the train and test folders    
    '''
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

def generate_categories_dict(predictable_categories):
    print('Generating categories dict')
    categories_dict = {}
    for i in range(len(predictable_categories) + 1):
        if i == 0:
            categories_dict['background'] = 0
        else:
            categories_dict[predictable_categories[i - 1]] = i
    return categories_dict


def preprocess_data_from_images(datafolder_path, predictable_categories, shape=None):
    '''
    Returns a list of images and their respective one-hot-encoded masks.
        Parameters:
            categories_dict: dictionary of categories
            shape: image dimensions that will be provided to the model
            option: loads the data from the train or test folder
            datafolder_path: path to the folder that contains the train and test folders    
    '''

    print('Preprocessing data')
    training_data_X = []
    training_data_Y = []

    # Looping through each annotated image
    for image_path in glob(datafolder_path + '/images/*.*'):
        # Getting image name
        filename = image_path.split('\\')[-1]

        # Reading image
        X = np.array(io.imread(image_path), dtype=np.uint8)

        # Array of masks of each category, they will later be merged into a single mask through one-hot-encoding
        Y = []

        # Initialize a matrix of zeros with same dimension as X (background with zeros)
        background = np.zeros((X.shape[0], X.shape[1], 1), dtype=np.uint8)

        # Looping through each category
        for category in predictable_categories:
            mask_filename = filename.replace('.png', '_' + category + '.png')

            mask_path = image_path.replace(
                filename, mask_filename).replace('images', 'masks')

            # We read the mask
            mask = io.imread(mask_path)
            mask = mask[:, :, -1]

            # We replace 255 to 1 ( for the one hot encoding)
            mask[mask == 255] = 1

            # Reshaping mask
            mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))

            # Appending to Y
            Y.append(mask)

        # Background mask is made from the 0s of the sum of the other masks
        # We loop through each class mask ( excluding background )
        for y in Y:
            # We sum them together so we'll have a mask with 0s and 1s ( 1 = classes )
            background += y

        # We invert the 1 ( not background ) and 0 values so that the background will have 1 as a representation
        background = 1 - background

        # Adding background mask in the beginning of the list
        Y.insert(0, background)

        # Concatenating each mask in the Y list into a single mask
        final_Y = None
        for (i, y) in enumerate(Y):
            if i == 0:
                final_Y = y
            else:
                final_Y = np.concatenate((final_Y, y), axis=-1)

        X = np.array(X, dtype=np.uint8)
        final_Y = np.array(final_Y, dtype=np.uint8)

        if shape != None:
            X = cv.resize(
                X, shape, interpolation=cv.INTER_NEAREST)

            final_Y = cv.resize(
                final_Y, shape, interpolation=cv.INTER_NEAREST)

        # X0 = img = (64, 128, 3) # training_data_X = (23,64,128,3)
        training_data_X.append(X)
        # Y0 = (64, 128, 4)  # training_data_Y = (23,64,128,4)
        training_data_Y.append(final_Y)

    # Shuffling Xs and Ys ( zipping them together to keep the same shuffling order )
    training = list(zip(training_data_X * 20, training_data_Y * 20))
    #random.shuffle(training)
    training_data_X, training_data_Y = zip(*training)

    print('Num of images loaded', len(training_data_X))
    return np.array(training_data_X), np.array(training_data_Y, dtype=np.float32)


def preprocess_data(categories_dict, shape, option=None, datafolder_path='data/processed'):
    '''
    Returns a list of images and their respective one-hot-encoded masks.
        Parameters:
            categories_dict: dictionary of categories
            shape: image dimensions that will be provided to the model
            option: loads the data from the train or test folder
            datafolder_path: path to the folder that contains the train and test folders    
    '''
    with open('configs/paths.yaml') as f:
        paths = yaml.load(f, Loader=yaml.FullLoader)

    with open('configs/env.yaml') as f:
        model_env = yaml.load(f, Loader=yaml.FullLoader)

    training_data_X = []
    training_data_Y = []
    
    class_counter = {}
    for category in categories_dict:
        class_counter[category] = 0
    
    if option is not None:
        datafolder_path += '/' + option

    # Looping through each annotated image
    for image_path in glob(datafolder_path+'/images/*.png'):
        # Getting image name
        filename = image_path.split('\\')[-1]
        # Reading image
        X = np.array(io.imread(image_path), dtype=np.uint8)

        # Array of masks of each category, they will later be merged into a single mask through one-hot-encoding
        Y = []

        # Initialize a matrix of zeros with same dimension as X (background with zeros)
        background = np.zeros((X.shape[0], X.shape[1], 1), dtype=np.uint8)

        # Fetch masks related to the image
        masks_paths = glob(datafolder_path+'/masks/*/' + filename)

        # Looping through each category
        for cat, key in categories_dict.items():
            # Ignoring background, it will be treated later
            if cat == 'background':
                pass
            else:
                # Checking if the image has a mask corresponding to this class
                has_cat = False
                for mask_path in masks_paths:
                    if cat in mask_path:
                        has_cat = True
                        class_counter[cat] += 1
                        break
                # If it does
                if has_cat:
                    # We read the mask
                    mask = io.imread(mask_path)
                    # We replace 255 to 1 ( for the one hot encoding)
                    mask[mask == 255] = 1
                    # Reshaping mask
                    #print('mask shape',mask.shape)
                    mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
                    #print('mask shape',mask.shape)
                    #input()
                    mask = np.reshape(mask, (mask.shape[0], mask.shape[1], 1))
                    #print('mask shape',mask.shape)
                    #print(mask[0][0])
                    #input()
                # If not
                else:
                    # Create a mask of zeros
                    mask = np.zeros(
                        (X.shape[0], X.shape[1], 1), dtype=np.uint8)

                # Appending to Y
                Y.append(mask)

        # Background mask is made from the 0s of the sum of the other masks
        # We loop through each class mask ( excluding background )
        for y in Y:
            # We sum them together so we'll have a mask with 0s and 1s ( 1 = classes )
            background += y

        # We invert the 1 ( not background ) and 0 values so that the background will have 1 as a representation
        background = 1 - background

        # Adding background mask in the beginning of the list
        Y.insert(0, background)

        # Concatenating each mask in the Y list into a single mask
        final_Y = None
        for (i, y) in enumerate(Y):
            if i == 0:
                final_Y = y
            else:
                final_Y = np.concatenate((final_Y, y), axis=-1)

        X = np.array(X, dtype=np.uint8)
        final_Y = np.array(final_Y, dtype=np.uint8)

        X = cv.resize(
            X, shape, interpolation=cv.INTER_NEAREST)

        final_Y = cv.resize(
            final_Y, shape, interpolation=cv.INTER_NEAREST)
        # X0 = img = (64, 128, 3) # training_data_X = (23,64,128,3)
        training_data_X.append(X)
        # Y0 = (64, 128, 4)  # training_data_Y = (23,64,128,4)
        training_data_Y.append(final_Y)

    # Looping through each non annotated image during training
    if (model_env['use_non_annotated_images_test'] == True and option == 'test') or (
            model_env['use_non_annotated_images_train'] == True and option == 'train'):
        print('Preprocessing non annotated images for', option)
        non_annot_count = 0
        for image_path in glob(datafolder_path + '/' + paths['non_annotated_dir_name'] + '/*.png'):
            non_annot_count += 1
            threshold = math.inf
            if option == 'test':
                threshold = model_env['num_of_non_annotated_images_test']
            elif option == 'train':
                threshold = model_env['num_of_non_annotated_images_train']

            if non_annot_count >= threshold:
                break
            # Getting image name
            filename = image_path.split('\\')[-1]

            # Reading image
            X = np.array(io.imread(image_path), dtype=np.uint8)

            # Array of masks of each category, they will later be merged into a single mask through one-hot-encoding
            Y = []

            # Initialize a matrix of zeros with same dimension as X (background with zeros)
            background = np.ones((X.shape[0], X.shape[1], 1), dtype=np.uint8)
            Y.append(background)
            # Looping through each category
            for cat, key in categories_dict.items():
                # Ignoring background, it will be treated later
                if cat != 'background':
                    # Create a mask of zeros
                    mask = np.zeros(
                        (X.shape[0], X.shape[1], 1), dtype=np.uint8)
                    # Appending to Y
                    Y.append(mask)
                else:
                    class_counter['background'] += 1

            # Concatenating each mask in the Y list into a single mask
            final_Y = None
            for (i, y) in enumerate(Y):
                if i == 0:
                    final_Y = y
                else:
                    final_Y = np.concatenate((final_Y, y), axis=-1)

            X = np.array(X, dtype=np.uint8)
            final_Y = np.array(final_Y, dtype=np.uint8)

            X = cv.resize(
                X, shape, interpolation=cv.INTER_NEAREST)
            final_Y = cv.resize(
                final_Y, shape, interpolation=cv.INTER_NEAREST)

            # X0 = img = (64, 128, 3) # training_data_X = (23,64,128,3)
            training_data_X.append(X)
            # Y0 = (64, 128, 4)  # training_data_Y = (23,64,128,4)
            training_data_Y.append(final_Y)

    # Shuffling Xs and Ys ( same way order )
    training = list(zip(training_data_X, training_data_Y))
    random.shuffle(training)
    training_data_X, training_data_Y = zip(*training)

    print('Num of samples for', option, ':', len(training_data_X))
    return np.array(training_data_X), np.array(training_data_Y, dtype=np.float32), class_counter


def samples_to_datagenerator(samples):
    '''
    Creates and returns generators from an array of samples (X, Y)
        Parameters:
            samples: array of samples [X, Y]
        Returns:
            zip(X_generator,Y_generator)
    '''
    with open('configs/env.yaml') as f:
        model_env = yaml.load(f, Loader=yaml.FullLoader)
        
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
    seed = model_env['seed']
    batch_size = model_env['batch_size']
    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)

    image_generator = image_datagen.flow(
        images, batch_size=batch_size, seed=seed)

    mask_generator = mask_datagen.flow(masks, batch_size=batch_size, seed=seed)

    # combine generators into one which yields image and masks
    return zip(image_generator, mask_generator)
