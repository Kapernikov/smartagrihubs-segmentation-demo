import os
import shutil
from glob import iglob
import json

from pycocotools.coco import COCO


def create_missing_directories(cleaned_data_dir_path, categories_dict):
    '''
    Create dataset directories for training and testing folders and their subfolders depending on the classes.
    Structure represents as follow :
        - cleaner_folder
            - train
                - masks
                    - class
                - images
            - test
                - masks
                    - class
                - images

        Parameters:
            cleaned_data_dir_path (str): Path to the cleaned data directory
            categories_dict (dict): Dictionary containing all the categories
    '''
    
    # Creating missing directories if needed
    if os.path.isdir(cleaned_data_dir_path):
        print('Directories already created')
    else:
        print("Creating directories")
        # Root folder
        create_folder(cleaned_data_dir_path)

        # Train folder and subfolders
        create_folder(cleaned_data_dir_path+'/train/images')
        create_folder(cleaned_data_dir_path+'/train/masks')
        for CatName in categories_dict:
            create_folder(cleaned_data_dir_path+'/train/masks/'+CatName)
        
        # Test folder and subfolders
        create_folder(cleaned_data_dir_path+'/test/images')        
        create_folder(cleaned_data_dir_path+'/test/masks')
        for CatName in categories_dict:
            create_folder(cleaned_data_dir_path+'/test/masks/'+CatName)

        # Non annotated images folder
        create_folder(cleaned_data_dir_path+'/' +
                        model_env.non_annotated_dir_name)

def create_folder(directory_path):
    '''
    Create a folder if it doesn't exist.
    Parameters:
        directory_path (str): Path to the folder to create
    '''
    try:
        os.makedirs(directory_path)
    except:
        print(directory_path, "directory already existing")

def clean_history_directories(num_files_threshold,models_path):
    '''
    Delete directories of models with insufficient informations ( early stopping)
    Parameters:
        num_files_threshold (int): Minimum of number of files that a model must contain
        history_directory_path (str): Path to the folder that contains the models
    '''
    
    # Remove directories in output directory that contain less than num_files_threshold files
    if os.path.isdir(models_path):
        print('Cleaning history directory')
        for directory in os.listdir(models_path):
            if len(os.listdir(os.path.join(models_path, directory))) < num_files_threshold:
                shutil.rmtree(os.path.join(
                    models_path, directory))

def move_images(annotations_dir_path, from_folder, to_folder, category_id=[2]):
    '''
    Move images with a specific category ids to another folder 
    ( Used to move images with cluster of carrots out of the test set)
    Parameters:
        annotations_dir_path (str): Path to the annotations directory
        from_folder (str): Name of the folder to move from
        to_folder (str): Name of the folder to move to
        category_id ([int]): Ids of the categories
    '''

    if not isinstance(category_id, list):
        category_id = [category_id]

    image_paths = []
    for filename in list(iglob(annotations_dir_path+'/**/*.json', recursive=True)):
        coco_collection = COCO(filename)
        image_ids = coco_collection.getImgIds(catIds=category_id)
        # Get image name from ids
        image_paths += [coco_collection.loadImgs(ids=id)[0]['file_name']
                        for id in image_ids]
        
        print(image_paths)

    for image_path in image_paths:
        # Check if / in image name
        if '/' in image_path:
            image_name = image_path.split('/')[-1]
            # Check if image in from_folder
            image_path = os.path.join(from_folder, image_name)
            if os.path.exists(image_path):
                print('Moving image :', image_path)
                shutil.move(image_path, to_folder)
            else:
                print('Image not found :', image_path)

def clean_folder(base_path):
    '''
    Delete previous saved predictions based on the model timestamp/name.
    Parameters:
        base_path (str): Path to the results directory
    '''
    path = os.path.join(base_path, 'predictions')
    if os.path.isdir(path):
        shutil.rmtree(path)

