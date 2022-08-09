# Semantic segmentation for defect detection

This project is an implementation of the U-Net architecture for defect detection using semantic segmentation.

Table of contents
=================
- [Semantic segmentation for defect detection](#semantic-segmentation-for-defect-detection)
- [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [Quick start: training and evaluation](#quick-start-training-and-evaluation)
  - [Cashew-peanut dataset](#cashew-peanut-dataset)
  - [Dataset structure](#dataset-structure)
  - [Configuration](#configuration)
  - [Usage](#usage)

<a name="installation"></a>
## Installation

```
 pip install -r requirements.txt
```

<a name="quick-start"></a>
## Quick start: training and evaluation

Before running a model, make sure you have a correct [structure of dataset](#dataset-structure). To train and/or evaluate a model, run the main script. The computation will start according to the [configuration parameters](#configuration). 
```
python main_cashew_peanut.py
```

<a name="cashew-peanut-dataset"></a>
## Cashew-peanut dataset

Dataset is versioned with DVC. In order to get it, run the following command in the root folder of this repo 

```
dvc pull
```

<a name="dataset-structure"></a>
## Dataset structure

We assume the images and masks to be arranged in the following structure:
```
data
└──train/test 
   ├── images
   │   └── img.png
   └── masks
       ├── class1_name
       │   └── img.png
       └── class2_name
           └── img.png       
```

Here `data` is a path specified in a `processed_data_dir_path` variable in [configs/paths_config.py](https://github.com/obaumgartner/DL_Thesis_Defect-detection-using-UNet/blob/main/configs/paths_config.py). Each image should have an independent mask for each class. The corresponding masks should be placed in the folders with class names. `train` and `test` folders have fixed names and should have the same structure. Data from `train` folder will be used for training only and will be split into train and validation subsets according to configuration files.  

<a name="configuration"></a>
## Configuration

Data and model configuration is done through [configs/env.py](https://github.com/obaumgartner/DL_Thesis_Defect-detection-using-UNet/blob/main/configs/env.py) 

```
# Data configuration
    # Categories that we want to predict, all other categories will be considered as background
    predictable_categories = ['cashew', 'peanut']
    desired_input_dimensions = (128, 128)  # (Width, Height), needs to be dividble by 32
    train_test_split = 0.7
    validation_split = 0.4
    
# Model configuration
    # Model name to load from folder. If empty, a new model is created
    loaded_timestamp = "2022-06-02_21_55_08_475750"
    
    train = True # If True, the model will train.
    test = True

# Training configuration
    num_epochs = 50
    batch_size = 64
    filters = [32, 64, 128, 256] # Each number of filters represents a downsampling block (bottleneck included)
    loss_name = 'focal_tversky_loss'
    metrics_name = ['OneHotMeanIoU']
    optimizer_name = 'sgd'
```

Paths of used folders can also be changed through [configs/paths.py](https://github.com/obaumgartner/DL_Thesis_Defect-detection-using-UNet/blob/main/configs/paths_config.py)
```
raw_data_dir_path = 'data/raw'
processed_data_dir_path = 'data/processed'
non_annotated_dir_name = 'non_annotated'
models_versioning_dir_path = 'models_versioning'
results_directory_path = 'results'
tuning_directory_path = 'tuning'
```
<a name="usage"></a>
## Usage

Data preprocessing is done through preprocess_data() by passing these arguments : 
* Dictionary representing each class and its associated id
* Desired shape if the data needs to be reshaped and therefore be able to train the model on different input sizes without saving the images locally
* "train" or "test" option retrieve the data from one of these folders
* Path of the folder containing the processed dataset
```
Xs_train, Ys_train, classes_counter_train = preprocess_data(
        categories_dict={'background': 0, 'peanut': 1, 'cashew': 2},
        shape=(128,128),
        option='train',
        datafolder_path="data/processed")
```

Model creation/loading
```
# Creating model
model = create_model(shape=(128,128)
                     num_classes=3, 
                     filters =[32, 64, 128, 256])
# Loading model
timestamp = "2022-06-02_21_42_04_758988"
model = load_model(timestamp)
```

Training
```
history = train_model(
        model=model, 
        path_log='results/2022-06-02_21_42_04_758988', 
        Xs=Xs_train, 
        Ys=Ys_train)
```



Testing
```
predictions = test_model(
        model=model, 
        Ys=Ys_test,  
        prediction_threshold=0.8)
```
