# Semantic segmentation for defect detection

This project is an implementation of the U-Net architecture for defect detection using semantic segmentation.

Table of contents
=================
- [Semantic segmentation for defect detection](#semantic-segmentation-for-defect-detection)
- [Table of contents](#table-of-contents)
  - [Installation](#installation)
  - [Quick start: training and evaluation](#quick-start-training-and-evaluation)
  - [Dataset structure](#dataset-structure)

<a name="installation"></a>
## Installation

Use [pipenv](https://pipenv.pypa.io/en/latest/) to install virtual environment:
```
pipenv shell
```

<a name="quick-start"></a>
## Quick start: training and evaluation

[Training](https://github.com/Kapernikov/smartagrihubs-segmentation-demo/blob/master/train.ipynb) and [inference](https://github.com/Kapernikov/smartagrihubs-segmentation-demo/blob/master/inference.ipynb) for defect detection in hazelnuts is described in notebooks with appropriate titles.

Alternatively, one can run the whole pipeline in command line:
```
python main.py
```

We also provide a sample model for inference located in `models_versioning/whiny_red_mastiff`.

<a name="dataset-structure"></a>
## Dataset structure

We use hazelnut subset of [the MVTEC anomaly detection dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad). For simplicity, we have combined three different defect classes (cuts, holes, and scratches) into one single `object` class representing defects.

We assume the images and masks to be arranged in the following structure:
```
data
├── images
│   └── img.png
└── masks
    ├── class1_object
    │   └── img.png
    └── class2_object
        └── img.png       
```

Here `data` is a path specified in a `data` variable in [configs/env.yaml](https://github.com/Kapernikov/smartagrihubs-segmentation-demo/blob/master/configs/env.yaml). Each image should have an independent mask for each class. The corresponding masks should be placed in the folders with class names.
