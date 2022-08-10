import json
import os
import shutil
import yaml

from matplotlib import pyplot as plt, patches as mpatches

from utils.plotting import write_image_cv

from omegaconf import OmegaConf

# Save history of training and validation
def save_losses(history, path_log):
    """ """
    
    print('Saving losses')
    keys = list(history.history.keys())
    loss_keys,val_loss_keys = keys[:(len(keys)//2)],keys[(len(keys)//2):]
    for i, key in enumerate(loss_keys):
        tmp_fig = plt.figure(figsize=(10, 5))
        plt.plot(history.history[key], label=key)
        plt.plot(history.history[val_loss_keys[i]], label=val_loss_keys[i])
        plt.title(key.title())
        plt.ylabel(key.title())
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        tmp_filename = os.path.join(path_log, key + '.png')

        # Check if file already exists
        count = 1
        while os.path.isfile(tmp_filename):
            # increment filename
            tmp_filename = os.path.join(path_log, key + '_' + str(count) + '.png')
            count += 1
        tmp_fig.savefig(tmp_filename)
        plt.close(tmp_fig)

# Saving model informations ( summary, model to JSON, weights, env.py )
def save_model(model, path_log):
    """ """
    print('Saving model summary and weights')

    # Saving model summary
    with open(os.path.join(path_log, 'model_summary.txt'), 'w') as file:
        model.summary(print_fn=lambda x: file.write(x + '\n'))

    # Serialize model to JSON
    model_json = model.to_json()
    with open(os.path.join(path_log, 'model.json'), "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights(os.path.join(path_log, 'model.h5'))

# Save params
def save_params(path_log, cfg):
    """ """
    print('Saving model params')

    num_classes = len(cfg.MODEL.categories) + 1
    decay = cfg.TRAINING.learning_rate/cfg.TRAINING.num_epochs
    params = {
        'num_classes': num_classes,
        'desired_input_dimensions': cfg.DATA.img_dims,
        'seed': cfg.DATA.seed,
        'use_data_generator': cfg.DATA.use_data_generator,
        'num_epochs': cfg.TRAINING.num_epochs,
        'batch_size': cfg.TRAINING.batch_size,
        'validation_split': cfg.TRAINING.validation_split,
        'verbose': cfg.TRAINING.verbose,
        'optimizer_name': cfg.TRAINING.optimizer_name,
        'learning_rate': cfg.TRAINING.learning_rate,
        'loss_name': cfg.TRAINING.loss_name,
        'metrics_name': cfg.TRAINING.metrics_name,
        'shuffle':  cfg.TRAINING.shuffle,
        'decay' : decay
    }

    with open(os.path.join(path_log, 'params.json'), 'w') as file:
        json.dump(params, file)

def load_params(model_name, cfg):
    """ """
    path = os.path.join(cfg.DIRS.history, model_name, 'params.json')
    print('Loading parameters')

    with open(path) as json_file:
        params = json.load(json_file)
    print(params)

def save_result_as_figure(image, ground_truth, prediction, path, color_map):
    """ save figures with original image, ground truth and prediction masks """

    if image.max() >= 1.:
        image = image/255.

    fig = plt.figure(figsize=(10, 5))
    
    patches = []
    for i, p in enumerate(color_map.keys()):
        color = tuple(ci/255. for ci in color_map[p])
        patches.append(mpatches.Patch(color=color, label=p))

    fig.legend(handles=patches)

    rows = 1
    columns = 3

    fig.add_subplot(rows, columns, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Image")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)
    plt.imshow(ground_truth)
    plt.axis('off')
    plt.title("Ground Truth")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, columns, 3)
    plt.imshow(prediction)
    plt.axis('off')
    plt.title("Prediction")

    fig.savefig(path)
    plt.close('all')

def save_predictions(X, gts_2RGB, preds_2RGB, path_log, confusion_results, confusion_matrix, color_map):
    """ save images with results to local disc """

    def cleanup_results_folders(path: str) -> None:
        """ clean up results folders """
        if os.path.exists(path) and len(os.listdir(path)) > 0:
            shutil.rmtree(path)        

    def create_results_folders(path: str, labels: list) -> None:
        """ create folders for results recording """
        os.makedirs(path, exist_ok=True)
        folders = ['tp', 'tn', 'fp', 'fn']
        for label in labels:
            os.makedirs(os.path.join(path, label), exist_ok=True)
            for f in folders:
                os.makedirs(os.path.join(path, label, f), exist_ok=True)

    print('Saving predictions')
    labels = list(confusion_matrix.keys())

    base_path = os.path.join(path_log, 'predictions')
    cleanup_results_folders(base_path)
    create_results_folders(base_path, labels)

    # Show each image with its prediction and ground truth
    for i, (y, prediction, image, confusion_result) in enumerate(zip(gts_2RGB, preds_2RGB, X, confusion_results)):
        img_name = f'img_pred_{i}.png'
        for j, label in enumerate(labels):
            path = os.path.join(base_path, label, confusion_result[j], img_name)
            #print(image[:,:,:3].shape)
            save_result_as_figure(image[:,:,:3], y, prediction, path, color_map)