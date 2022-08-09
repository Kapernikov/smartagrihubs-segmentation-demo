import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import yaml

from omegaconf import OmegaConf

matplotlib.use("Agg")

# Pixel accuracy = the percent of pixels in your image that are classified correctly.
def OneHotPixelAccuracy(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)
    result = y_pred == y_true
    count_true = np.count_nonzero(result)
    val = (count_true/y_pred.size)
    return val

# One-Hot Mean IoU (multiclass)
def OneHotMeanIoU(y_true, y_pred, output_classes=3):
    m = tf.keras.metrics.OneHotMeanIoU(num_classes=output_classes)
    m.update_state(y_true, y_pred, sample_weight=None)
    result = m.result().numpy()
    return result

# Binary_MeanIoU
def binary_meanIoU(y_true, y_pred):
    m = tf.keras.metrics.MeanIoU(num_classes=2)
    m.update_state(y_true, y_pred)
    return m.result().numpy()

# Confusion matrix
def get_confusion_indices(y_trues, y_preds, categories_dict, pixel_thres=1, meanIoU_threshold=0.8):
    
    prediction_results = []
    
    dict = {}

    for j in range(1, len(categories_dict)):
        dict[str(j)] = {
        'fp': 0,
        'fn': 0,
        'tp': 0,
        'tn': 0
        }

    for (i, (y_true, y_pred)) in enumerate(zip(y_trues, y_preds)):
        
        prediction_result = []
        # range 1 == ignore background at index 0
        for cls in range(1, len(categories_dict)):
            key = str(cls)
            
            # Step that we don't need
            y_pred_cls = y_pred[:, :, cls]
            y_true_cls = y_true[:, :, cls]

            has_defect = False

            gt_pixels = np.count_nonzero(y_true_cls == 1)
            pred_pixels = np.count_nonzero(y_pred_cls == 1)

            if gt_pixels > pixel_thres:
                #print('GT has defect')
                has_defect = True
            else:
                #print('GT has no defect')
                pass
                
            if pred_pixels > pixel_thres:
                #print('Pred has defect')
                if has_defect == False:  # GT has no defect, but prediction has.
                    #print('False positive confirmed')
                    dict[key]['fp'] += 1
                    prediction_result.append('fp')
                else:
                    pred_meanIoU = binary_meanIoU(y_true, y_pred)
                    if pred_meanIoU > meanIoU_threshold:
                        dict[key]['tp'] += 1
                        prediction_result.append('tp')
                    else:
                        #print('False negative confirmed')
                        dict[key]['fn'] += 1
                        prediction_result.append('fn')

            else:
                if has_defect == False:
                    #print('Confirmed true negative, both GT and pred have no defect')
                    dict[key]['tn'] += 1
                    prediction_result.append('tn')
                else:
                    #print('False negative, GT has defect but prediction has no defect')
                    dict[key]['fn'] += 1
                    prediction_result.append('fn')

        prediction_results.append(prediction_result)
    
    categories = list(categories_dict.keys())
    categories.remove('background')
    categories.insert(0,'background')

    classes_dict = {}
    for i,category in enumerate(categories):
        if i == 0:
            continue

        classes_dict[category] = dict[str(i)]
    return classes_dict, prediction_results


def save_confusion_matrix(confusion_matrix,timestamp,class_name,class_counter=None):
 # read config paths
    cfg_path = OmegaConf.load('configs/paths.yaml')
    cfg = OmegaConf.load('configs/env.yaml')

    tp,tn,fp,fn = confusion_matrix['tp'], confusion_matrix['tn'], confusion_matrix['fp'], confusion_matrix['fn']
                
    positives = [tp,fp]
    negatives = [fn,tn]
    c = np.array([positives,negatives])
    accuracy = (tp+tn)/(tp+tn+fp+fn)
            
    ax = sns.heatmap(c, annot=True, fmt='', cmap='Blues',cbar=False)
    
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = True, labeltop=True)
    ax.set_title('True Values')
    ax.set_xlabel('Confusion matrix : '+str(class_name).title()+'- accuracy : '+str(round(accuracy*100))+'%')
    ax.set_ylabel('Predicted Values')
    
    if class_counter is not None:
        plt.annotate(class_counter, xy=(0,0), xycoords='axes fraction',fontsize=8)
    
    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['True','False'])
    ax.yaxis.set_ticklabels(['True','False'])
    
    confusion_matrices_path = os.path.join(cfg_path.DIRS.results, timestamp,'confusion_matrices')
    if not os.path.exists(confusion_matrices_path):
        os.makedirs(confusion_matrices_path)

    path = os.path.join(confusion_matrices_path, 'conf_matrix_'+class_name+'.png')
    if os.path.isfile(path):
       os.remove(path)
    plt.savefig(path, dpi=300)
    plt.close('all')    
