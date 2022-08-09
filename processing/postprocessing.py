import cv2
import numpy as np
import colorsys

def HSVToRGB(h, s, v): 
 (r, g, b) = colorsys.hsv_to_rgb(h, s, v) 
 return (int(255*r), int(255*g), int(255*b)) 
 
def getDistinctColors(n): 
 huePartition = 1.0 / (n + 1) 
 return list((HSVToRGB(huePartition * value, 1.0, 1.0) for value in range(0, n)))

def create_color_map(categories):
    # background color set to black
    color_map = {'background': (0, 0, 0)}
    colors = getDistinctColors(len(categories))
    for category,color in zip(categories,colors):
        color_map[category] = color
    return color_map

def onehot_2RGB(onehot, color_map):
    '''
    Converts a one-hot encoded image to RGB based on a color map

            Parameters:
                    onehot (np.array): A one-hot encoded image with shape (height, width, num_classes)
                    color_map (dict): A dictionary mapping each class to a RGB color

            Returns:
                    onehot (np.array): An RGB image with shape (height, width, 3)
    '''

    # TODO : Make it variable for N classes
    onehot = cv2.cvtColor(onehot, cv2.COLOR_GRAY2RGB)

    for (i, color) in enumerate(color_map.values()):
        np.place(onehot[:, :, 0], onehot[:, :, 0] == i, color[0])
        np.place(onehot[:, :, 1], onehot[:, :, 1] == i, color[1])
        np.place(onehot[:, :, 2], onehot[:, :, 2] == i, color[2])

    return onehot


def encode_masks_to_rgb(Y, predictions, color_map):
    '''
    OLD: samples_OneHot_2RGB
    Converts samples of one-hot encoded ground truth and predictions to RGB images based on a color map.
    Returns two lists of RGB images.
            Parameters:
                    onehot (numpy array): A one-hot encoded image with shape (height, width, num_classes)
                    color_map (dict): A dictionary mapping each class to a RGB color
    '''

    def encode_mask(img, color_map):
        """ """
        # Returns the index of each maximum value along the last axis (classes)
        img = np.argmax(img, axis=-1)
        img = np.expand_dims(img, axis=-1)
        img = img.astype(dtype=np.uint8)
        img = onehot_2RGB(img, color_map)
        return img

    print('Converting GTs and predictions to RGB images')

    gts_2RGB = [encode_mask(Y[i,::], color_map) for i in range(Y.shape[0])]
    preds_2RGB = [encode_mask(predictions[i,::], color_map) for i in range(predictions.shape[0])]

    return gts_2RGB, preds_2RGB
