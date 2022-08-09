import keras.backend as K
import tensorflow as tf


def get_loss_fn_by_name(name):
    '''
    Returns a loss function by name.
        - x_entropy
        - focal_tversky_loss
        - dice_loss_multilabel
        - cce
        - combined
    '''    

    if name == 'x_entropy':
        return x_entropy
    
    elif name == 'focal_tversky_loss':
        return focal_tversky_loss

    elif name == 'dice_loss_multilabel':
        return dice_loss_multi
    
    elif name == 'cce':
        return tf.keras.losses.CategoricalCrossentropy() 

    elif name == 'combined':
        return combined_loss_function
    else:
        return name


def dice_coeff_multi(y_true, y_pred,alpha=1,beta=1):
    # y shape [None, 128, 128, 3] = [batches,width,height,channels]
    # We want to keep the channels as batches
    # Basically, every permutation that puts dimension 3 ( channels ) to the first place will work
    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))
    # y shape [3, 128, 128, None]
    
    # Turn a nD tensor into a 2D tensor with same 0th dimension. 
    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)
    # y shape [3, 128 * 128 * None]

    true_pos = K.sum(y_true_pos * y_pred_pos, axis=1)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos), axis=1)
    false_pos = K.sum((1-y_true_pos)*y_pred_pos, axis=1) 

    alpha = 1 
    beta = 1

    return 2 * true_pos / (2 * true_pos + alpha*false_neg + beta*false_pos)


def combined_loss_function(y_true, y_pred):
    focal_loss = focal_tversky_loss(y_true, y_pred)
    dice_loss = dice_loss_multi(y_true, y_pred)
    return focal_loss + dice_loss

def dice_loss_multi(y_true, y_pred):
    loss = dice_coeff_multi(y_true,y_pred)
    return 1 - loss


def x_entropy(y_true, y_pred,output_classes=3):
    loss = 0
    for index in range(1,output_classes):    
        loss += tf.reduce_mean(
        K.categorical_crossentropy(y_true[:,:,:,index],y_pred[:,:,:,index]))
    return loss/3.0
    

# Weighted CCE
def weighted_categorical_crossentropy(weights=[1.0,1.0,1.0]):
    # weights = [0.9,0.05,0.04,0.01]
    def wcce(y_true, y_pred):
        Kweights = K.constant(weights)
        if not tf.is_tensor(y_pred): y_pred = K.constant(y_pred)
        y_true = K.cast(y_true, y_pred.dtype)
        return K.categorical_crossentropy(y_true, y_pred) * K.sum(y_true * Kweights, axis=-1)
    return wcce

# FOCAL LOSS FUNCTIONS

# Tversky function outputs a value between 0-1
def tversky(y_true, y_pred):
    smooth = 1
    # y shape [None, 128, 128, 3] = [batches,width,height,channels]
    # We want to keep the channels as batches
    # Basically, every permutation that puts dimension 3 ( channels ) to the first place will work
    y_true = K.permute_dimensions(y_true, (3,1,2,0))
    y_pred = K.permute_dimensions(y_pred, (3,1,2,0))
    # y shape [3, 128, 128, None]
    
    # Turn a nD tensor into a 2D tensor with same 0th dimension. 
    y_true_pos = K.batch_flatten(y_true)
    y_pred_pos = K.batch_flatten(y_pred)
    # y shape [3, 128 * 128 * None]

    # Keepdims = True, we can manipulate the weights on a class level.
    # Much faster way    

    true_pos = K.sum(y_true_pos * y_pred_pos, axis=1)
    # true_pos shape : Tensor([None]) = Dimension 1 = Scalar
    false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
    false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1) 
    
    # alpha + beta = 1
    # alpha large: penalize false negatives
    # alpha small: penalize false positives
    alpha = 0.6 # 0.7
    beta = 1 - alpha

    return (true_pos + smooth)/(true_pos + alpha*false_neg + beta*false_pos + smooth)


def focal_tversky_loss(y_true, y_pred):    
    tversky_ = tversky(y_true, y_pred)
    gamma = 0.75 # 0.75
    return K.sum(K.pow((1-tversky_), gamma))

