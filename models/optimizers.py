import math
import yaml

import tensorflow as tf

#from configs.env import model_env


def get_optimizer_by_name(opt_name,learning_rate):
    if opt_name == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    elif opt_name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    elif opt_name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate,epsilon=1e-08)
        

def get_lr_scheduler_by_name(lr_scheduler_name):
    if lr_scheduler_name == 'time_based':
        return lr_time_based_decay
    elif lr_scheduler_name == 'exponential':
        return lr_exp_decay
    
    elif lr_scheduler_name == 'constant_decay':
        return lr_time_constant_decay
    else:
        return None

def lr_time_constant_decay(epoch,lr):
    with open('configs/env.yaml') as f:
        model_env = yaml.load(f, Loader=yaml.FullLoader)
    decay = model_env['learning_rate']/model_env['num_epochs']
    learning_rate = lr - decay
    return round(learning_rate,6)

# Dynamic learning rate
def lr_time_based_decay(epoch, lr):
    with open('configs/env.yaml') as f:
        model_env = yaml.load(f, Loader=yaml.FullLoader)
    decay = model_env['learning_rate']/model_env['num_epochs']
    learning_rate = lr * 1 / (1 + decay * epoch)
    return round(learning_rate,6)

def lr_exp_decay(epoch, lr):
    with open('configs/env.yaml') as f:
        model_env = yaml.load(f, Loader=yaml.FullLoader)
    decay = model_env['learning_rate']/model_env['num_epochs']
    learning_rate = lr * math.exp(-decay * epoch)
    return round(learning_rate,6)

