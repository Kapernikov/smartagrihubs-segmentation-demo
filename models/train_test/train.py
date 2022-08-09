import math
import os
import yaml

import keras
from keras import Model
from keras.callbacks import CSVLogger

import models.losses as losses
import models.metrics as metrics
import models.optimizers as optimizers
from processing.preprocessing import samples_to_datagenerator

from omegaconf import OmegaConf

def train_model(model: Model, path_log: str, modelname: str, Xs: list, Ys: list):

    cfg_path = OmegaConf.load('configs/paths.yaml')

    cfg = OmegaConf.load('configs/env.yaml')
    
    loss = losses.get_loss_fn_by_name(cfg.TRAINING.loss_name)
    
    my_lr_scheduler = keras.callbacks.LearningRateScheduler(
        optimizers.lr_time_based_decay, verbose=1)

    optimizer = optimizers.get_optimizer_by_name(
        cfg.TRAINING.optimizer_name, cfg.TRAINING.learning_rate)
    
    num_classes = len(cfg.MODEL.categories) + 1
    mtrcs = metrics.get_metrics_by_name(
        cfg.TRAINING.metrics_name, num_classes)

    model.compile(optimizer=optimizer, loss=loss,
                  metrics=mtrcs, run_eagerly=True)

    csv_logger = CSVLogger(os.path.join(
        path_log, 'log.csv'), append=True, separator=';')

    print('use data generator', cfg.DATA.use_data_generator)

    checkpoint_filepath = f"./{cfg_path.DIRS.history}/{modelname}"
    os.makedirs(checkpoint_filepath, exist_ok=True)
    checkpoint_filepath += '/best_model.hdf5'

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                                save_weights_only=True,
                                                                monitor='val_1Hot_Mean_iou',
                                                                mode='max',
                                                                save_best_only=True
                                                                )
    if not cfg.DATA.use_data_generator:
        history = model.fit(
            x=Xs,
            y=Ys,
            validation_split = cfg.TRAINING.validation_split,
            epochs = cfg.TRAINING.num_epochs,
            shuffle = cfg.TRAINING.shuffle,
            batch_size = cfg.TRAINING.batch_size,
            verbose = cfg.TRAINING.verbose,
            callbacks=[csv_logger, my_lr_scheduler, model_checkpoint_callback]
        )

    else:  # train model with generator

        validation_split = cfg.TRAINING.validation_split
        batch_size = cfg.TRAINING.batch_size
        num_epochs = cfg.TRAINING.num_epochs
        verbose = cfg.TRAINING.verbose
        shuffle = cfg.TRAINING.shuffle

        train_samples = Xs[:int(Xs.shape[0]*(1-validation_split))
                          ], Ys[:int(Ys.shape[0]*(1-validation_split))]

        val_samples = Xs[int(Xs.shape[0]*(1-validation_split))
                            :], Ys[int(Ys.shape[0]*(1-validation_split)):]

        train_generator = samples_to_datagenerator(train_samples)
        val_generator = samples_to_datagenerator(val_samples)

        # We take the ceiling because we do not drop the remainder of the batch
        def compute_steps_per_epoch(x): return int(
            math.ceil(1. * x / batch_size))

        steps_per_epoch = compute_steps_per_epoch(train_samples[0].shape[0])
        validation_steps = compute_steps_per_epoch(val_samples[0].shape[0])

        steps_per_epoch = int(len(train_samples[0])/batch_size)
        validation_steps = int(len(val_samples[0])/batch_size)

        print('Steps per epoch', steps_per_epoch)
        print('Validation steps', validation_steps)

        history = model.fit(
            train_generator,
            validation_data=val_generator,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            epochs=num_epochs,
            shuffle=shuffle,
            batch_size=batch_size,
            verbose=verbose,
            callbacks=[csv_logger, my_lr_scheduler, model_checkpoint_callback]
        )

    return history
