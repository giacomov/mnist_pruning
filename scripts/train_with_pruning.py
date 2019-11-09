#!/usr/bin/env python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import os
from tensorflow_model_optimization.sparsity import keras as sparsity
import numpy as np

from models import get_model
from training import train


def get_data(dataset, num_classes):

    d = np.load(dataset)
    x_train = d['x_train']
    y_train = d['y_train']
    x_test = d['x_test']
    y_test = d['y_test']
    input_shape = d['input_shape']

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, input_shape


def go(batch_size, epochs, dataset):

    num_classes = 10

    x_train, y_train, x_test, y_test, input_shape = get_data(dataset, num_classes)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    num_train_samples = x_train.shape[0]
    end_step = np.ceil(1.0 * num_train_samples / batch_size).astype(np.int32) * epochs
    print('End step: ' + str(end_step))

    pruning_params = {
        'pruning_schedule': sparsity.PolynomialDecay(initial_sparsity=0.50,
                                                     final_sparsity=0.90,
                                                     begin_step=2000,
                                                     end_step=end_step,
                                                     frequency=100)
    }

    model = train(get_model(input_shape, num_classes, pruning_params=pruning_params),
                  x_train, y_train, batch_size, epochs, x_test, y_test,
                  pruning=True)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    keras_file = "mnist_optimized.h5"
    print('Saving model to: ', keras_file)
    # Save removing pruning apparatus
    tf.keras.models.save_model(sparsity.strip_pruning(model), keras_file, include_optimizer=False)


if __name__ == "__main__":

    cmd = argparse.ArgumentParser("Split MNIST dataset in train/test")

    cmd.add_argument("-b", "--batch_size", default=128, help="Batch size for training")
    cmd.add_argument("-e", "--epochs", default=12, help="Number of epochs")
    cmd.add_argument("-d", "--dataset", default="dataset.npz",
                     type=lambda x: os.path.abspath(os.path.expandvars(os.path.expanduser(x))))

    args = cmd.parse_args()

    go(args.batch_size, args.epochs, args.dataset)
