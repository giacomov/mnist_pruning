#!/usr/bin/env python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np
import os

from training import train
from models import get_model


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

    model = train(get_model(input_shape, num_classes),
                  x_train, y_train, batch_size, epochs, x_test, y_test)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    keras_file = "mnist_not_optimized.h5"
    print('Saving model to: ', keras_file)
    tf.keras.models.save_model(model, keras_file, include_optimizer=False)


if __name__ == "__main__":

    cmd = argparse.ArgumentParser("Split MNIST dataset in train/test")

    cmd.add_argument("-b", "--batch_size", default=128, help="Batch size for training")
    cmd.add_argument("-e", "--epochs", default=10, help="Number of epochs")
    cmd.add_argument("-d", "--dataset", default="dataset.npz",
                     type=lambda x: os.path.abspath(os.path.expandvars(os.path.expanduser(x))))

    args = cmd.parse_args()

    go(args.batch_size, args.epochs, args.dataset)
