#!/usr/bin/env python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import os
import collections
from tensorflow.keras.models import load_model
from tfkerassurgeon.operations import delete_channels
import numpy as np

from training import train


def get_model(input_model):

    model = load_model(input_model)
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer="adam",
        metrics=['accuracy'])

    model.summary()

    # Computing the L1 norm of filter weights
    ordered_filters = collections.OrderedDict()

    for i, layer_name in enumerate(['conv2d', 'conv2d_1']):

        weight = model.get_layer(layer_name).get_weights()[0]
        weights_dict = {}
        num_filters = len(weight[0, 0, 0, :])

        # compute the L1-norm of each filter weight and store it in a dictionary

        for j in range(num_filters):
            weights_dict[j] = np.sum(abs(weight[:, :, :, j]))

        # sort the filter as per their ascending L1 value
        weights_dict_sort = sorted(weights_dict.items(), key=lambda kv: kv[1])

        print('ll norm conv layer {}\n'.format(i + 1), weights_dict_sort)

        # get the L1-norm of weights from the dictionary and plot it
        weights_value = []

        for elem in weights_dict_sort:
            weights_value.append(elem[1])

        ordered_filters[layer_name] = collections.OrderedDict(weights_dict_sort)

    # Remove filters
    f = 0.9

    new_model = model

    for layer_name in ordered_filters:
        weight = model.get_layer(layer_name).get_weights()[0]
        n = int(len(weight[0, 0, 0, :]) * f)

        channels_to_remove = [x for x in ordered_filters[layer_name]][:n]

        new_model = delete_channels(new_model, new_model.get_layer(layer_name), channels=channels_to_remove, copy=False)

    new_model.summary()

    return new_model


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


def go(batch_size, epochs, dataset, input_model):

    num_classes = 10

    x_train, y_train, x_test, y_test, input_shape = get_data(dataset, num_classes)

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    model = train(get_model(input_model),
                  x_train, y_train, batch_size, epochs, x_test, y_test)

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    keras_file = "mnist_pruned.h5"
    print('Saving model to: ', keras_file)
    tf.keras.models.save_model(model, keras_file, include_optimizer=False)


if __name__ == "__main__":

    cmd = argparse.ArgumentParser("Split MNIST dataset in train/test")

    cmd.add_argument("-b", "--batch_size", default=128, help="Batch size for training")
    cmd.add_argument("-e", "--epochs", default=12, help="Number of epochs")
    cmd.add_argument("-d", "--dataset", default="dataset.npz",
                     type=lambda x: os.path.abspath(os.path.expandvars(os.path.expanduser(x))))
    cmd.add_argument("-m", "--optimized_model", default="mnist_optimized.h5",
                     type=lambda x: os.path.abspath(os.path.expandvars(os.path.expanduser(x))))

    args = cmd.parse_args()

    go(args.batch_size, args.epochs, args.dataset, args.optimized_model)
