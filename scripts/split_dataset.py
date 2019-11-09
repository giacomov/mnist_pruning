#!/usr/bin/env python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np


def go():

    # input image dimensions
    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    if tf.keras.backend.image_data_format() == 'channels_first':

        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)

    else:

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    return x_train, x_test, y_train, y_test, input_shape


if __name__ == "__main__":

    cmd = argparse.ArgumentParser("Split MNIST dataset in train/test")

    x_train, x_test, y_train, y_test, input_shape = go()

    np.savez("dataset.npz",
             x_train=x_train,
             y_train=y_train,
             x_test=x_test,
             y_test=y_test,
             input_shape=input_shape)
