#!/usr/bin/env python
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import argparse
import timeit
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.python.client import timeline


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


def go(dataset, m1, m2):

    num_classes = 10

    x_train, y_train, x_test, y_test, input_shape = get_data(dataset, num_classes)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    timings = []

    for label, model_file in zip(['Original', 'After pruning'], [m1, m2]):

        print(f"\n{label}:")
        model = load_model(model_file)

        model.summary()

        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.train.AdamOptimizer(),
            metrics=['accuracy'])

        score = model.evaluate(x_test, y_test, verbose=0)
        print('\nTest loss:', score[0])
        print('Test accuracy:', score[1])

        # Timeit
        model = load_model(model_file)
        model.compile(
            loss=tf.keras.losses.categorical_crossentropy,
            optimizer=tf.train.AdamOptimizer(),
            metrics=['accuracy'],
            options=run_options, run_metadata=run_metadata)
        timing = timeit.timeit(lambda: model.predict(x_test), number=5)
        timings.append(timing)
        print(f"Timing: {timing:.2f} s for {x_test.shape[0]} items")
        print("------------------------------------")

        tl = timeline.Timeline(run_metadata.step_stats)
        ctf = tl.generate_chrome_trace_format()
        with open(f'{os.path.basename(model_file).split(os.path.sep)[0]}_chrome_tracing.json', 'w+') as f:
            f.write(ctf)

    print(f"\nSpeed-up: {(timings[0] / timings[1]):.2f}x")


if __name__ == "__main__":

    cmd = argparse.ArgumentParser("Measure performance")

    cmd.add_argument("-d", "--dataset", default="dataset.npz",
                     type=lambda x: os.path.abspath(os.path.expandvars(os.path.expanduser(x))))
    cmd.add_argument("-i", "--initial_model", default="mnist_not_optimized.h5",
                     type=lambda x: os.path.abspath(os.path.expandvars(os.path.expanduser(x))))
    cmd.add_argument("-f", "--final_model", default="mnist_pruned.h5",
                     type=lambda x: os.path.abspath(os.path.expandvars(os.path.expanduser(x))))

    args = cmd.parse_args()

    go(args.dataset, args.initial_model, args.final_model)
