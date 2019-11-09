import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow_model_optimization.sparsity import keras as sparsity


def get_model(input_shape, num_classes, pruning_params=None):

    l = tf.keras.layers

    layers = [
        l.Conv2D(
            32, 5, padding='same', activation='relu', input_shape=input_shape),
        l.MaxPooling2D((2, 2), (2, 2), padding='same'),
        l.BatchNormalization(),
        l.Conv2D(64, 5, padding='same', activation='relu'),
        l.MaxPooling2D((2, 2), (2, 2), padding='same'),
        l.Flatten(),
        l.Dense(1024, activation='relu', activity_regularizer=tf.keras.regularizers.l1(0.01)),
        l.Dropout(0.4),
        l.Dense(num_classes, activation='softmax')
    ]

    if pruning_params is not None:

        # Add pruning in the right places
        for i in [0, 3]:
            layers[i] = sparsity.prune_low_magnitude(layers[i], **pruning_params)

    model = tf.keras.Sequential(layers)

    return model
