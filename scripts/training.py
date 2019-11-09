import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')


def train(model, x_train, y_train, batch_size, epochs, x_test, y_test, pruning=False):

    # Fit
    if pruning:

        import tensorflow_model_optimization.sparsity

        callbacks = [
            tensorflow_model_optimization.sparsity.keras.UpdatePruningStep()
        ]

    else:

        callbacks = []

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=tf.train.AdamOptimizer(),
        metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=callbacks,
              validation_data=(x_test, y_test))

    return model
