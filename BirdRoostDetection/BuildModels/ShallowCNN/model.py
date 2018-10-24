from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Flatten, Dense
import keras


def build_model(inputDimensions, lr=.0001):
    """Build the shallow CNN model.

    Args:
        inputDimensions: The image dimention input must be in the following
            format: (img_width, img_height, channel)
        lr: The learning rate of the network.

    Returns:
        The shallow CNN model.
    """

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5),
                     input_shape=inputDimensions))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    num_classes = 2
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.adam(lr),
                  metrics=['accuracy'])

    return model

def smaller_build_model(inputDimensions, lr=.0001):
    """Build the shallow CNN model.

    Args:
        inputDimensions: The image dimention input must be in the following
            format: (img_width, img_height, channel)
        lr: The learning rate of the network.

    Returns:
        The shallow CNN model.
    """

    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5),
                     input_shape=inputDimensions))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (5, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(25))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    num_classes = 2
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.adam(lr),
                  metrics=['accuracy'])

    return model
