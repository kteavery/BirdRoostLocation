from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Activation, Flatten, Dense
import BirdRoostLocation.LoadSettings as settings
from BirdRoostLocation.BuildModels.ShallowCNN.coord import CoordinateChannel2D
import keras


def build_model(inputDimensions, lr=0.0001, coord_conv=False, problem="detection"):
    """Build the shallow CNN model.

    Args:
        inputDimensions: The image dimention input must be in the following
            format: (img_width, img_height, channel)
        lr: The learning rate of the network.
        coord_conv: Adds CoordConv layers

    Returns:
        The shallow CNN model.
    """
    print("Input dimensions:")
    print(inputDimensions)

    model = Sequential()

    if coord_conv == True:
        model.add(CoordinateChannel2D(input_shape=inputDimensions))
        model.summary()
        model.add(Conv2D(5, kernel_size=(5, 5)))
    else:
        # add input_shape param since Conv2D is first layer
        model.add(Conv2D(8, kernel_size=(5, 5), input_shape=inputDimensions))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    if coord_conv == True:
        model.add(CoordinateChannel2D(use_radius=True))
    model.add(Conv2D(8, (5, 5)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    if coord_conv == True:
        model.add(CoordinateChannel2D(use_radius=True))
    model.add(Conv2D(16, (5, 5)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    if coord_conv == True:
        model.add(CoordinateChannel2D(use_radius=True))
    model.add(Conv2D(32, (5, 5)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    if coord_conv == True:
        model.add(CoordinateChannel2D(use_radius=True))
    model.add(Conv2D(64, (5, 5)))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(500))
    model.add(Activation("relu"))
    # model.add(BatchNormalization())
    num_classes = 2  # detection: yes, no OR location: lat, long

    if problem == "detection":
        model.add(Dense(num_classes, activation="softmax"))
        model.compile(
            loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.adam(lr),
            metrics=["accuracy"],
        )
    else:  # localization
        model.add(Dense(num_classes, activation="sigmoid"))
        model.compile(
            loss=keras.losses.mean_squared_error,
            optimizer=keras.optimizers.adam(lr),
            metrics=["mae", "mape", "cosine"],
        )

    model.summary()

    return model
