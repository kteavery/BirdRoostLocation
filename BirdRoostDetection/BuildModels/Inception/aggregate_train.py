import random
import os.path
import numpy as np
from os import walk
from os.path import join
from os.path import splitext

from BirdRoostDetection import utils
from BirdRoostDetection.BuildModels import ml_utils
import BirdRoostDetection.LoadSettings as settings
from BirdRoostDetection.ReadData import BatchGenerator

from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import BatchNormalization, Activation, Dense
import keras
import os

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

dual_pol_fields = [utils.Radar_Products.reflectivity.fullname,
                   utils.Radar_Products.velocity.fullname,
                   utils.Radar_Products.cc.fullname,
                   utils.Radar_Products.diff_reflectivity.fullname]
legacy_fields = [utils.Radar_Products.reflectivity.fullname,
                 utils.Radar_Products.velocity.fullname]


def getListOfFilesInDirectory(dir, fileType):
    """Search a directory for files of a specific type. Return filepath list.

    Args:
        dir: String, root path. i.e. /tmp/images/
        fileType: String, look for files of a specific type in directory,
            i.e ".txt" or ".png"

    Returns:
        list, a list of file paths to files of type fileType
    """
    fileNames = []
    for root, dirs, files in walk(dir):
        for f in files:
            if splitext(f)[1].lower() == fileType:
                fullPath = join(root, f)
                fileNames.append(fullPath)
    return fileNames


def create_image_lists(radar_product):
    result = {}
    batch_generator = BatchGenerator.Color_Image_Batch_Generator(
        ml_label_csv=settings.LABEL_CSV,
        ml_split_csv=settings.ML_SPLITS_DATA,
        high_memory_mode=False)

    ml_label_set = [batch_generator.no_roost_sets, batch_generator.roost_sets]
    if (radar_product == utils.Radar_Products.diff_reflectivity or
            radar_product == utils.Radar_Products.cc):
        ml_label_set = [batch_generator.no_roost_sets_V06,
                        batch_generator.roost_sets_V06]
    for ml_label, label_name in zip(
            ml_label_set,
            ['NoRoost', 'Roost']):
        result[label_name] = {}
        for ml_set in [utils.ML_Set.training, utils.ML_Set.validation,
                       utils.ML_Set.testing]:
            image_list = []
            for image in ml_label[ml_set]:
                label = batch_generator.label_dict[image]
                image_list.append(label.images[radar_product])
            key = str.lower(ml_set.fullname)
            result[label_name][key] = image_list
    return result


def get_bottleneck_list(image_lists, radar_field):
    for label in image_lists.keys():
        for ml_set in image_lists[label].keys():
            for i in range(len(image_lists[label][ml_set])):
                image_path = image_lists[label][ml_set][
                                 i] + '_' + 'inception_v3' + '.txt'
                image_path = image_path.replace('radarimages', 'bottleneck')
                temp = image_path.replace(radar_field.fullname, '{0}')
                image_lists[label][ml_set][i] = temp

    return image_lists


def get_bottleneck_path(image_lists, label_name, index, category):
    label_lists = image_lists[label_name]
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    return category_list[mod_index]


def get_bottleneck(image_lists, label_name, index, category, radar_field):
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          category)
    bottleneck_path = bottleneck_path.format(radar_field)

    with open(bottleneck_path, 'r') as bottleneck_file:
        bottleneck_string = bottleneck_file.read()
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    return bottleneck_values


def get_random_cached_bottlenecks(image_lists, how_many, category,
                                  radar_fields):
    class_count = len(image_lists.keys())
    bottlenecks = []
    ground_truths = []
    filenames = []
    if how_many >= 0:
        for unused_i in range(how_many):
            label_index = random.randrange(class_count)
            label_name = list(image_lists.keys())[label_index]
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            image_name = get_bottleneck_path(image_lists, label_name,
                                             image_index, category)
            bottleneck = []
            for radar_field in radar_fields:
                bottleneck.append(
                    get_bottleneck(image_lists, label_name, image_index,
                                   category, radar_field))
            bottleneck = np.array(bottleneck).reshape(-1)
            ground_truth = np.zeros(class_count, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            filenames.append(image_name)
    else:
        # Retrieve all bottlenecks.
        for label_index, label_name in enumerate(image_lists.keys()):
            for image_index, image_name in enumerate(
                    image_lists[label_name][category]):
                image_name = image_lists, label_name, image_index, category
                bottleneck = []
                for radar_field in radar_fields:
                    bottleneck.append(
                        get_bottleneck(image_lists, label_name, image_index,
                                       category, radar_field))
                bottleneck = np.array(bottleneck).reshape(-1)
                ground_truth = np.zeros(class_count, dtype=np.float32)
                ground_truth[label_index] = 1.0
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                filenames.append(image_name)
    return np.array(bottlenecks), np.array(ground_truths), np.array(filenames)


def train(model, bottleneck_list, num_iterations, save_file, radar_fields,
          callback_dir='/tmp/model_log'):
    # Setup callbacks
    callback = TensorBoard(callback_dir)
    callback.set_model(model)
    train_names = ['train_loss', 'train_accuracy']
    val_names = ['val_loss', 'val_accuracy']

    progress_string = '{} Epoch: {} Loss: {} Accuracy {}'

    for batch_no in range(num_iterations):
        try:
            x, y, _ = get_random_cached_bottlenecks(image_lists=bottleneck_list,
                                                    how_many=64,
                                                    category='training',
                                                    radar_fields=radar_fields)
            train_logs = model.train_on_batch(x, y)
            print progress_string.format(utils.ML_Set.training.fullname,
                                         batch_no,
                                         train_logs[0], train_logs[1])
            ml_utils.write_log(callback, train_names, train_logs, batch_no)

        except Exception as e:
            print e.message
        if (batch_no % 1 == 0):
            model.save_weights(save_file)
            try:
                x_, y_, _ = get_random_cached_bottlenecks(
                    image_lists=bottleneck_list, how_many=64,
                    category='validation',
                    radar_fields=radar_fields)

                val_logs = model.test_on_batch(x_, y_)
                ml_utils.write_log(callback, val_names, val_logs, batch_no)
                print progress_string.format(utils.ML_Set.validation.fullname,
                                             batch_no,
                                             val_logs[0], val_logs[1])
            except Exception as e:
                print e.message
    model.save_weights(save_file)


def create_model(input_dimention, save=None):
    # input_dim = 4 for dual-pol, 2 for legagy
    model = Sequential()
    model.add(Dense(256, input_dim=input_dimention, activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(2, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.sgd(lr=.0001),
                  metrics=['accuracy'])

    if save is not None:
        model.load_weights(save)
    return model


def main():
    os.chdir(settings.WORKING_DIRECTORY)
    dual_pol = True

    if dual_pol:
        radar_field = utils.Radar_Products.cc
        radar_fields = dual_pol_fields
        save = 'dual_pol.h5'
        model = create_model(8192, save)
        callback_dir = 'model_log/dual_pol/'
    else:
        radar_field = utils.Radar_Products.reflectivity
        radar_fields = legacy_fields
        save = 'legacy.h5'
        model = create_model(4096, save)
        callback_dir = 'model_log/legacy/'


    image_lists = create_image_lists(radar_field)
    bottleneck_list = get_bottleneck_list(image_lists, radar_field)
    train(model, bottleneck_list, 1500, save, radar_fields, callback_dir)


if __name__ == '__main__':
    main()
