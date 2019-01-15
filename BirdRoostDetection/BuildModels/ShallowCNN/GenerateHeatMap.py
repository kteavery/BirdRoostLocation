import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import BirdRoostDetection.BuildModels.ShallowCNN.model as ml_model
import BirdRoostDetection.LoadSettings as settings
from BirdRoostDetection import utils
from BirdRoostDetection.BuildModels import ml_utils
from BirdRoostDetection.ReadData import BatchGenerator

fill_color = 255
nan = float('nan')


def prediction_heat_map(model, img, width, height, stride):
    heat_map = []
    img_width = img.shape[1]
    img_height = img.shape[2]
    height_range = ((img_height / height) * (height / stride)) - (
        height / stride) + 1
    width_range = ((img_width / width) * (width / stride)) - (
        width / stride) + 1
    for i in range(width_range):
        for j in range(height_range):
            img_heat_map = prediction_heat_map_helper(i, j, width, height,
                                                      stride, img,
                                                      model, False)
            heat_map.append(img_heat_map)
    heat_map = np.array(heat_map)
    return heat_map


def prediction_heat_map_helper(i, j, width, height, stride, img, model,
                               show=False):
    width_start = i * stride
    height_start = j * stride
    x = np.copy(img)
    x[:, width_start:width_start + width,
    height_start:height_start + height, :].fill(fill_color)
    prediction = model.predict(x=x)[0][0]
    # print prediction
    if (show):
        plt.imshow(x[0])
        plt.show()
    heat_map = np.full((x.shape[1:3]), nan)
    heat_map[width_start:width_start + width,
    height_start:height_start + height].fill(prediction)
    return heat_map


def create_heatmaps(log_path, radar_product, epoch=''):
    batch_generator = BatchGenerator.Batch_Generator(
        ml_label_csv=settings.LABEL_CSV,
        ml_split_csv=settings.ML_SPLITS_DATA,
        default_batch_size=64)

    save_file = ml_utils.KERAS_SAVE_FILE.format(radar_product.fullname, '{}')

    titles = ['No Roost', 'Roost']
    checkpoint_path = log_path + ml_utils.CHECKPOINT_DIR
    model = ml_model.build_model(inputDimensions=(240, 240, 3))
    print(os.path.join(checkpoint_path, save_file.format(epoch)))
    model.load_weights(os.path.join(checkpoint_path, save_file.format(epoch)))
    batch_generator.get_batch(utils.ML_Set.testing,
                              radar_product)

    x, y, filenames = batch_generator.get_batch(utils.ML_Set.testing,
                                                radar_product)
    for i in range(len(filenames)):
        img = x[i:i + 1]
        label = y[i:i + 1]
        filename = filenames[i]
        loss, acc = model.evaluate(img, label)
        print(loss, acc, label, filename)
        prediction = model.predict(img)[0][0]
        print(prediction)
        heat_maps = []

        heat_maps.append(
            prediction_heat_map(model, img, 48, 48, 16))
        heat_maps.append(
            prediction_heat_map(model, img, 60, 60, 30))
        heat_maps.append(
            prediction_heat_map(model, img, 80, 80, 20))
        for j in range(len(heat_maps)):
            img_heat_map = np.array(heat_maps[j])
            heat_maps[j] = np.nanmean(img_heat_map, axis=0)

        heat_maps = np.array(heat_maps)

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
        for dat, ax in zip(
                [img[0], heat_maps[0], heat_maps[1], heat_maps[2]],
                axes.flat):
            im = ax.imshow(dat, vmin=0, vmax=1, cmap='jet')

        fig.colorbar(im, ax=axes.ravel().tolist(), orientation='horizontal',
                     aspect=60)

        fig.suptitle(
            '{0}: {1}, {2}'.format(titles[label[0][0]], prediction, filename))
        save_file = filename + '.png'
        print(save_file)
        plt.savefig(save_file)
        # plt.show()


def main():
    assert (int(sys.argv[1]) >= 0 and int(sys.argv[1]) <= 3), \
        "Radar product command line argument must be one of the following " \
        "values: 0, 1, 2, 3"
    radar_product = utils.Radar_Products(int(sys.argv[1]))
    log_path = ml_utils.LOG_PATH.format(radar_product.fullname)
    if len(sys.argv) == 3:
        log_path = sys.argv[2]

    create_heatmaps(log_path=log_path,
                    radar_product=radar_product,
                    epoch=str(1700))


if __name__ == "__main__":
    os.chdir(settings.WORKING_DIRECTORY)
    main()
