import os.path
import BirdRoostLocation.LoadSettings as settings
from BirdRoostLocation.BuildModels.Inception import aggregate_train
import os
from BirdRoostLocation import utils
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from BirdRoostLocation.BuildModels.Inception import retrain
import math
import matplotlib.pyplot as plt
from BirdRoostLocation.ReadData import BatchGenerator

WIDTH = 60
HEIGHT = 60
STRIDE = 30

FILL_COLOR = 0.9999


def get_heat_map_images(
    sess, model, i, j, height, stride, images, bottleneck_tensor, resized_image_tensor
):
    width_start = i * stride
    height_start = j * stride

    images = np.copy(images)
    images = (images + 1) / 2
    fill_color = 0.999

    bottleneck = []
    for img in images:
        x = np.copy(img)
        x[
            :,
            width_start : width_start + height,
            height_start : height_start + HEIGHT,
            :,
        ].fill(FILL_COLOR)
        bottleneck_values = sess.run(bottleneck_tensor, {resized_image_tensor: x})
        bottleneck.append(bottleneck_values)
    bottleneck = np.array(bottleneck).reshape((8192))
    bottleneck = np.array([bottleneck])

    save = "dual_pol.h5"
    model = aggregate_train.create_model(8192, save)

    prediction = model.predict(bottleneck)[0][0]
    print(prediction)

    heat_map = np.full((x.shape[1:3]), float("nan"))
    heat_map[
        width_start : width_start + height, height_start : height_start + height
    ].fill(prediction)
    return heat_map


def prediction_heat_map_on_bottleneck(
    sess, model, images, height, stride, bottleneck_tensor, resized_image_tensor
):
    heat_map = []
    img_height = 299
    increment = int(math.ceil((img_height % height) / float(stride)))

    height_width_range = (
        ((img_height / height) * (height / stride)) - (height / stride) + 1 + increment
    )

    for i in range(height_width_range):
        for j in range(height_width_range):
            heat_map.append(
                get_heat_map_images(
                    sess,
                    model,
                    i,
                    j,
                    height,
                    stride,
                    images,
                    bottleneck_tensor,
                    resized_image_tensor,
                )
            )
    heat_map = np.array(heat_map)
    return heat_map


def create_heatmap_for_filename(image_path):
    print(os.path.basename(image_path) + ".png")

    # Gather information about the model architecture we'll be using.
    model_dir = "tf_files/models/"
    model_info = retrain.create_model_info("inception_v3")
    retrain.maybe_download_and_extract(model_info["data_url"], model_dir)
    graph, bottleneck_tensor, resized_image_tensor = retrain.create_model_graph(
        model_info, model_dir
    )

    save = "dual_pol.h5"
    model = aggregate_train.create_model(8192, save)
    with tf.Session(graph=graph) as sess:
        # Set up the image decoding sub-graph.
        jpeg_data_tensor, decoded_image_tensor = retrain.add_jpeg_decoding(
            model_info["input_width"],
            model_info["input_height"],
            model_info["input_depth"],
            model_info["input_mean"],
            model_info["input_std"],
        )

        image_paths = [
            image_path.format(radar_field)
            for radar_field in aggregate_train.dual_pol_fields
        ]

        images = []
        for image_path in image_paths:
            image_data = gfile.FastGFile(image_path, "rb").read()
            resized_input_values = sess.run(
                decoded_image_tensor, {jpeg_data_tensor: image_data}
            )
            images.append(resized_input_values)

        img = images[0]
        label = ""

        heat_maps = []

        heat_maps.append(
            prediction_heat_map_on_bottleneck(
                sess, model, images, 48, 16, bottleneck_tensor, resized_image_tensor
            )
        )
        heat_maps.append(
            prediction_heat_map_on_bottleneck(
                sess, model, images, 60, 30, bottleneck_tensor, resized_image_tensor
            )
        )
        heat_maps.append(
            prediction_heat_map_on_bottleneck(
                sess, model, images, 80, 20, bottleneck_tensor, resized_image_tensor
            )
        )
        for j in range(len(heat_maps)):
            img_heat_map = np.array(heat_maps[j])
            heat_maps[j] = np.nanmean(img_heat_map, axis=0)

        heat_maps = np.array(heat_maps)

        fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
        for dat, ax in zip(
            [img[0], heat_maps[0], heat_maps[1], heat_maps[2]], axes.flat
        ):
            im = ax.imshow(dat, vmin=0, vmax=1, cmap="jet")

        fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", aspect=60)

        save_file = os.path.basename(image_path) + ".png"
        print(save_file)
        plt.savefig(save_file)


def main():
    os.chdir(settings.WORKING_DIRECTORY)
    dual_pol = True

    batch_generator = BatchGenerator.Single_Product_Batch_Generator(
        ml_label_csv=settings.LABEL_CSV,
        ml_split_csv=settings.ML_SPLITS_DATA,
        default_batch_size=8,
        high_memory_mode=False,
    )

    radar_product = utils.Radar_Products(3)
    _, _, filenames = batch_generator.get_batch(
        utils.ML_Set.testing, dualPol=dual_pol, radar_product=radar_product
    )

    for filename in filenames:
        radar = filename[0:4]
        year = filename[4:8]
        month = filename[8:10]
        day = filename[10:12]
        print(radar, year, month, day)
        image_path = "radarimages/{0}_Color/{1}/{2}/{3}/{4}" "/{5}_{0}.png".format(
            "{0}", radar, year, month, day, filename
        )
        create_heatmap_for_filename(image_path)


if __name__ == "__main__":
    main()
