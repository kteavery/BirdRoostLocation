import os
import pandas
import math
import ntpath
from BirdRoostLocation.ReadData import Labels
import numpy as np
from BirdRoostLocation import utils
from BirdRoostLocation.PrepareData import NexradUtils
from BirdRoostLocation import LoadSettings as settings
from BirdRoostLocation.BuildModels.ShallowCNN import model as shallow_model
import tensorflow as tf
from keras.models import model_from_json
import keras


class Batch_Generator:
    """This class organized the machine learning labels and creates ML batches.

    Class Variables:
        self.root_dir: The directory where the radar images are stored
        self.ml_sets: A dictionary containing a list of files that are part of
            the given ml set
        self.batch_size: the size of the minibatch learning batches
        self.label_dict: A dictionary of the labels, the key is the filename,
        and the value is a ML_Label object.
    """

    def __init__(
        self,
        ml_label_csv,
        ml_split_csv,
        validate_k_index=3,
        test_k_index=4,
        default_batch_size=8,
        root_dir=utils.RADAR_IMAGE_DIR,
    ):
        self.label_dict = {}
        self.root_dir = root_dir
        self.no_roost_sets = {}
        self.roost_sets = {}
        self.no_roost_sets_V06 = {}
        self.roost_sets_V06 = {}
        self.batch_size = default_batch_size
        # print("__init__")
        # print(ml_split_csv)
        # print(validate_k_index)
        # print(test_k_index)
        self.__set_ml_sets(ml_split_csv, validate_k_index, test_k_index)

    def __set_ml_sets(self, ml_split_csv, validate_k_index, test_k_index):
        """Create Train, test, and Validation set from k data folds.

        The k data folds are saved out to ml_split_csv. The fold at the given
        test and train indices as set to their corresponding set. The rest
        of the data is put into train. This method will initialize the following
        class variables: self.train, self.validation, and self.test. Each of
        these contains a list of filenames that correspond with the set.

        Args:
            ml_split_csv: A path to a csv file, where the csv has three columns,
            'AWS_file', 'Roost', and 'split_index'.
            validate_k_index: The index of the validation set.
            test_k_index: The index of the test set.
        """

        # print(ml_split_csv)
        ml_split_pd = pandas.read_csv(ml_split_csv)
        # print("ml_split_pd.head()")
        # print(ml_split_pd.head())

        # Remove files that weren't found
        all_files = utils.getListOfFilesInDirectory(self.root_dir + "data", ".png")
        print("ROOT DIR")
        print(self.root_dir + "data")
        # print(all_files)

        all_files_dict = {}
        for i in range(len(all_files)):
            all_files_dict[os.path.basename(all_files[i])[2:25]] = True
            # print(all_files_dict)
            # print(os.path.basename(all_files[i])[0:23])

        for index, row in ml_split_pd.iterrows():
            if all_files_dict.get(row["AWS_file"]) is None:
                ml_split_pd.drop(index, inplace=True)
        # print(ml_split_pd.head())
        # Sort into train, test, and validation sets
        print("LENGTHS OF NO ROOST/ROOST:")
        print(len(ml_split_pd[ml_split_pd.Roost != True]))
        print(len(ml_split_pd[ml_split_pd.Roost]))

        self.__set_ml_sets_helper(
            self.no_roost_sets,
            self.no_roost_sets_V06,
            ml_split_pd[ml_split_pd.Roost != True],
            validate_k_index,
            test_k_index,
        )
        self.__set_ml_sets_helper(
            self.roost_sets,
            self.roost_sets_V06,
            ml_split_pd[ml_split_pd.Roost],
            validate_k_index,
            test_k_index,
        )

    def __set_ml_sets_helper(self, ml_sets, ml_sets_V06, ml_split_pd, val_k, test_k):
        no_val_pd = ml_split_pd[ml_split_pd.split_index != val_k]
        ml_sets[utils.ML_Set.training] = list(
            no_val_pd[no_val_pd.split_index != test_k]["AWS_file"]
        )
        ml_sets[utils.ML_Set.validation] = list(
            ml_split_pd[ml_split_pd.split_index == val_k]["AWS_file"]
        )
        ml_sets[utils.ML_Set.testing] = list(
            ml_split_pd[ml_split_pd.split_index == test_k]["AWS_file"]
        )

        for key in list(ml_sets.keys()):
            ml_sets_V06[key] = []
            for item in ml_sets[key]:
                if int(item[-1]) >= 6:
                    ml_sets_V06[key].append(item)

            np.random.shuffle(ml_sets[key])
            np.random.shuffle(ml_sets_V06[key])

    def get_batch_indices(self, ml_sets, ml_set, num_temporal_data=0):
        # print(ml_sets)
        # print(len(ml_sets[ml_set]))
        # print(ml_set)
        # print(self.batch_size / 2)
        indices = np.random.randint(
            low=0, high=len(ml_sets[ml_set]), size=int(self.batch_size / 2)
        )
        # print(indices)
        return indices

    def get_batch(self, ml_set, dualPol, radar_product=None):
        ground_truths = []
        train_data = []
        filenames = []
        roost_sets = self.roost_sets
        no_roost_sets = self.no_roost_sets
        if dualPol:
            roost_sets = self.roost_sets_V06
            no_roost_sets = self.no_roost_sets_V06

        return train_data, ground_truths, filenames, roost_sets, no_roost_sets

    def single_product_batch_param_helper(
        self,
        filename,
        filenames,
        radar_product,
        problem,
        model_type,
        train_data,
        ground_truths,
    ):
        is_roost = int(self.label_dict[filename].is_roost)
        polar_radius = float(self.label_dict[filename].polar_radius)
        polar_theta = float(self.label_dict[filename].polar_theta)
        roost_size = float(self.label_dict[filename].radius)
        images = self.label_dict[filename].get_image(radar_product)
        # print(self.label_dict[filename].images[radar_product])
        # print(len(indices))
        # print(self.label_dict[filename])
        # print(type(self.label_dict[filename]))
        # print(filename)
        # print(radar_product)
        # print(self.label_dict[filename].get_image(radar_product))

        # print("np.array(images).shape")
        # print(np.array(images).shape)
        if images != []:
            print(filename)
            print(float(self.label_dict[filename].polar_radius))
            print(float(self.label_dict[filename].polar_theta))
            # filenames.append(filename)

            # if np.array(train_data).size == 0:
            #    train_data = images
            #    train_data = np.array(train_data)
            # else:
            #    train_data = np.concatenate((train_data, np.array(images)), axis=0)

            # print("train_data.shape - problem?")
            # print(train_data.shape)

            if problem == "detection":
                if np.array(train_data).size == 0:
                    train_data = images
                    train_data = np.array(train_data)
                else:
                    train_data = np.concatenate((train_data, np.array(images)), axis=0)

                if np.array(ground_truths).size == 0:
                    ground_truths = [[is_roost, 1 - is_roost]] * np.array(images).shape[
                        0
                    ]
                else:
                    ground_truths = np.concatenate(
                        (
                            ground_truths,
                            [[is_roost, 1 - is_roost]] * np.array(images).shape[0],
                        ),
                        axis=0,
                    )
            else:  # localization
                radii = np.array([polar_radius] * np.array(images).shape[0])

                if not np.isnan(np.sum(radii)):
                    if np.array(train_data).size == 0:
                        train_data = images
                        train_data = np.array(train_data)
                    else:
                        train_data = np.concatenate(
                            (train_data, np.array(images)), axis=0
                        )

                    # print("train_data.shape - problem?")
                    # print(train_data.shape)

                    radii = np.array([polar_radius] * np.array(images).shape[0])
                    thetas = []
                    # print("radii.shape")
                    # print(radii.shape)

                    # print("NOT NAN")
                    # print(radii)
                    for i in range(len(images)):
                        thetas.append(
                            adjustTheta(
                                self,
                                polar_theta,
                                self.label_dict[filename].images[radar_product][i],
                            )
                        )

                    if model_type == "shallow_cnn":
                        # print("SHALLOW_CNN")
                        pairs = list(
                            zip(normalize(radii, 2, 0), normalize(thetas, 360, 0))
                        )
                        pairs = [list(x) for x in pairs]

                        if np.array(ground_truths).size == 0:
                            ground_truths = pairs
                        else:
                            ground_truths = np.concatenate(
                                (ground_truths, pairs), axis=0
                            )
                    else:  # unet
                        # print("UNET")
                        # print("Roost Size: ")

                        masks = np.zeros((len(radii), 240, 240))
                        if type(roost_size) != float or math.isnan(roost_size):
                            roost_size = 28.0
                            # print(roost_size)
                        else:
                            roost_size = roost_size / 1000  # convert to km
                            # print(roost_size)

                        mask_roost_size = (roost_size / 300) * (240 / 2)

                        mask_radii = [(radius / 300) * (240 / 2) for radius in radii]
                        # print(radii)
                        # print(mask_radii)
                        # print(thetas)

                        vconvert_to_cart = np.vectorize(convert_to_cart)
                        cart_x, cart_y = vconvert_to_cart(mask_radii, thetas)

                        for k, mask in enumerate(masks):
                            # print(filename)
                            # print("RADII")
                            # print(mask_radii)
                            # print("THETAS")
                            # print(thetas)
                            # print("CART_X")
                            # print(cart_x)
                            # print("CART_Y")
                            # print(cart_y)
                            # print("K")
                            # print(k)
                            mask[
                                120 - int(round(list(cart_y)[k])),
                                120 + int(round(list(cart_x)[k])),
                            ] = 1.0
                            # print("mask")

                            color_pts = points_in_circle_np(
                                mask_roost_size,
                                y0=120 - int(round(list(cart_y)[k])),
                                x0=120 + int(round(list(cart_x)[k])),
                            )
                            # print("color points")
                            for pt in color_pts:
                                mask[pt[0], pt[1]] = 1.0
                            # print("append to ground truth")
                            # ground_truths = np.concatenate(
                            #     (ground_truths, mask), axis=0
                            # )

                            if np.array(ground_truths).size == 0:
                                ground_truths = mask
                            else:
                                ground_truths = np.concatenate(
                                    (ground_truths, mask), axis=0
                                )
                            # print("ground_truths")
                            # print(ground_truths.shape)
                            # print("train_shape")
                            # print(train_data.shape)
        # print("train_data.shape")
        train_data = np.array(train_data)
        # print(train_data.shape)
        return train_data, ground_truths

    def single_product_batch_params(
        self,
        ground_truths,
        train_data,
        filenames,
        roost_sets,
        no_roost_sets,
        ml_set,
        radar_product,
        model_type,
        problem,
    ):
        # print(filenames)
        if filenames == []:
            for ml_sets in [roost_sets, no_roost_sets]:
                if ml_sets[ml_set]:  # in case you only train on true or false labels
                    indices = Batch_Generator.get_batch_indices(self, ml_sets, ml_set)
                    for i, index in enumerate(indices):
                        filename = ml_sets[ml_set][index]
                        # print(len(indices))
                        # print(i)
                        train_data, ground_truths = Batch_Generator.single_product_batch_param_helper(
                            self,
                            filename,
                            filenames,
                            radar_product,
                            problem,
                            model_type,
                            train_data,
                            ground_truths,
                        )
                        # print(np.array(train_data).shape)
                        # print(np.array(ground_truths).shape)
                        filenames.append(filename)
                    # print(filenames)
        else:
            for filename in filenames:
                train_data, ground_truths = Batch_Generator.single_product_batch_param_helper(
                    self,
                    filename,
                    filenames,
                    radar_product,
                    problem,
                    model_type,
                    train_data,
                    ground_truths,
                )

        truth_shape = np.array(ground_truths).shape
        # print("truth shape: ")
        # print(truth_shape)

        try:
            if problem == "detection":
                ground_truths = np.array(ground_truths).reshape(
                    truth_shape[0], truth_shape[1]
                )

            train_data_np = np.array(train_data)
            shape = train_data_np.shape
            # print("shape")
            # print(shape)
            train_data_np = train_data_np.reshape(
                shape[0], shape[1], shape[2], shape[3]
            )

            return train_data_np, np.array(ground_truths), np.array(filenames)
        except IndexError as e:
            print(e)
            return None, None, None


class Single_Product_Batch_Generator(Batch_Generator):
    def __init__(
        self,
        ml_label_csv,
        ml_split_csv,
        validate_k_index=3,
        test_k_index=4,
        default_batch_size=settings.DEFAULT_BATCH_SIZE,
        root_dir=utils.RADAR_IMAGE_DIR,
        high_memory_mode=False,
    ):
        Batch_Generator.__init__(
            self,
            ml_label_csv,
            ml_split_csv,
            validate_k_index,
            test_k_index,
            default_batch_size,
            root_dir,
        )
        ml_label_pd = pandas.read_csv(ml_label_csv)
        # print("ml_label_pd")
        # print(ml_label_pd.head())
        for _, row in ml_label_pd.iterrows():
            self.label_dict[row["AWS_file"]] = Labels.ML_Label(
                row["AWS_file"], row, self.root_dir, high_memory_mode
            )
            # print(self.label_dict[row["AWS_file"]])

    def get_batch(
        self,
        ml_set,
        dualPol,
        radar_product=None,
        num_temporal_data=0,
        model_type="shallow_cnn",
        problem="detection",
    ):
        """Get a batch of data for machine learning. As a default, a batch
        contains data from a single radar product.

        Args:
            ml_set: ML_Set enum value, train, test, or validation.
            radar_product: Radar_Product enum value, reflectivity, velocity,
                zdr, or rho_hv.

        Returns:
            train_data, ground_truth, filenames:
                The ground truth is an array of batch size, where each item
                in the array contains a single ground truth label.
                The train_data is an array of images, corresponding to the
                ground truth values.
                filenames is an array of filenames, corresponding to the
                ground truth values.
        """
        ground_truths, train_data, filenames, roost_sets, no_roost_sets = Batch_Generator.get_batch(
            self, ml_set, dualPol, radar_product
        )

        return Batch_Generator.single_product_batch_params(
            self,
            ground_truths,
            train_data,
            filenames,
            roost_sets,
            no_roost_sets,
            ml_set,
            radar_product,
            model_type,
            problem,
        )


class Multiple_Product_Batch_Generator(Batch_Generator):
    def __init__(
        self,
        ml_label_csv,
        ml_split_csv,
        validate_k_index=3,
        test_k_index=4,
        default_batch_size=8,
        root_dir=utils.RADAR_IMAGE_DIR,
        high_memory_mode=False,
    ):
        Batch_Generator.__init__(
            self,
            ml_label_csv,
            ml_split_csv,
            validate_k_index,
            test_k_index,
            default_batch_size,
            root_dir,
        )
        ml_label_pd = pandas.read_csv(ml_label_csv)
        for _, row in ml_label_pd.iterrows():
            self.label_dict[row["AWS_file"]] = Labels.ML_Label(
                row["AWS_file"], row, self.root_dir, high_memory_mode
            )

    # channels will be RGB values, first dimension will be radar products
    def get_batch(
        self,
        ml_set,
        dualPol,
        batch_size=8,
        radar_product=None,
        loaded_models=None,
        num_temporal_data=0,
        model_type="shallow_cnn",
        problem="detection",
    ):
        """Get a batch of data for machine learning. This batch contains data
        with four channels in it, one for each radar product. For dualPol data
        this will be four radar products, and for legacy data this will be two
        radar products.

        Args:
            ml_set: ML_Set enum value, train, test, or validation.
            dualPol: Boolean, true if the data is dual pol, false if the radar
            data is legacy.

        Returns:
            train_data, ground_truth, filenames:
                The ground truth is an array of batch size, where each item
                in the array contains a single ground truth label.
                The train_data is an array of images, corresponding to the
                ground truth values.
                filenames is an array of filenames, corresponding to the
                ground truth values.
        """
        ground_truths, train_data, filenames, roost_sets, no_roost_sets = Batch_Generator.get_batch(
            self, ml_set, dualPol, radar_product=None
        )
        train_list = []
        truth_list = []
        pred_list = []
        file_list = []

        if dualPol:
            radar_products = [
                utils.Radar_Products.cc,
                utils.Radar_Products.diff_reflectivity,
                utils.Radar_Products.reflectivity,
                utils.Radar_Products.velocity,
            ]

        else:
            radar_products = utils.Legacy_radar_products

        for k, radar_product in enumerate(radar_products):
            print(radar_product)
            train, truth, filenames = Batch_Generator.single_product_batch_params(
                self,
                ground_truths,
                train_data,
                filenames,
                roost_sets,
                no_roost_sets,
                ml_set,
                radar_product,
                model_type,
                problem,
            )

            # if str(radar_product) == "Radar_Products.cc":
            #     product_str = "Rho_HV"
            # elif str(radar_product) == "Radar_Products.diff_reflectivity":
            #     product_str = "Zdr"
            # elif str(radar_product) == "Radar_Products.reflectivity":
            #     product_str = "Reflectivity"
            # else:
            #     product_str = "Velocity"

            # json_file = open(
            #     settings.WORKING_DIRECTORY
            #     + "model/"
            #     + str(product_str)
            #     + "/checkpoint/"
            #     + str(product_str)
            #     + ".json",
            #     "r",
            # )
            # loaded_model_json = json_file.read()
            # json_file.close()
            # model = model_from_json(loaded_model_json)

            # print(
            #     settings.WORKING_DIRECTORY
            #     + "model/"
            #     + str(product_str)
            #     + "/checkpoint/"
            #     + str(product_str)
            #     + ".h5"
            # )
            # model.load_weights(
            #     settings.WORKING_DIRECTORY
            #     + "model/"
            #     + str(product_str)
            #     + "/checkpoint/"
            #     + str(product_str)
            #     + ".h5"
            # )

            predictions = []
            print("len(train)")
            print(len(train))
            for i in range(0, len(train), batch_size):
                train_batch = []
                for j in range(0, batch_size):
                    train_batch.append(train[i])

                train_batch = np.array(train_batch)

                print("train_batch.shape")
                print(train_batch.shape)
                pred = loaded_models[k].predict_proba(train_batch)
                predictions.append(np.array([pred, 1 - pred]))

            train_list.append(train)
            truth_list.append(truth)
            pred_list.append(predictions)
            file_list.append(filenames)

            # print("filenames, train_list, truth_list, pred_list, file_list")
            # print(filenames)
            # print(np.array(train_list).shape)
            # print(np.array(truth_list).shape)
            # print(np.array(pred_list).shape)
            # print(np.array(file_list).shape)

        return (
            np.array(train_list),
            np.array(truth_list),
            np.array(pred_list),
            np.array(file_list),
        )


def normalize(self, x, maxi, mini):
    if type(x) is list:
        return [(y - mini) / (maxi - mini) for y in x]
    else:
        return (x - mini) / (maxi - mini)


def adjustTheta(self, theta, path):
    filename = os.path.splitext(ntpath.basename(path))[0]
    parts = filename.split("_")
    if "flip" in parts:
        if theta > 180.0:
            theta = 540 - theta
        else:
            theta = 180 - theta

    # rotation
    try:
        if "noise" in parts:
            degree_offset = int(parts[-2])
        else:
            degree_offset = int(parts[-1])
        theta += degree_offset
    except ValueError:
        return theta

    return theta


def convert_to_cart(radius, theta):
    return radius * math.cos(theta), radius * math.sin(theta)


def points_in_circle_np(radius, y0=0, x0=0):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    y, x = np.where((y_[:, np.newaxis] - y0) ** 2 + (x_ - x0) ** 2 <= radius ** 2)
    for y, x in zip(y_[y], x_[x]):
        yield y, x
