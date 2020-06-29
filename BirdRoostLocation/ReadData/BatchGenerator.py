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
        default_batch_size=settings.DEFAULT_BATCH_SIZE,
        root_dir=utils.RADAR_IMAGE_DIR,
    ):
        self.label_dict = {}
        self.root_dir = root_dir
        self.no_roost_sets = {}
        self.roost_sets = {}
        self.no_roost_sets_V06 = {}
        self.roost_sets_V06 = {}
        self.batch_size = default_batch_size
        print("ML LABEL CSV")
        print(ml_label_csv)
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

        print("ML SPLIT CSV")
        print(ml_split_csv)
        ml_split_pd = pandas.read_csv(ml_split_csv)

        # Remove files that weren't found
        all_files = utils.getListOfFilesInDirectory(self.root_dir + "data", ".png")
        print("ROOT DIR")
        print(self.root_dir + "data")

        all_files_dict = {}
        for i in range(len(all_files)):
            all_files_dict[os.path.basename(all_files[i])[2:25]] = True

        for index, row in ml_split_pd.iterrows():
            if all_files_dict.get(row["AWS_file"]) is None:
                ml_split_pd.drop(index, inplace=True)

        print("LENGTHS OF NO ROOST/ROOST:")
        print(len(ml_split_pd[ml_split_pd.Roost != True]))
        print(len(ml_split_pd[ml_split_pd.Roost]))

        print("BEFORE self.__set_ml_sets_helper - NO ROOST")
        self.__set_ml_sets_helper(
            self.no_roost_sets,
            self.no_roost_sets_V06,
            ml_split_pd[ml_split_pd.Roost != True],
            validate_k_index,
            test_k_index,
        )
        print("AFTER self.__set_ml_sets_helper - NO ROOST")
        self.__set_ml_sets_helper(
            self.roost_sets,
            self.roost_sets_V06,
            ml_split_pd[ml_split_pd.Roost],
            validate_k_index,
            test_k_index,
        )
        print("AFTER self.__set_ml_sets_helper - ROOST")

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
        print("ml_sets[utils.ML_Set....]")

        for key in list(ml_sets.keys()):
            ml_sets_V06[key] = []
            for item in ml_sets[key]:
                if int(item[-1]) >= 6:
                    ml_sets_V06[key].append(item)

            np.random.shuffle(ml_sets[key])
            np.random.shuffle(ml_sets_V06[key])

    def get_batch_indices(self, ml_sets, ml_set, num_temporal_data=0):
        indices = np.random.randint(
            low=0, high=len(ml_sets[ml_set]), size=int(self.batch_size / 2)
        )
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
        images,
    ):
        is_roost = int(self.label_dict[filename][0].is_roost)
        polar_radius = [
            float(self.label_dict[filename][i].polar_radius)
            for i in range(len(self.label_dict[filename]))
        ]
        polar_theta = [
            float(self.label_dict[filename][i].polar_theta)
            for i in range(len(self.label_dict[filename]))
        ]
        roost_size = [
            float(self.label_dict[filename][i].radius)
            for i in range(len(self.label_dict[filename]))
        ]

        # print("np.array(images).shape")
        # print(np.array(images).shape)

        if images != []:
            #     print(filename)
            #     print(len(self.label_dict[filename]))

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
                print("ground truths shape")
                print(np.array(ground_truths).shape)

            else:  # localization
                all_radii = np.array([])
                all_thetas = np.array([])

                for k in range(len(polar_radius)):
                    radii = np.array([polar_radius[k]] * np.array(images).shape[0])

                    if not np.isnan(np.sum(radii)):
                        mask_radii = [(radius / 300) * (240 / 2) for radius in radii]
                        thetas = []

                        for i in range(len(images)):
                            thetas.append(
                                adjustTheta(
                                    self,
                                    polar_theta[k],
                                    self.label_dict[filename][0].images[radar_product][
                                        i
                                    ],
                                )
                            )

                        all_radii = np.append(all_radii, np.array(mask_radii))
                        all_thetas = np.append(all_thetas, np.array(thetas))

                # print(len(self.label_dict[filename]))
                # print(str(len(all_radii) / len(self.label_dict[filename])))
                all_radii = np.reshape(
                    all_radii,
                    (
                        len(self.label_dict[filename]),
                        int(len(all_radii) / len(self.label_dict[filename])),
                    ),
                )
                all_thetas = np.reshape(
                    all_thetas,
                    (
                        len(self.label_dict[filename]),
                        int(len(all_thetas) / len(self.label_dict[filename])),
                    ),
                )
                # print("ALL RADII, ALL THETAS")
                # print(all_radii.shape)
                # print(all_thetas.shape)
                masks = np.zeros((len(all_radii[0]), 240, 240))
                if type(roost_size) != float or math.isnan(roost_size):
                    roost_size = 28.0
                else:
                    roost_size = roost_size / 1000  # convert to km

                mask_roost_size = (roost_size / 300) * (240 / 2)

                vconvert_to_cart = np.vectorize(convert_to_cart)
                try:
                    cart_x, cart_y = vconvert_to_cart(all_radii, all_thetas)
                except ValueError as e:
                    return train_data, ground_truths

                for k in range(cart_x.shape[0]):

                    for j in range(cart_x.shape[1]):
                        try:
                            masks[j][
                                120 - int(round(cart_y[k][j])),
                                120 + int(round(cart_x[k][j])),
                            ] = 1.0

                            color_pts = points_in_circle_np(
                                mask_roost_size,
                                y0=120 - int(round(cart_y[k][j])),
                                x0=120 + int(round(cart_x[k][j])),
                            )
                            for pt in color_pts:
                                masks[j][pt[0], pt[1]] = 1.0
                        except IndexError as e:
                            pass

                if np.array(train_data).size == 0:
                    train_data = images
                    train_data = np.array(train_data)
                else:
                    train_data = np.concatenate((train_data, np.array(images)), axis=0)

                if np.array(ground_truths).size == 0:
                    ground_truths = masks
                else:
                    ground_truths = np.concatenate((ground_truths, masks), axis=0)

        train_data = np.array(train_data)
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
        is_eval=False,
    ):
        extended_filenames = np.array([])
        if filenames == []:
            for ml_sets in [roost_sets, no_roost_sets]:
                if ml_sets[ml_set]:  # in case you only train on true or false labels
                    indices = Batch_Generator.get_batch_indices(self, ml_sets, ml_set)
                    for i, index in enumerate(indices):
                        filename = ml_sets[ml_set][index]
                        if filename not in extended_filenames:
                            # print(len(indices))
                            # print(i)
                            images = self.label_dict[filename][0].get_image(
                                radar_product
                            )
                            if images != []:
                                train_data, ground_truths = Batch_Generator.single_product_batch_param_helper(
                                    self,
                                    filename,
                                    filenames,
                                    radar_product,
                                    problem,
                                    model_type,
                                    train_data,
                                    ground_truths,
                                    images,
                                )

                            #### !!!!
                            if model_type == "shallow_cnn" and is_eval == False:
                                extended_filenames = np.append(
                                    extended_filenames, filename
                                )
                            elif model_type == "shallow_cnn" and is_eval == True:
                                extended_filenames = np.append(
                                    extended_filenames,
                                    [filename]
                                    * (len(train_data) - len(extended_filenames)),
                                )
                            else:  # unet
                                extended_filenames = np.append(
                                    extended_filenames,
                                    [filename]
                                    * (len(train_data) - len(extended_filenames)),
                                )
                            # print(len(images))
                            # print([filename])
        else:
            for filename in filenames:
                images = self.label_dict[filename][0].get_image(radar_product)
                if images != []:
                    train_data, ground_truths = Batch_Generator.single_product_batch_param_helper(
                        self,
                        filename,
                        filenames,
                        radar_product,
                        problem,
                        model_type,
                        train_data,
                        ground_truths,
                        images,
                    )

                ### !!!!
                if model_type == "shallow_cnn" and is_eval == False:
                    extended_filenames = np.append(extended_filenames, filename)
                elif model_type == "shallow_cnn" and is_eval == True:
                    extended_filenames = np.append(
                        extended_filenames,
                        [filename] * (len(train_data) - len(extended_filenames)),
                    )
                else:  # unet
                    extended_filenames = np.append(
                        extended_filenames,
                        [filename] * (len(train_data) - len(extended_filenames)),
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
            train_data_np = train_data_np.reshape(
                shape[0], shape[1], shape[2], shape[3]
            )

            print("RETURN SHAPES")
            print(train_data_np.shape)
            print(ground_truths.shape)
            print(extended_filenames.shape)
            return train_data_np, np.array(ground_truths), np.array(extended_filenames)
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
            # print(row["AWS_file"])

            if row["AWS_file"] not in self.label_dict:
                # print("new")
                self.label_dict[row["AWS_file"]] = []

            self.label_dict[row["AWS_file"]].append(
                Labels.ML_Label(row["AWS_file"], row, self.root_dir, high_memory_mode)
            )

            # print('self.label_dict[row["AWS_file"]]')
            # print(self.label_dict[row["AWS_file"]])

    def get_batch(
        self,
        ml_set,
        dualPol,
        radar_product=None,
        num_temporal_data=0,
        model_type="shallow_cnn",
        problem="detection",
        filenames=[],
        is_eval=False,
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
        if len(filenames) == 0:
            ground_truths, train_data, filenames, roost_sets, no_roost_sets = Batch_Generator.get_batch(
                self, ml_set, dualPol, radar_product
            )
        else:
            ground_truths, train_data, _, roost_sets, no_roost_sets = Batch_Generator.get_batch(
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
            is_eval,
        )


class Multiple_Product_Batch_Generator(Batch_Generator):
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
        print("after Batch_Generator.__init__")
        print(ml_label_csv)

        ml_label_pd = pandas.read_csv(ml_label_csv)
        print(ml_label_pd.shape)
        for _, row in ml_label_pd.iterrows():
            if row["AWS_file"] not in self.label_dict:
                # print("new")
                self.label_dict[row["AWS_file"]] = [
                    Labels.ML_Label(
                        row["AWS_file"], row, self.root_dir, high_memory_mode
                    )
                ]
            else:
                # print("duplicate")
                # print(row["AWS_file"])
                self.label_dict[row["AWS_file"]].append(
                    Labels.ML_Label(
                        row["AWS_file"], row, self.root_dir, high_memory_mode
                    )
                )

    # channels will be RGB values, first dimension will be radar products
    def get_batch(
        self,
        ml_set,
        dualPol,
        batch_size=settings.DEFAULT_BATCH_SIZE,
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

        radar_products = [
            utils.Radar_Products.cc,
            utils.Radar_Products.diff_reflectivity,
            utils.Radar_Products.reflectivity,
            utils.Radar_Products.velocity,
        ]

        for k, product in enumerate(radar_products):
            print(product)
            print("BEFORE")
            print(len(filenames))
            train, truth, filenames = Batch_Generator.single_product_batch_params(
                self,
                ground_truths,
                train_data,
                filenames,
                roost_sets,
                no_roost_sets,
                ml_set,
                product,
                model_type,
                problem,
            )
            print("AFTER")
            print(len(filenames))

            predictions = np.array([])
            print("batch size")
            print(batch_size)
            for i in range(0, len(train), batch_size):
                train_batch = []
                for j in range(0, batch_size):
                    if (i + j) < len(train):
                        train_batch.append(train[i + j])

                train_batch = np.array(train_batch)

                # print("train_batch.shape")
                # print(train_batch.shape)
                if len(train_batch) > 0:
                    pred = loaded_models[k].predict_proba(train_batch)
                    # if len(predictions) == 0:
                    #     predictions = np.array([pred, 1 - pred])
                    #     print(predictions.shape)
                    # else:
                    # print(np.array([pred, 1 - pred]).shape)
                    # print(np.array(pred).shape)
                    # pred_shape = pred.shape
                    # predictions = np.append(predictions, np.array([pred, 1 - pred]))
                    # predictions = np.reshape(predictions, (pred_shape[0], 2))
                    predictions = np.append(predictions, np.array(pred))
                    # print("predictions.shape")
                    # print(predictions.shape)

            train_list.append(np.array(train))
            truth_list.append(np.array(truth))
            file_list.append(np.array(filenames))

            # print(np.array(truth_list).shape)
            try:
                predictions = np.reshape(
                    predictions, (np.array(truth_list).shape[1], 2)
                )
            except Exception as e:
                print(e)
                return None, None, None, None
            # print(predictions)
            # print(type(predictions))
            # print(type(predictions[0]))
            # predictions = np.array(predictions)
            # print(predictions.shape)

            pred_list.append(predictions)

            # print(np.array(train_list).shape)
            # print(np.array(truth_list).shape)
            # print(np.array(pred_list).shape)
            # print(np.array(file_list).shape)

        print("train_list, truth_list, pred_list, file_list")
        print(np.array(train_list).shape)
        print(np.array(truth_list).shape)
        print(np.array(pred_list).shape)
        print(np.array(file_list).shape)

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
