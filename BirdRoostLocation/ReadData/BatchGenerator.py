import os
import pandas
import ntpath
from BirdRoostLocation.ReadData import Labels
import numpy as np
from BirdRoostLocation import utils
from BirdRoostLocation.PrepareData import NexradUtils
from BirdRoostLocation import LoadSettings as settings


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
        print("Ks: ")
        print(validate_k_index)
        print(test_k_index)

        print(ml_split_csv)

        ml_split_pd = pandas.read_csv(ml_split_csv)

        # Remove files that weren't found
        all_files = utils.getListOfFilesInDirectory(self.root_dir, ".png")

        all_files_dict = {}
        for i in range(len(all_files)):
            all_files_dict[os.path.basename(all_files[i])[2:25]] = True

        for index, row in ml_split_pd.iterrows():
            if all_files_dict.get(row["AWS_file"]) is None:
                ml_split_pd.drop(index, inplace=True)

        # Sort into train, test, and validation sets
        # print("LENGTHS OF NO ROOST/ROOST:")
        # print(len(ml_split_pd[ml_split_pd.Roost != True]))
        # print(len(ml_split_pd[ml_split_pd.Roost]))

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
        print(test_k)
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


class Small_Image_Batch_Generator(Batch_Generator):
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

    def get_batch(self, ml_set, dualPol, radar_product=None, num_temporal_data=0):
        """Get a batch of data for machine learning. As a default a batch
        contains data from for a single radar product.

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
        if ml_set is utils.ML_Set.testing:

            ground_truths, train_data, filenames, roost_sets, _ = Batch_Generator.get_batch(
                self, ml_set, dualPol, radar_product
            )
            # for ml_sets in [roost_sets, no_roost_sets]:

            # indices = range(len(roost_sets[ml_set]))
            indices = np.random.randint(low=0, high=len(roost_sets[ml_set]), size=750)
            for index in indices:
                filename = roost_sets[ml_set][index]
                label = self.label_dict[filename]
                image = self.label_dict[filename].get_image(radar_product)
                radar_loc = NexradUtils.getRadarLocation(filename[0:4])
                y = (radar_loc[0] - label.latitude) * 89 + 120
                x = (radar_loc[1] - label.longitude) * 72.8 + 120
                for i in range(5):
                    for j in range(5):
                        is_small_roost = 0
                        x_start = i * 40
                        x_end = i * 40 + 80
                        y_start = j * 40
                        y_end = j * 40 + 80

                        if x >= x_start and x <= x_end and y >= y_start and y <= y_end:
                            is_small_roost += 1

                        small_image = image[x_start:x_end, y_start:y_end]
                        ground_truths.append([is_small_roost, 1 - is_small_roost])
                        filenames.append(filename)
                        train_data.append(small_image)
            train_data_np = np.array(train_data)
            shape = train_data_np.shape
            train_data_np = train_data_np.reshape(shape[0], shape[1], shape[2], 1)
            return train_data_np, np.array(ground_truths), np.array(filenames)
        else:
            ground_truths, train_data, filenames, roost_sets, _ = Batch_Generator.get_batch(
                self, ml_set, dualPol, radar_product
            )

            indices = Batch_Generator.get_batch_indices(self, roost_sets, ml_set)
            for index in indices:
                filename = roost_sets[ml_set][index]
                label = self.label_dict[filename]
                image = self.label_dict[filename].get_image(radar_product)
                radar_loc = NexradUtils.getRadarLocation(filename[0:4])
                y = (radar_loc[0] - label.latitude) * 89 + 120
                x = (radar_loc[1] - label.longitude) * 72.8 + 120
                for i in range(5):
                    for j in range(5):
                        is_small_roost = 0
                        x_start = i * 40
                        x_end = i * 40 + 80
                        y_start = j * 40
                        y_end = j * 40 + 80

                        if x >= x_start and x <= x_end and y >= y_start and y <= y_end:
                            is_small_roost += 1

                        small_image = image[x_start:x_end, y_start:y_end]
                        ground_truths.append([is_small_roost, 1 - is_small_roost])
                        filenames.append(filename)
                        train_data.append(small_image)
            train_data_np = np.array(train_data)
            shape = train_data_np.shape
            train_data_np = train_data_np.reshape(shape[0], shape[1], shape[2], 1)
            return train_data_np, np.array(ground_truths), np.array(filenames)


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
        for _, row in ml_label_pd.iterrows():
            self.label_dict[row["AWS_file"]] = Labels.ML_Label(
                row["AWS_file"], row, self.root_dir, high_memory_mode
            )
            # print(self.label_dict[row["AWS_file"]])

    def normalize(self, x, maxi, mini):
        if list(x):
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
            degree_offset = int(parts[-1])
            theta += degree_offset
        except ValueError:
            return theta

        return theta

    def get_batch(
        self,
        ml_set,
        dualPol,
        radar_product=None,
        num_temporal_data=0,
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

        print("Get batch: ")
        print(str(len(roost_sets)))
        print(str(len(no_roost_sets)))
        for key in roost_sets:
            print(len(roost_sets[key]))
        for key in no_roost_sets:
            print(len(no_roost_sets[key]))

        for ml_sets in [roost_sets, no_roost_sets]:
            if ml_sets[ml_set]:  # in case you only train on true or false labels
                indices = Batch_Generator.get_batch_indices(self, ml_sets, ml_set)

                for index in indices:
                    filename = ml_sets[ml_set][index]
                    is_roost = int(self.label_dict[filename].is_roost)
                    polar_radius = float(self.label_dict[filename].polar_radius)
                    polar_theta = float(self.label_dict[filename].polar_theta)
                    images = self.label_dict[filename].get_image(radar_product)
                    # print(images)
                    # print(self.label_dict[filename].images[radar_product])

                    if images != []:
                        filenames.append(filename)
                        if np.array(train_data).size == 0:
                            train_data = images
                            train_data = np.array(train_data)
                        else:
                            train_data = np.concatenate(
                                (train_data, np.array(images)), axis=0
                            )

                        if problem == "detection":
                            if np.array(ground_truths).size == 0:
                                ground_truths = [[is_roost, 1 - is_roost]] * np.array(
                                    images
                                ).shape[0]
                            else:
                                ground_truths = np.concatenate(
                                    (
                                        ground_truths,
                                        [[is_roost, 1 - is_roost]]
                                        * np.array(images).shape[0],
                                    ),
                                    axis=0,
                                )
                        else:  # localization
                            radii = [polar_radius for i in range(len(images))]
                            thetas = []
                            for i in range(len(images)):
                                thetas.append(self.adjustTheta(
                                    polar_theta,
                                    self.label_dict[filename].images[
                                        radar_product
                                    ][i],
                                ))

                            if np.array(ground_truths).size == 0:
                                ground_truths = [
                                    [
                                        self.normalize(radii, 2, 0),
                                        self.normalize(
                                            thetas,
                                            360,
                                            0,
                                        ),
                                    ]
                                ]
                            else:
                                ground_truths = np.concatenate(
                                    (
                                        ground_truths,
                                        [
                                            [
                                                self.normalize(polar_radius, 2, 0),
                                                self.normalize(
                                                    polar_theta,
                                                    360,
                                                    0,
                                                ),
                                            ]
                                        ],
                                    ),
                                    axis=0,
                                )

        truth_shape = np.array(ground_truths).shape
        # print(truth_shape)
        # print(np.array(ground_truths).shape)

        ground_truths = np.array(ground_truths).reshape(truth_shape[0], truth_shape[1])

        train_data_np = np.array(train_data)
        shape = train_data_np.shape
        train_data_np = train_data_np.reshape(shape[0], shape[1], shape[2], shape[3])
        return train_data_np, np.array(ground_truths), np.array(filenames)


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

    # TODO update this so to use 3D convolutions
    # channels will be RGB values, first dimension will be radar products
    def get_batch(self, ml_set, dualPol, radar_product=None, num_temporal_data=0):
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
            self, ml_set, dualPol, radar_product
        )
        for ml_sets in [roost_sets, no_roost_sets]:
            indices = Batch_Generator.get_batch_indices(self, ml_sets, ml_set)
            for index in indices:
                filename = ml_sets[ml_set][index]
                filenames.append(filename)
                is_roost = int(self.label_dict[filename].is_roost)
                images = []
                if dualPol:
                    radar_products = utils.Radar_Products
                else:
                    radar_products = utils.Legacy_radar_products
                for radar_product in radar_products:
                    image = self.label_dict[filename].get_image(radar_product)
                    images.append(image)
                ground_truths.append([is_roost, 1 - is_roost])
                train_data.append(images)
        # Update to channel last ordering
        train_data = np.rollaxis(np.array(train_data), 1, 4)
        return train_data, np.array(ground_truths), np.array(filenames)


class Temporal_Batch_Generator(Batch_Generator):
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
            ml_split_csv,
            validate_k_index,
            test_k_index,
            default_batch_size,
            root_dir,
        )
        ml_label_pd = pandas.read_csv(ml_label_csv)
        for _, row in ml_label_pd.iterrows():
            Labels.Temporal_ML_Label(
                row["AWS_file"], row, root_dir, high_memory_mode, self.label_dict
            )

    def get_batch(self, ml_set, dualPol, radar_product=None, num_temporal_data=0):
        ground_truths, train_data, filenames, roost_sets, no_roost_sets = Batch_Generator.get_batch(
            self, ml_set, dualPol, radar_product
        )

        for ml_sets in [roost_sets, no_roost_sets]:
            indices = Batch_Generator.get_batch_indices(self, ml_sets, ml_set)

            for index in indices:
                filename = ml_sets[ml_set][index]
                filenames.append(filename)
                is_roost = int(self.label_dict[filename].is_roost)
                images = []
                channel_files = self.label_dict[filename].fileNames[:]
                for image_name in channel_files.splitlines(True):
                    image = self.label_dict[image_name].get_image(
                        radar_product
                    )  # original image + augmented images
                    # print("self.label_dict[image_name].fileName")
                    # print(self.label_dict[image_name].fileName)

                    if isinstance(image, (list,)):
                        for i in image:
                            if i is not None:
                                images.append(i)
                    else:
                        if image is not None:
                            images.append(image)

                # print(images)
                # print(channel_files)
                # print(str((num_temporal_data * 24) + 1))
                # print(np.array(images).shape)

                if len(images) == (num_temporal_data * 24) + 1:  # 24 or 3?
                    ground_truths.append([is_roost, 1 - is_roost])
                    train_data.append(images)
        # print("TRAIN DATA")
        # print(np.array(train_data).shape)
        train_data = np.rollaxis(np.array(train_data), 1, 4)

        # print(train_data.shape)

        return train_data, np.array(ground_truths), np.array(filenames)


class Color_Image_Batch_Generator(Batch_Generator):
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
            ml_split_csv,
            validate_k_index,
            test_k_index,
            default_batch_size,
            root_dir,
        )
        ml_label_pd = pandas.read_csv(ml_label_csv)
        for _, row in ml_label_pd.iterrows():
            self.label_dict[row["AWS_file"]] = Labels.Color_ML_Label(
                row["AWS_file"], row, self.root_dir, high_memory_mode
            )

    def get_batch(self, ml_set, dualPol, radar_product=None, num_temporal_data=0):
        """Get a batch of data for machine learning. As a default a batch
        contains data from for a single radar product.

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
        for ml_sets in [roost_sets, no_roost_sets]:
            indices = Batch_Generator.get_batch_indices(self, ml_sets, ml_set)
            for index in indices:
                filename = ml_sets[ml_set][index]
                filenames.append(filename)
                is_roost = int(self.label_dict[filename].is_roost)
                image = self.label_dict[filename].get_image(radar_product)
                ground_truths.append([is_roost, 1 - is_roost])
                train_data.append(image)
        train_data_np = np.array(train_data)
        shape = train_data_np.shape
        train_data_np = train_data_np.reshape(shape[0], shape[1], shape[2], 1)
        return train_data_np, np.array(ground_truths), np.array(filenames)
