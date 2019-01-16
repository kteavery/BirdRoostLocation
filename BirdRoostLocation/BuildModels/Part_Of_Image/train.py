import argparse
import os
import BirdRoostLocation.LoadSettings as settings
from BirdRoostLocation.BuildModels.ShallowCNN import model as keras_model
from keras.callbacks import TensorBoard
from BirdRoostLocation import utils
from BirdRoostLocation.BuildModels import ml_utils
from BirdRoostLocation.ReadData import BatchGenerator
import numpy as np


def train(log_path, radar_product, eval_increment=5,
          num_iterations=2500, checkpoint_frequency=100, lr=.0001,
          model_name=utils.ML_Model.Shallow_CNN, dual_pol=True,
          high_memory_mode=False, num_temporal_data=0):
    """"Train the shallow CNN model on a single radar product.

    Args:
        log_path: The location of the save directory. The model checkpoints,
            model weights, and the tensorboard events are all saved in this
            directory.
        radar_product: The radar product the model is training on. This should
            be a value of type utils.Radar_Products.
        eval_increment: How frequently the model prints checks validation result
        num_iterations: The number of training iterations the model will run.
        checkpoint_frequency: How many training iterations should the model
            perform before saving out a checkpoint of the model training.
        lr: The learning rate of the model, this value must be between 0 and 1.
            e.g. .1, .05, .001
        model_name: Select the model to train. Must be of type utils.ML_Model
        dual_pol: True if data training on dual polarization radar data, false
            when training on legacy data.
        high_memory_mode: True if training in high memory mode. High memory
            mode reduces the amount of IO operations by keeping all the data in
            memory during trainig. Not recommended for computes with fewer than
            8 GB of memory.
    """
    save_file = ml_utils.KERAS_SAVE_FILE.format(radar_product.fullname, '{}')

    checkpoint_path = log_path + ml_utils.CHECKPOINT_DIR
    if not os.path.exists(checkpoint_path):
        os.makedirs(os.path.dirname(checkpoint_path))
    batch_generator = BatchGenerator.Small_Image_Batch_Generator(
        ml_label_csv=settings.LABEL_CSV,
        ml_split_csv=settings.ML_SPLITS_DATA,
        high_memory_mode=high_memory_mode)
    model = keras_model.smaller_build_model(inputDimensions=(80, 80, 1), lr=lr)

    # Setup callbacks
    callback = TensorBoard(log_path)
    callback.set_model(model)
    train_names = ['train_loss', 'train_accuracy']
    val_names = ['val_loss', 'val_accuracy']

    progress_string = '{} Epoch: {} Loss: {} Accuracy {}'

    for batch_no in range(num_iterations):
        try:
            x, y, _ = batch_generator.get_batch(
                ml_set=utils.ML_Set.training,
                dualPol=dual_pol,
                radar_product=radar_product,
                num_temporal_data=num_temporal_data)


            train_logs = model.train_on_batch(x, y)
            print(progress_string.format(utils.ML_Set.training.fullname,
                                         batch_no,
                                         train_logs[0], train_logs[1]))
            ml_utils.write_log(callback, train_names, train_logs, batch_no)
        except Exception as e:
            print(e.message)
        if (batch_no % eval_increment == 0):
            model.save_weights(log_path + save_file.format(''))
            try:
                x_, y_, _ = batch_generator.get_batch(
                    ml_set=utils.ML_Set.validation,
                    dualPol=dual_pol,
                    radar_product=radar_product,
                    num_temporal_data=num_temporal_data)

                val_logs = model.test_on_batch(x_, y_)
                ml_utils.write_log(callback, val_names, val_logs, batch_no)
                print(progress_string.format(utils.ML_Set.validation.fullname,
                                             batch_no,
                                             val_logs[0], val_logs[1]))
            except Exception as e:
                print(e.message)

        if batch_no % checkpoint_frequency == 0 \
                or batch_no == num_iterations - 1:
            model.save_weights(
                os.path.join(checkpoint_path, save_file.format(batch_no)))

    model.save_weights(save_file)


def main(results):
    os.chdir(settings.WORKING_DIRECTORY)
    radar_product = utils.Radar_Products(results.radar_product)
    model = utils.ML_Model.Shallow_CNN
    log_path = 'model/small_images/'

    train(log_path=log_path,
          radar_product=radar_product,
          eval_increment=results.eval_increment,
          num_iterations=results.num_iterations,
          checkpoint_frequency=results.checkpoint_frequency,
          lr=results.learning_rate,
          model_name=model,
          dual_pol=results.dual_pol,
          high_memory_mode=results.high_memory_mode,
          num_temporal_data=results.num_temporal_data)


if __name__ == "__main__":
    os.chdir(settings.WORKING_DIRECTORY)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-r',
        '--radar_product',
        type=int,
        default=0,
        help="""
            Use an integer to select a radar_product from the following list:
                0 : Reflectivity
                1 : Velocity
                2 : Correlation Coefficient
                3 : Differential Reflectivity
            """
    )

    parser.add_argument(
        '-e',
        '--eval_increment',
        type=int,
        default=5,
        help="""How frequently the model prints out the validation results."""
    )

    parser.add_argument(
        '-n',
        '--num_iterations',
        type=int,
        default=2500,
        help="""The number of training iterations the model will run"""
    )

    parser.add_argument(
        '-c',
        '--checkpoint_frequency',
        type=int,
        default=100,
        help="""
            How many training iterations should the model perform before saving 
            out a checkpoint of the model training.
            """
    )

    parser.add_argument(
        '-lr',
        '--learning_rate',
        type=float,
        default=.0001,
        help="""
            The learning rate of the model, this value must be between 0 and 1
            .e.g. .1, .05, .001
            """
    )


    parser.add_argument(
        '-d',
        '--dual_pol',
        type=bool,
        default=True,
        help="""
            This field will only be used if model = 1 
            True if model is training on dual polarization radar data, false if 
            the model is training on legacy data.
            """
    )

    parser.add_argument(
        '-hm',
        '--high_memory_mode',
        type=bool,
        default=False,
        help="""
            If true then all of the data will be read in at the beginning and 
            stored in memory. Otherwise only one batch of data will be in 
            memory at a time. high_memory_mode is good for machines with slow 
            IO and at least 8 GB of memory available.
            """
    )
    parser.add_argument(
        '-td',
        '--num_temporal_data',
        type=int,
        default=1,
        help="""
                Only applied to temporal model. This indicates how many time
                frames in either direction used for training. 0 will give array
                size of 1, 1 -> 3, 2 -> 5, and 3 -> 7.
                """
    )
    results = parser.parse_args()
    main(results)
