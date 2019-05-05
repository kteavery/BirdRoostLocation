"""Train the shallow CNN model on a single radar product.

Use command line arguments to select which radar product to train the model on.
Optionally input the location of the save file where the default is
model/radar_product/
Use an integer to select a radar_product from the following list:
    0 : Reflectivity
    1 : Velocity
    2 : Correlation Coefficient
    3 : Differential Reflectivity

Example command:
python train.py \
--radar_product=0 \
--log_path=model/Reflectivity/ \
--eval_increment=5 \
--num_iterations=2500 \
--checkpoint_frequency=100 \
--learning_rate=.001 \
--model=0 \
--dual_pol=True \
--high_memory_mode=True
"""
import argparse
import os
import BirdRoostLocation.LoadSettings as settings
from BirdRoostLocation.BuildModels.ShallowCNN import model as keras_model
from keras.callbacks import TensorBoard
from BirdRoostLocation import utils
from BirdRoostLocation.BuildModels import ml_utils
from BirdRoostLocation.ReadData import BatchGenerator
import datetime


def train(log_path, radar_product, eval_increment=5,
          num_iterations=2500, checkpoint_frequency=100, lr=.0001,
          model_name=utils.ML_Model.Shallow_CNN, dual_pol=True,
          high_memory_mode=False, num_temporal_data=0):
    """Train the shallow CNN model on a single radar product.

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

    print("MODEL NAME")
    print(model_name)
    if model_name == utils.ML_Model.Shallow_CNN:
        batch_generator = BatchGenerator.Single_Product_Batch_Generator(
            ml_label_csv=settings.LABEL_CSV,
            ml_split_csv=settings.ML_SPLITS_DATA,
            high_memory_mode=high_memory_mode)
        model = keras_model.build_model(
            inputDimensions=(240, 240, 3), lr=lr, coordConv=False)

    elif model_name == utils.ML_Model.Shallow_CNN_All:
        batch_generator = BatchGenerator.Multiple_Product_Batch_Generator(
            ml_label_csv=settings.LABEL_CSV,
            ml_split_csv=settings.ML_SPLITS_DATA,
            high_memory_mode=high_memory_mode)
        model = keras_model.build_model(
            inputDimensions=(240, 240, 4), lr=lr, coordConv=False)

    else:
        batch_generator = BatchGenerator.Temporal_Batch_Generator(
            ml_label_csv=settings.LABEL_CSV,
            ml_split_csv=settings.ML_SPLITS_DATA,
            high_memory_mode=False)
        model = keras_model.build_model(
            inputDimensions=(240, 240, num_temporal_data * 3 + 1),
            lr=lr,
            coordConv=False)

    # Setup callbacks
    callback = TensorBoard(log_path)
    callback.set_model(model)
    train_names = ['train_loss', 'train_accuracy']
    val_names = ['val_loss', 'val_accuracy']

    progress_string = '{} Epoch: {} Loss: {} Accuracy {}'

    for batch_no in range(num_iterations):
        x, y, _ = batch_generator.get_batch(
            ml_set=utils.ML_Set.training,
            dualPol=dual_pol,
            radar_product=radar_product,
            num_temporal_data=num_temporal_data)
        print(len(y))

        print("X AND Y: ")
        print(x.shape)
        print(y.shape)
        train_logs = model.train_on_batch(x, y)
        print(progress_string.format(utils.ML_Set.training.fullname,
                                     batch_no,
                                     train_logs[0], train_logs[1]))
        ml_utils.write_log(callback, train_names, train_logs, batch_no)

        if (batch_no % eval_increment == 0):
            currentDT = datetime.datetime.now()
            model.save_weights(log_path + str(currentDT) +
                               save_file.format(''))
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
                x_, y_, _, x, y = None

            except Exception as e:
                print(e)

        if batch_no % checkpoint_frequency == 0 \
                or batch_no == num_iterations - 1:
            model.save_weights(
                os.path.join(checkpoint_path, save_file.format(batch_no)))

    print("SAVE FILE")
    print(save_file)
    model.save_weights(save_file)


def main(results):
    os.chdir(settings.WORKING_DIRECTORY)
    radar_product = utils.Radar_Products(results.radar_product)
    model = utils.ML_Model(results.model)
    if results.log_path is None:
        if results.model == 1:
            log_path = ml_utils.LOG_PATH.format(model.fullname,
                                                str(results.dual_pol))
        elif results.model == 0:
            log_path = ml_utils.LOG_PATH.format(model.fullname,
                                                radar_product.fullname)
        else:
            log_path = ml_utils.LOG_PATH_TIME.format(model.fullname,
                                                     results.num_temporal_data * 2 + 1,
                                                     radar_product.fullname)
    else:
        log_path = results.log_path

    print("Log path: ")
    print(log_path)
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
        default=1,
        help="""
            Use an integer to select a radar_product from the following list:
                0 : Reflectivity
                1 : Velocity
                2 : Correlation Coefficient
                3 : Differential Reflectivity
            """
    )

    parser.add_argument(
        '-l',
        '--log_path',
        type=str,
        default=None,
        help="""
            Optionally input the location of the save file where the default is
            model/radar_product
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
        '-m',
        '--model',
        type=int,
        default=0,
        help="""
            Use an integer to select a model from the following list:
                0 : Shallow CNN
                1 : Shallow CNN, all radar products
                2 : Shallow CNN, temporal model
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
