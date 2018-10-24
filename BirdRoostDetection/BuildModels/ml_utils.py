import tensorflow as tf

KERAS_SAVE_FILE = '{}{}.h5'
LOG_PATH = 'model/{}/{}/'
LOG_PATH_TIME =  'model/{}/{}/{}/'
CHECKPOINT_DIR = '/checkpoint/'

def write_log(callback, names, logs, batch_no):
    """Write out progress training keras model to tensorboard.

    When training a model that takes a long time to train, it can be useful
    to be able to see how progress is going while the model is training.
    Tensorboard allows us to see a graph of how well the learning is doing
    over time. We setup the model, the name of the graphs we are creating, as
    well as the x and y values of the graph.

    Code taken from :
        https://gist.github.com/joelthchao/ef6caa586b647c3c032a4f84d52e3a11

    Args:
        callback: The callback to the tensorboard, can be set as follows
            callback = TensorBoard(log_path)
            callback.set_model(model)
        names: An array of graph names, these are the graphs that will be
            displayed in tensorboard.
        logs: the y values on the graphs, these is an array of values
            corresponding with the names array.
        batch_no: The batch number, this will serve as the x value of the
            graphs.
    """
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()
