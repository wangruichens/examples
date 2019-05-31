import tensorflow as tf
from tensorflow import keras
from tensorflow import flags,logging
from build_input import *

logging.set_verbosity(tf.logging.INFO)
flags.DEFINE_string('model_dir', 'output/ckpt', '')
flags.DEFINE_integer('save_checkpoints_steps', 10000, '')
flags.DEFINE_integer('feature_embed_size', 50, '')
flags.DEFINE_string('dnn_hidden_units', '200,200,200', '')
flags.DEFINE_float('dnn_dropout_rate', 0.8, '')
flags.DEFINE_integer('train_steps', 312500, '')
flags.DEFINE_integer('batch_size', 128, '')
flags.DEFINE_integer('shuffle_buffer_size', 128 * 2, '')
flags.DEFINE_float('learning_rate', 0.05, '')
FLAGS = flags.FLAGS

linear_feature_columns, embed_feature_columns = create_feature_columns(FLAGS.feature_embed_size)

def fm_1d(feature,linear_inputs):
    linear_feature  = feature_column.input_layer(feature, linear_inputs)
    fm_linear_1d = [keras.layers.Dense(1, name='fm_dense_1d')(linear_feature)]
    return fm_linear_1d


def fm_2d(feature,embed_inputs):
    embed_feature = feature_column.input_layer(feature, embed_inputs)



def deepfm_model():
