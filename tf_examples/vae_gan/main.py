import math
import os

import numpy as np
import scipy.misc
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
from scipy.misc import imsave
from tensorflow.examples.tutorials.mnist import input_data

from vae import VAE
from gan import GAN

flags=tf.flags
logging=tf.logging

flags.DEFINE_integer('batch_size',64,'batch size')
flags.DEFINE_integer("updates_per_epoch", 1000, "number of updates per epoch")
flags.DEFINE_integer("max_epoch", 10, "max epoch")
flags.DEFINE_float("learning_rate", 1e-2, "learning rate")
flags.DEFINE_string("working_directory", "./result", "")
flags.DEFINE_integer("hidden_size", 10, "size of the hidden VAE unit")
flags.DEFINE_string("model", "vae", "gan or vae")

FLAGS=flags.FLAGS

if __name__ == '__main__':
    data_directory='../mnist/'
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    if not os.path.exists(FLAGS.working_directory):
        os.makedirs(FLAGS.working_directory)
    mnist=input_data.read_data_sets(data_directory,one_hot=True)
    assert FLAGS.model in ['vae','gan']
    model = VAE(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)
    # if FLAGS.model == 'gan':
    # model = GAN(FLAGS.hidden_size, FLAGS.batch_size, FLAGS.learning_rate)

    print('Start training...')
    for epoch in range(FLAGS.max_epoch):
        training_loss = 0.0

        for i in range(FLAGS.updates_per_epoch):
            images, _ = mnist.train.next_batch(FLAGS.batch_size)
            loss_value = model.update_params(images)
            training_loss += loss_value

        training_loss = training_loss / \
                        (FLAGS.updates_per_epoch * FLAGS.batch_size)

        print("Loss %f" % training_loss)

        model.generate_and_save_images(
            FLAGS.batch_size, FLAGS.working_directory,epoch)
