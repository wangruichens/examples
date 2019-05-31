import tensorflow as tf
from tensorflow.contrib import layers


def encoder(input_tensor, output_size):
    net = tf.reshape(input_tensor, [-1, 28, 28, 1])
    net = layers.conv2d(net, 32, 5, stride=2)
    net = layers.conv2d(net, 64, 5, stride=2)
    net = layers.conv2d(net, 128, 5, stride=2, padding='VALID')
    net = layers.dropout(net, keep_prob=0.9)
    net = layers.flatten(net)
    return layers.fully_connected(net, output_size, activation_fn=None)


def discriminator(input_tensor):
    return encoder(input_tensor, 1)


def decoder(input_tensor):
    net = tf.expand_dims(input_tensor, 1)
    net = tf.expand_dims(net, 1)
    net = layers.conv2d_transpose(net, 128, 3, padding='VALID')
    net = layers.conv2d_transpose(net, 64, 5, padding='VALID')
    net = layers.conv2d_transpose(net, 32, 5, stride=2)
    net = layers.conv2d_transpose(net, 1, 5, stride=2, activation_fn=tf.nn.sigmoid)
    net = layers.flatten(net)
    return net
