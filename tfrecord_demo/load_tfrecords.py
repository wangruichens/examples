import tensorflow as tf
from dataset import dataset_utils
import configs

depth=3
width=182
height=182


def parser(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    # input format.
    features = tf.parse_single_example(
            serialized_example,
            features={
                'image/class/label': tf.FixedLenFeature([], tf.int64),
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'image/height': tf.FixedLenFeature([], tf.int64),
                'image/width': tf.FixedLenFeature([], tf.int64),
            })
    image = tf.image.decode_png(features['image/encoded'], channels=3)
    # TODO change into
    image = tf.reshape(image, [182, 182, 3])
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['image/class/label'], tf.int32)
    # TODO Custom preprocessing... like crop,rotate,reshape,etc..
    return image, label


def make_batch(data_dir,subset,batch_size):
    # Repeat infinitely.
    dataset = tf.data.TFRecordDataset(data_dir).repeat()
    # Parse records.
    dataset = dataset.map(parser, batch_size)

    # Potentially shuffle records.
    if subset == 'train':
        min_after_dequeue = 5000
        capacity = min_after_dequeue + 3 * batch_size
        # Ensure that the capacity is sufficiently large to provide good random
        # shuffling.
        dataset = dataset.shuffle(capacity)

    # Batch it up.
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch = iterator.get_next()

    return image_batch, label_batch


