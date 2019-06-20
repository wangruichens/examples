import tensorflow as tf
import matplotlib.pyplot as plt

filename = 'F:\\lfw_dataset\\test_train_00000-of-00001.tfrecord'  # address to save the hdf5 file


def read_and_decode(filename):
    filename_queue = tf.train.string_input_producer([filename])

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'image/class/label': tf.FixedLenFeature([], tf.int64),
                                           'image/encoded' : tf.FixedLenFeature([], tf.string),
                                       })

    img = tf.image.decode_png(features['image/encoded'], channels=3)
    img = tf.reshape(img, [182, 182, 3])
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5
    label = tf.cast(features['image/class/label'], tf.int32)

    return img, label

def main():
    img, label = read_and_decode(filename)
    img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=30, capacity=2000,
                                                    min_after_dequeue=1000)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(10):
            val, l,lmax= sess.run([img_batch, label_batch,label])
            plt.imshow(val[0])
            plt.show()
            print(val.shape, l)


if __name__ == "__main__":
    main()