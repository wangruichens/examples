import tensorflow as tf

filename = 'test.tfrecord'

def main():
    dataset = tf.data.TFRecordDataset(filename)

    def _parse_function(example_proto):
        keys_to_features = {'feature': tf.FixedLenFeature((54),tf.int64),
                            'label': tf.FixedLenFeature((7),tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        return parsed_features['feature'],parsed_features['label']

    # Parse the record into tensors.
    dataset = dataset.map(_parse_function)
    # # Shuffle the dataset
    # dataset = dataset.shuffle(buffer_size=1)
    # Repeat the input indefinitly
    dataset = dataset.repeat(1)
    # Generate batches
    dataset = dataset.batch(5)
    # Create a one-shot iterator
    iterator = dataset.make_one_shot_iterator()
    f, l = iterator.get_next()
    with tf.Session() as sess:
        print(sess.run([f, l]))


if __name__ == "__main__":
    main()