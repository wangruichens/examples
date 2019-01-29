import tensorflow as tf
import numpy as np
print(tf.VERSION)
print(tf.keras.__version__)


def main():
    filename = 'train.tfrecord'
    dataset = tf.data.TFRecordDataset(filename)

    def _parse_function(example_proto):
        keys_to_features = {'feature': tf.FixedLenFeature((54),tf.int64),
                            'label': tf.FixedLenFeature((7),tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        x=parsed_features['feature']
        y=parsed_features['label']
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        return x,y

    # Parse the record into tensors.
    dataset = dataset.map(_parse_function)
    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=1)
    # Repeat the input indefinitly
    dataset = dataset.repeat()
    # Generate batches
    dataset = dataset.batch(512)

    def create_model():
        # model
        inputs = tf.keras.Input(shape=(54,))  # Returns a placeholder tensor

        # A layer instance is callable on a tensor, and returns a tensor.
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        predictions = tf.keras.layers.Dense(7, activation='softmax')(x)

        model = tf.keras.Model(inputs=inputs, outputs=predictions)

        model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
        return model

    model=create_model()
    # model.summary()
    # Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
    iterator = dataset.make_one_shot_iterator()
    x, y = iterator.get_next()
    model.fit(x=x,y=y, epochs=1,steps_per_epoch=100000)


    # tf.keras.models.save_model(model,'keras_model',overwrite=True)
    # # model.save('keras_model')
    # new_model = tf.keras.models.load_model('keras_model')
    # new_model.summary()


    ############################
    dataset2 = tf.data.TFRecordDataset('test.tfrecord')
    def _parse_function(example_proto):
        keys_to_features = {'feature': tf.FixedLenFeature((54),tf.int64),
                            'label': tf.FixedLenFeature((7),tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        x=parsed_features['feature']
        # y=parsed_features['label']
        # x = tf.cast(x, tf.float32)
        # y = tf.cast(y, tf.float32)
        return x

    # Parse the record into tensors.
    dataset2 = dataset2.map(_parse_function)
    # Repeat the input indefinitly
    dataset2 = dataset2.repeat(1)
    # Generate batches
    dataset2 = dataset2.batch(1000)

    iterator = dataset2.make_one_shot_iterator()
    t = iterator.get_next()
    sess = tf.Session()
    import csv
    f=open('out.csv', 'w')
    while 1:
        try:
            a=sess.run([t])
            a=np.asarray(a)
            a=np.squeeze(a,axis=0)
            print(a.shape)
            res = model.predict(a)
            # print(res.tolist())
            # submission=np.concatenate((submission,res),axis=0)
            xres=[]
            for x in res:
                f.write(str(np.argmax(x)+1)+'\n')
        except tf.errors.OutOfRangeError:
            print("End of dataset")  # ==> "End of dataset"
            break
    #
    # x=np.ones((3,23))
    # print(x.shape)
    # res=model.predict(x)
    # print(res)


if __name__ == "__main__":
    main()