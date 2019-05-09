# Auther        : wangrc
# Date          : 2019-05-07
# Description   :
# Refers        :
# Returns       :
import argparse
from pyspark.sql import SparkSession
import tensorflow as tf
from tensorflow.python.client import device_lib


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default='mlg')
    args = parser.parse_args()
    return args


def df_to_hive(spark, df, table_name):
    tmp_table_name = "tmp_" + table_name
    df.registerTempTable(tmp_table_name)
    delete_sql = "drop table if exists " + table_name
    create_sql = "create table " + table_name + " as select * from " + tmp_table_name
    spark.sql(delete_sql)
    spark.sql(create_sql)


def main(args):
    print(get_available_gpus())
    gpu_num=len(get_available_gpus())
    ss = SparkSession.builder \
        .appName("train_from_tfrecord") \
        .enableHiveSupport() \
        .getOrCreate()

    path = 'hdfs://cluster/user/wangrc/mnist.tfrecord/train/part-r-0000'
    filenames = []
    for i in range(10):
        filenames.append(path + str(i))

    raw_dataset = tf.data.TFRecordDataset(filenames)
    feature_description = {
        'label': tf.FixedLenFeature((), tf.int64),
        'features': tf.FixedLenFeature((784), tf.int64),
        }

    #Note here:
    #  tf.enable_eager_execution() fetch data when you call it.
    #  So the data shape can not be inferred. Need to specific the input data shape with (-1,28,28,1) and the label (-1,1)
    def _parse_function(example_proto):
        parsed_feature= tf.parse_single_example(example_proto, feature_description)
        return tf.reshape(tf.cast(parsed_feature['features'], tf.float32),[28,28,1]), \
               tf.cast(parsed_feature['label'], tf.float32)

    dataset = raw_dataset.map(_parse_function,num_parallel_calls=4).shuffle(buffer_size=256).prefetch(256).batch(256).repeat()
    # for p in dataset.take(1):
    #     print(repr(p))

    inputs = tf.keras.layers.Input(shape=(28,28,1))
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=predictions)

    if gpu_num>1:
        model = tf.keras.utils.multi_gpu_model(model, gpus=gpu_num)
    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    model.summary()
    model.fit(dataset, epochs=10, steps_per_epoch=100)
    tf.keras.backend.clear_session()

if __name__ == '__main__':

    args = parse_args()
    main(args)