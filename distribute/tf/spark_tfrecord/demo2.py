# export HADOOP_HOME=/tmp/wangrc/hadoop-2.7.3
# export CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)

import tensorflow as tf
from pyspark.sql import SparkSession
tf.enable_eager_execution()

####################load dataframe#########################
ss = SparkSession.builder \
    .appName("train") \
    .enableHiveSupport() \
    .getOrCreate()
df=ss.sql('select * from mlg.g_tfrecord_test')


####################save dataframe to tfrecord #########################
folder='test.tfrecord'
path='hdfs://cluster/user/wangrc/test.tfrecord/part-r-00000'
df.repartition(1).write.format("tfrecords").mode("overwrite").option("recordType", "Example").save(folder)


####################load tfrecord to dataframe #########################
df = ss.read.format("tfrecords").option("recordType", "Example").load(path)



####################read tfrecord DEPRECATED #########################
record_iterator = tf.python_io.tf_record_iterator(path=path)
for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    print(example)


####################read tfrecord #########################
raw_dataset = tf.data.TFRecordDataset(path)
for raw_record in raw_dataset.take(2):
    print(repr(raw_record))


####################decode tfrecord #########################


def _parse_function(example_proto):
    feature_description = {
        'id': tf.FixedLenFeature([], tf.int64, default_value=0),
        'IntegerCol': tf.FixedLenFeature([], tf.int64, default_value=0),
        'LongCol': tf.FixedLenFeature([], tf.int64, default_value=0),
        'FloatCol': tf.FixedLenFeature([], tf.float32, default_value=0.0),
        'DoubleCol': tf.FixedLenFeature([], tf.float32, default_value=0.0),
        'VectorCol': tf.FixedLenFeature((2), tf.float32, default_value=[0, 0]),
        'StringCol': tf.FixedLenFeature([], tf.string, default_value=''),
    }
    return tf.parse_single_example(example_proto, feature_description)

dataset = raw_dataset.map(_parse_function)

for p in dataset.take(2):
    print(repr(p))


