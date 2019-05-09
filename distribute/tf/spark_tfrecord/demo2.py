# df.repartition(1).write.format("tfrecords").option("recordType", "Example").save(path)
# ss = SparkSession.builder \
#     .appName("train") \
#     .enableHiveSupport() \
#     .getOrCreate()
# df=ss.sql('select * from mlg.g_tfrecord_test')

path='hdfs://cluster/user/wangrc/test1.tfrecord/part-r-0000'
filenames=[]
for i in range(10):
    filenames.append(path+str(i))

print(filenames)

import tensorflow as tf
tf.enable_eager_execution()


path='hdfs://cluster/user/wangrc/test1.tfrecord/part-r-00000'
raw_dataset = tf.data.TFRecordDataset(path)
for raw_record in raw_dataset.take(2):
    print(repr(raw_record))


feature_description = {
    'id': tf.FixedLenFeature([], tf.int64, default_value=0),
    'IntegerCol': tf.FixedLenFeature([], tf.int64, default_value=0),
    'LongCol': tf.FixedLenFeature([], tf.int64, default_value=0),
    'FloatCol': tf.FixedLenFeature([], tf.float32, default_value=0.0),
    'DoubleCol': tf.FixedLenFeature([], tf.float32, default_value=0.0),
    'VectorCol': tf.FixedLenFeature((2), tf.float32, default_value=[0,0]),
    'StringCol': tf.FixedLenFeature([], tf.string, default_value=''),
    }

def _parse_function(example_proto):
  return tf.parse_single_example(example_proto, feature_description)

dataset = raw_dataset.map(_parse_function)

for p in dataset.take(2):
    print(repr(p))



record_iterator = tf.python_io.tf_record_iterator(path='hdfs://cluster/user/wangrc/tfrecord_test/part-r-00000')
for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    print(example)