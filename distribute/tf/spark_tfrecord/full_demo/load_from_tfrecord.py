# Auther        : wangrc
# Date          : 2019-05-07
# Description   :
# Refers        :
# Returns       :
import argparse
from pyspark.sql import SparkSession
import tensorflow as tf

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



    ss = SparkSession.builder \
        .appName("train_from_tfrecord") \
        .enableHiveSupport() \
        .getOrCreate()

    path = 'hdfs://cluster/user/wangrc/mnist.tfrecord/train/part-r-0000'
    filenames = []
    for i in range(10):
        filenames.append(path + str(i))

    # raw_dataset = tf.data.TFRecordDataset(filenames)

    record_iterator = tf.python_io.tf_record_iterator(path='hdfs://cluster/user/wangrc/test1.tfrecord/part-r-00000')
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        print(example)

    #
    # feature_description = {
    #     'label': tf.FixedLenFeature([], tf.int64),
    #     'features': tf.FixedLenFeature((784), tf.int64),
    #     }
    #
    # def _parse_function(example_proto):
    #   return tf.parse_single_example(example_proto, feature_description)
    #
    # dataset = raw_dataset.map(_parse_function)


if __name__ == '__main__':
    args = parse_args()
    main(args)