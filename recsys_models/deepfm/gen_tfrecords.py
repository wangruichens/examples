from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("deepfm_tfrecords") \
    .enableHiveSupport() \
    .getOrCreate()
all_df = spark.sql("select * from mlg.tangjtest_20190515_article_rank_sample_2 limit 100")
all_df.repartition(1).write.format("tfrecords")\
    .mode("overwrite").option("recordType", "Example")\
    .save('./deepfm_data/')

