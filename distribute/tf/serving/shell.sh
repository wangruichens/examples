$ python trainer.py --job_name=ps
$ python trainer.py --job_name=worker



docker run -it --rm tensorflow/tensorflow:latest-gpu \
   python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"