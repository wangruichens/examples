from __future__ import print_function
import tensorflow as tf
import argparse
import time
import os

FLAGS = None
log_dir = '/logdir'


def main():
    # Distributed Baggage
    cluster = tf.train.ClusterSpec({
        'ps': ['localhost:2222'],
        'worker': ['localhost:2223']
    })  # lets this node know about all other nodes
    if FLAGS.job_name == 'ps':  # checks if parameter server
        server = tf.train.Server(cluster, job_name="ps", task_index=FLAGS.task_index)
        server.join()
    else:
        is_chief = (FLAGS.task_index == 0)  # checks if this is the chief node
        server = tf.train.Server(cluster, job_name="worker", task_index=FLAGS.task_index)

        # Graph
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:0",
                cluster=cluster)):
            a = tf.Variable(tf.truncated_normal(shape=[2]), dtype=tf.float32)
            b = tf.Variable(tf.truncated_normal(shape=[2]), dtype=tf.float32)
            c = a + b

            target = tf.constant(50., shape=[2], dtype=tf.float32)
            loss = tf.reduce_mean(tf.square(c - target))
            global_step = tf.train.get_or_create_global_step()
            opt = tf.train.GradientDescentOptimizer(.01).minimize(loss,global_step=global_step)

        # Session
        # Supervisor
        hooks = [tf.train.StopAtStepHook(last_step=1000000)]
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction=0.5

        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=is_chief,
                                               checkpoint_dir="/tmp/train_logs",
                                               hooks=hooks,config=config) as mon_sess:
            for i in range(10000):
                if mon_sess.should_stop(): break
                mon_sess.run(opt)
                if i % 10 == 0:
                    res = mon_sess.run(c)
                    print(res)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Flags for defining the tf.train.ClusterSpec
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="One of 'ps', 'worker'"
    )
    # Flags for defining the tf.train.Server
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    main()