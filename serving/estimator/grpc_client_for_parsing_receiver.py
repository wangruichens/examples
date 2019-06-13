
# GRPC remote call using estimator model with build_parsing_serving_input_receiver_fn

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc

tf.app.flags.DEFINE_string('server', 'localhost:8555',
                           'Server host:port.')
tf.app.flags.DEFINE_string('model', 'iris',
                           'Model name.')
FLAGS = tf.app.flags.FLAGS


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def main(_):
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = FLAGS.model
    request.model_spec.signature_name = 'predict'

    feature_dict = {'SepalLength': _float_feature(value=2.5),
                    'SepalWidth': _float_feature(value=0.5),
                    'PetalLength': _float_feature(value=1.1),
                    'PetalWidth': _float_feature(value=1.1)}

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    serialized = example.SerializeToString()

    feature_dict2 = {'SepalLength': _float_feature(value=3.5),
                    'SepalWidth': _float_feature(value=3.5),
                    'PetalLength': _float_feature(value=1.1),
                    'PetalWidth': _float_feature(value=1.1)}

    example2 = tf.train.Example(features=tf.train.Features(feature=feature_dict2))

    serialized2 = example2.SerializeToString()

    batching=[serialized2,serialized,serialized,serialized]
    request.inputs['examples'].CopyFrom(
        tf.make_tensor_proto(batching, shape=[len(batching)]))

    result_future = stub.Predict.future(request, 5.0)
    prediction = result_future.result().outputs['probabilities']

    print(prediction)


if __name__ == '__main__':
    tf.app.run()
