import argparse
import tensorflow as tf
import iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000,
                    type=int, help='training step')


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    # print(dataset)
    dataset = dataset.shuffle(buffer_size=1000).repeat(
        count=None).batch(batch_size)
    # print(dataset)
    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels=None, batch_size=None):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    assert batch_size is not None, "batch size can not be none"
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def my_model_fn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
    logits = tf.layers.dense(net, params['n_classes'], activation=None)
    predicted_classes = tf.argmax(logits, 1)

    # return with accuracy
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(
        labels=labels,
        predictions=predicted_classes,
        name='acc_op'
    )
    metrics = {'accuracy': accuracy}
    # for tensorboard
    tf.summary.scalar('accuracy', accuracy[1])

    # return with loss, metrics(optional)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # return with loss ,train op
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    print(argv)
    args = parser.parse_args(argv[1:])
    (train_x, train_y), (test_x, test_y) = iris_data.load_data()

    print(test_x.head())
    my_feature_columns = []
    for key in train_x.keys():
        print(key)
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    my_checkpoint_config = tf.estimator.RunConfig(
        save_checkpoints_secs=20*60,
        keep_checkpoint_max=5,
    )
    classifier = tf.estimator.Estimator(
        model_fn=my_model_fn,
        model_dir='./models/iris',
        params={
            'feature_columns': my_feature_columns,
            'hidden_units': [10, 10],
            'n_classes': 3,
        }
    )

    classifier.train(
        input_fn=lambda: train_input_fn(
            train_x, train_y, args.batch_size),
        steps=args.train_steps
    )
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(test_x, test_y, args.batch_size)
    )
    print('\nTest accuracy:{accuracy:0.3f}\n'.format(**eval_result))

    expected = ['Setosa', 'Versicolor', 'Virginica']
    SPECIES = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }
    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(predict_x, batch_size=args.batch_size)
    )
    template = ('\n prediction is "{}" ({:.1f}%),expect {}')
    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]
        print(template.format(SPECIES[class_id], 100*probability, expec))


if __name__ == '__main__':
    print('executing...')
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
