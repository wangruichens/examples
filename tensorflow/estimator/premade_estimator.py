import argparse
import tensorflow as tf
import iris_data

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000,
                    type=int, help='training step')


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    print(dataset)
    dataset = dataset.shuffle(buffer_size=1000).repeat(
        count=None).batch(batch_size)
    print(dataset)
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
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[20, 20],
        n_classes=3,
        model_dir='./models/iris',
        config=my_checkpoint_config
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
