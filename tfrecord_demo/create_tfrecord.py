import random
import configs
from dataset.dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset


def main():
    # Check if there is a tfrecord_filename entered
    if not configs.tfrecord_filename:
        raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')

    # Check if there is a dataset directory entered
    if not configs.dataset_dir:
        raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')

    # If the TFRecord files already exist in the directory, then exit without creating the files again
    if _dataset_exists(dataset_dir=configs.dataset_dir, _NUM_SHARDS=configs.num_shards,
                       output_filename=configs.tfrecord_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return None

    # Get a list of photo_filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
    photo_filenames, class_names = _get_filenames_and_classes(configs.dataset_dir)

    # Refer each of the class name to a specific integer number for predictions later
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    # Find the number of validation examples we need
    num_validation = int(configs.validation_set_split_ratio * len(photo_filenames))

    # Divide the training datasets into train and test:
    random.seed(configs.random_seed)
    random.shuffle(photo_filenames)
    training_filenames = photo_filenames[num_validation:]
    validation_filenames = photo_filenames[:num_validation]

    # Convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids,
                     dataset_dir=configs.dataset_dir, tfrecord_filename=configs.tfrecord_filename,
                     _NUM_SHARDS=configs.num_shards)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir=configs.dataset_dir, tfrecord_filename=configs.tfrecord_filename,
                     _NUM_SHARDS=configs.num_shards)

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    write_label_file(labels_to_class_names, configs.dataset_dir)

    print('\nFinished converting the %s dataset!' % configs.tfrecord_filename)


if __name__ == "__main__":
    main()
