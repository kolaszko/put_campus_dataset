import tensorflow as tf
import os

NUM_CLASSES = 12

weights = [0, 3.03179758, 6.12635109, 6.62935517, 5.00587831, 5.10223856,
           5.70209559, 6.76043488, 7.14426043, 7.67151577, 6.27170891, 10.45687937]


def put_campus_dataset(path, image_size=None, augment_function=None, num_parallel_calls=tf.data.experimental.AUTOTUNE):

    def get_subdirectory_files(subdir):
        sub_path = os.path.join(path, subdir)
        return sorted([os.path.join(dp, f) for dp, dn, fn in
                       os.walk(os.path.expanduser(sub_path), followlinks=True) for f in
                       fn if f.endswith('.png')])

    def parse_function(image_path, labels_path):
        image = _read_image(image_path)
        labels = _read_labels(labels_path)

        if augment_function:
            image, labels = augment_function(image, labels)

        if image_size:
            image = tf.image.resize(image, image_size)
            labels = tf.image.resize(labels, image_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        return image, labels

    data_files = get_subdirectory_files('rgb')
    target_files = get_subdirectory_files('gt')

    assert len(data_files) == len(target_files)

    ds = tf.data.Dataset.from_tensor_slices((data_files, target_files)) \
        .shuffle(len(data_files)) \
        .map(parse_function, num_parallel_calls)

    return ds, len(data_files)


def _read_image(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_png(x, 3)
    x = tf.image.convert_image_dtype(x, tf.float32)
    return x


def _read_labels(path):
    x = tf.io.read_file(path)
    x = tf.image.decode_png(x, 1)
    x = tf.cast(x, tf.int32)
    return x


def get_weights():
    return tf.constant([0, 3.03179758, 6.12635109, 6.62935517, 5.00587831, 5.10223856,
                        5.70209559, 6.76043488, 7.14426043, 7.67151577, 6.27170891, 10.45687937])


def get_stats_weights():
    return tf.constant([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


def get_colors():
    return tf.constant(
        [[0, 0, 0], [122, 121, 116], [99, 104, 122], [212, 212, 129], [33, 37, 48],
         [7, 184, 4], [121, 224, 119], [87, 51, 4], [130, 105, 73], [48, 40, 29],
         [230, 5, 132], [3, 66, 19]], tf.uint8)


def get_names():
    return ['_background_',
            'concrete_paver',
            'concrete_slab',
            'granite_cubes',
            'asphalt',
            'grass',
            'artificial_grass',
            'dirt',
            'gravel',
            'rocks',
            'rubber',
            'vegetation']
