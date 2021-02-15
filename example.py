import matplotlib.pyplot as plt
import argparse

from put_campus_dataset import *


def example(path):
    ds, _ = put_campus_dataset(path)

    ds = ds.batch(1)

    for img, label in ds.take(1):
        plt.subplot(121)
        plt.imshow(tf.squeeze(img))

        plt.subplot(122)
        plt.imshow(tf.squeeze(label))

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Path to put_campus_dataset')

    args, _ = parser.parse_known_args()

    example(args.path)
