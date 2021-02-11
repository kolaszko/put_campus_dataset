from put_campus_dataset import *
import matplotlib.pyplot as plt


if __name__ == '__main__':

    ds, _ = put_campus_dataset('path_to_ds')

    ds = ds.batch(1)

    for img, label in ds.take(1):
        plt.subplot(121)
        plt.imshow(tf.squeeze(img))

        plt.subplot(122)
        plt.imshow(tf.squeeze(label))

    plt.show()

