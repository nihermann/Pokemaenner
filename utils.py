import tensorflow as tf
from time import time_ns
import os


def save_images(images, save_to="", prefix=""):
    """
    Used to save images to file.
    :param images: shape=(int number, x, y, channel): where all image values are in range [0,1].
    :param save_to: string - path to the destination directory. Don't forget the '/' at the end!
    :param prefix: string - name prefix for saved images.
    """
    # convert the images back to rgb values. Before: [0:1], After: [0:255]
    images = tf.cast(images * 255, tf.uint8)

    if not os.path.exists(save_to):
        os.mkdir(save_to)

    for image in images:
        # encode/compress the image to reduce memory (needed to write them to file).
        compressed_image = tf.io.encode_png(image, name=prefix)

        # take the nano secs as id. We exclude the last two numbers as they are always 0
        # and the first ones as they stay mostly the same.
        current_time = int(time_ns() / 100) % 10000

        tf.io.write_file(  # save to dir with prefix and nano sec id.
            filename=save_to + prefix + str(current_time) + ".png",
            contents=compressed_image
        )


def default_value(default, alternative):
    return default if alternative is None else alternative


if __name__ == "__main__":
    # ones = tf.random.uniform((64, 28, 28, 3))
    # save_images(ones, "pics/", "hihi")
    # print("done")
    pass
