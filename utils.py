import tensorflow as tf
from time import time_ns
import types
import os


def save_images(images, save_to="", prefix=""):
    """
    Used to save images to file.
    :param images: shape=(int number, x, y, channel): where all image values are in range [0,1].
    :param save_to: string - path to the destination directory. Don't forget the '/' at the end!
    :param prefix: string - name prefix for saved images.
    """
    # convert the images back to rgb values. Before: [-1:1], After: [0:255]
    images = tf.cast((images + 1) / 2 * 255, tf.uint8)

    if not os.path.exists(save_to):
        os.mkdir(save_to)

    for image in images:
        # encode/compress the image to reduce memory (needed to write them to file).
        compressed_image = tf.io.encode_png(image, name=prefix)

        # take the nano secs as id. We exclude the last two numbers as they are always 0
        # and the first ones as they stay mostly the same.
        current_time = int(time_ns() / 100) % 100000

        tf.io.write_file(  # save to dir with prefix and nano sec id.
            filename=save_to + prefix + str(current_time) + ".png",
            contents=compressed_image
        )


def default_value(default, alternative):
    return default if alternative is None else alternative


def get_padded_cols(images, border):
    padding = tf.constant([[0, 0], [border, border], [border, border], [0, 0]])
    images = tf.pad(images, padding, mode="constant", constant_values=0)
    cols = tf.reshape(images, shape=(4, -1, images.shape[1], 3))
    return cols


def stack_alternating(a, b):
    stack = tf.stack([a, b], axis=1)
    return tf.reshape(stack, (stack.shape[0] * stack.shape[1], *stack.shape[2:]))


def to_grid(images, border):
    if images.ndim == 5:
        col1 = get_padded_cols(images[0], border)
        col2 = get_padded_cols(images[1], border)
        cols = stack_alternating(col1, col2)
    else:
        cols = get_padded_cols(images, border)

    grid = cols[0]
    for col in cols[1:]:
        grid = tf.concat([grid, col], axis=1)

    return tf.expand_dims(grid, axis=0)


def setup_path(path: str, optional_join: str = None):
    path = path if path.endswith('/') else path + '/'
    if optional_join:
        path = os.path.join(path, optional_join)
    os.makedirs(path, exist_ok=True)
    return path


def transfer_method(method_name: str, from_A, to_B):
    method = getattr(from_A, method_name)
    setattr(to_B, method_name, types.MethodType(method, to_B))


def remainder_is_0(counter, frequency):
    try:
        return counter % frequency == 0
    except ZeroDivisionError:
        return False


if __name__ == "__main__":
    # ones = tf.random.uniform((64, 28, 28, 3))
    # save_images(ones, "pics/", "hihi")
    # print("done")
    from data import DataGenerator

    data = DataGenerator("./preprocessing/data128/", images_in_test_split=20)
    d = data.validation_generator.next()
    d = to_grid(tf.stack([d, d], 0), 20)
    save_images(d, "./test/")
