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
    """
    Small helper function which allows to store more complex default value and thus not making the methods default values huge and unreadable.
    :param default: The 'default' value, which will be returned if the 'alternative' is None.
    :param alternative: An 'alternative' value for the 'default' value.
    :return: Returns the 'default' value if the 'alternative' is None, else the alternative.
    """
    return default if alternative is None else alternative


def get_padded_cols(images, border):
    """
    Helper function which pads each tensor image and afterwards stacking it to four columns.
    The length of the column is dependent on the number of images provided.
    :param images: tensor with images of shape (B, H, W, C) where B is divisible by four.
    :param border: int - specifying the border width around the images.
    :return: tensor of shape (4, (H + 2*border) * B/4, W + 2*border, C) containing four columns of padded images.
    """
    padding = tf.constant([[0, 0], [border, border], [border, border], [0, 0]])
    images = tf.pad(images, padding, mode="constant", constant_values=0)
    cols = tf.reshape(images, shape=(4, -1, images.shape[1], 3))
    return cols


def stack_alternating(a, b):
    """
    Takes in two tensors with common shape and alternates the those in their first dimension.
    :param a: Tensor A - with ndim >= 3.
    :param b: Tensor B - with ndim >= 3.
    :return: Tensor with doubled length in the first dimension compared to A/B.
    """
    stack = tf.stack([a, b], axis=1)  # shape: (B, 2, H, W, C)
    return tf.reshape(stack, (stack.shape[0] * stack.shape[1], *stack.shape[2:]))  # shape: (2B, H, W, C)


def to_grid(images, border):
    """
    Projects a tensor of images into a grid with four columns if ndim == 4.
    If there is a fifth dim holding two whole batches, each batch will be placed next to the ones of the other batch.
    Like this a comparison of the two batches looks more natural and column size is eight.
    :param images: tensor with images of one of the following shapes: (2, B, H, W, C) || (B, H, W, C) where B is divisible by four.
    :param border: int - specifying the amount of padding around each image for the grid.
    :return: tensor containing a grid with all the images.
    """
    if images.ndim == 5:  # check if there are two batches for comparison.
        # pad both and get their cols.
        col1 = get_padded_cols(images[0], border)
        col2 = get_padded_cols(images[1], border)
        # alternate both cols
        cols = stack_alternating(col1, col2)
    else:
        # pad them and get them col wise.
        cols = get_padded_cols(images, border)

    # Building the grid.
    grid = cols[0]  # first col as the starting point.
    for col in cols[1:]:
        # concat all cols together.
        grid = tf.concat([grid, col], axis=1)

    return tf.expand_dims(grid, axis=0)  # return them with batch size 1.


def setup_path(path: str, optional_join: str = None):
    """
    Preprocessing of a path string. It adds the '/' if not already there and allows an optional join for subdirs.
    Furthermore it creates all the dirs if they not already exist.
    :param path: str - specifying the path.
    :param optional_join: a subdir which will be joined to the path string.
    :return: preprocessed path string.
    """
    path = path if path.endswith('/') else path + '/'
    if optional_join:
        path = os.path.join(path, optional_join)
    os.makedirs(path, exist_ok=True)
    return path


def transfer_method(method_name: str, from_A, to_B):
    """
    This function extracts a bound function (method) from A and transfers + bounds it to B.
    :param method_name: str - the name of the method to be transferred.
    :param from_A: class holding the method.
    :param to_B: class instance where the method should be bound to.
    :return: Nothing.
    """
    method = getattr(from_A, method_name)  # extract method
    setattr(to_B, method_name, types.MethodType(method, to_B))  # rebind it.


def remainder_is_0(counter, frequency):
    """
    Checks if the remainder of counter % frequency is 0.
    Instead of raising a ZeroDivisionError if frequency is 0, this function will just return False.
    This gives the opportunity to disable timed events.
    :param counter: int - usually the epoch or counter.
    :param frequency: int - the frequency of when to return True.
    :return: bool - True if counter % frequency is 0 | False if not or frequency is 0.
    """
    try:
        return counter % frequency == 0
    except ZeroDivisionError:
        return False


if __name__ == "__main__":
    # ones = tf.random.uniform((64, 28, 28, 3))
    # save_images(ones, "pics/", "hihi")
    # print("done")
    from aegan_data import DataGenerator

    data = DataGenerator("./preprocessing/data128/", images_in_validation_split=20)
    d = data.validation_generator.next()
    d = to_grid(tf.stack([d, d], 0), 20)
    save_images(d, "./test/")
