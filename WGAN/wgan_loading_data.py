import numpy as np
import tensorflow as tf


def image_data_to_numpy(IMG_DATA_PATH="preprocessing/data_all", IMG_SHAPE=(64, 64, 3),
                        IMG_SAVE_PATH="data_reshaped_as_array/images",
                        LBL_SAVE_PATH='data_reshaped_as_array/head_labels'):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        IMG_DATA_PATH,
        labels="inferred",
        label_mode="int",
        batch_size=32,
        image_size=(64, 64),
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        interpolation="bilinear",
        follow_links=False,
    )
    # transform the tensor dataset into more conventional numpy array
    labels = []
    images = []
    # iterate through the dataset
    for image, label in dataset:
        for i in range(len(image)):
            images.append(image[i])
            labels.append(label[i])

    images = np.array(images)
    images = images.reshape(images.shape[0], IMG_SHAPE[0], IMG_SHAPE[1], IMG_SHAPE[2])
    labels = np.array(labels)
    labels = labels.reshape(labels.shape[0], )

    # save the numpy array datasets
    np.save(IMG_SAVE_PATH, images)
    np.save(LBL_SAVE_PATH, labels)


def loading_data(IMG_DATA_PATH="preprocessing/data", IMG_SHAPE=(64, 64, 3),
                 IMG_SAVE_PATH="data_reshaped_as_array/images", LBL_SAVE_PATH='data_reshaped_as_array/head_labels'):
    # convert the image data into numpy arrays
    image_data_to_numpy(IMG_DATA_PATH, IMG_SHAPE, IMG_SAVE_PATH, LBL_SAVE_PATH)

    # load the numpy datasets defining image data and their label
    images = np.load(IMG_SAVE_PATH+".npy")
    labels = np.load(LBL_SAVE_PATH+".npy")

    # Reshape each sample to (28, 28, 1) and normalize the pixel values in the [-1, 1] range
    images = images.reshape(images.shape[0], *IMG_SHAPE).astype("float32")
    images = (images - 127.5) / 127.5

    return images, labels
