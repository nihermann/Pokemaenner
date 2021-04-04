import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from aegan_utils import setup_path


class DataGenerator:
    def __init__(self, img_path="images", batch_size=32, img_size=(64, 64), images_in_validation_split=0, horizontal_flip=True, shuffle=False):
        """
        Container class for a training generator and a validation generator for images.
        :param img_path: name of the image folder (image folder has to be in the same folder as the excecuting class file)
        :param batch_size: int
        :param img_size: tuple - (height, width) of the images.
        :param images_in_validation_split: int - how many pictures should be in the validation split.
        :param horizontal_flip: bool - whether to apply horizontal flip or not.
        :param shuffle: bool - whether the dataset should be shuffled or not.
        """
        self.img_path = setup_path(img_path)
        self.batch_size = batch_size
        self.img_size = img_size

        # validation split
        self.images_in_test_split = images_in_validation_split
        self.n = len(os.listdir(self.img_path+"data"))
        self.validation_split = images_in_validation_split / self.n

        # data augmentation
        self.horizontal_flip = horizontal_flip
        self.shuffle = shuffle

        self.training_generator = self._get_image_generator(training_subset=True)
        self.validation_generator = self._get_image_generator(training_subset=False)

    def _get_image_generator(self, training_subset=False):
        """Returns a Data generator for training or validation."""
        return ImageDataGenerator(
            preprocessing_function=lambda img: tf.cast(2*(img/255)-1, tf.float32),
            validation_split=(self.n % self.batch_size)/self.n if training_subset else self.validation_split,
            horizontal_flip=self.horizontal_flip and training_subset  # only augment if its the trainings subset
        ).flow_from_directory(
            directory=self.img_path,
            target_size=self.img_size,
            class_mode=None,
            batch_size=self.batch_size if training_subset else self.images_in_test_split,
            shuffle=self.shuffle and training_subset,
            subset="training" if training_subset else "validation"
        )



