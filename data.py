import tensorflow as tf
import os


class DataGenerator:
    """
    class for generating image data and splitting into dataset. The dataset is supposed to be structured like so:
    - Pokemaenner:
        - images:
            - class_label/s (only one for generative adversial network)
                xxx.png
    """

    def __init__(self, img_path="images", batch_size=32, img_height=256, img_width=256, validation_split=0.1,
                 shuffle=True):
        """

        :param img_path: name of the image folder (image folder has to be in the same folder as the excecuting class file)
        :param batch_size: int - default 32
        :param img_height: int - default 256 (changes the image size if it doesn't fit)
        :param img_width: int - default 256 (changes the image size if it doesn't fit)
        :param validation_split: float - procentual number to take from the original dataset and make it into a validation one (default 0.1)
        :param shuffle: bool - whether to shuffle or not
        """
        self.img_path = img_path
        self.batch_size = batch_size
        self.img_width = img_width
        self.img_height = img_height
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.trainings_data = self.generate_data()
        self.validation_data = self.generate_data("validation")

    def generate_data(self, split_name="training"):
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(os.getcwd(), self.img_path),
            labels='inferred',
            label_mode="categorical",  # can also be int, binary etc.
            color_mode='rgba',
            batch_size=self.batch_size,
            image_size=(self.img_height, self.img_width),  # reshape if not in the wanted size
            shuffle=self.shuffle,
            validation_split=self.validation_split,
            subset=split_name,
            seed=123
        )
        # maybe add tf.data.Dataset.from_tensors(ds) to make it a dataset the current output type is a BatchDataset
        return ds.map(self.augment)

    def augment(self, image, label):
        # add here what ever data augmentation you like
        image = tf.image.random_brightness(image, max_delta=0.05)
        # image = tf.image.flip_left_right(image)
        # image = tf.image.random_contrast(image)
        return image, label


if __name__ == "__main__":
    data = DataGenerator("images")

