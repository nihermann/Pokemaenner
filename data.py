import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from utils import setup_path


class DataGenerator:
    """
    class for generating image data and splitting into dataset. The dataset is supposed to be structured like so:
    - Pokemaenner:
        - images:
            - class_label/s (only one for generative adversial network)
                xxx.png
    """

    def __init__(self, img_path="images", batch_size=32, img_size=(64, 64), images_in_test_split=0, horizontal_flip=True, shuffle=False):
        """

        :param img_path: name of the image folder (image folder has to be in the same folder as the excecuting class file)
        :param batch_size: int - default 32
        :param img_height: int - default 256 (changes the image size if it doesn't fit)
        :param img_width: int - default 256 (changes the image size if it doesn't fit)
        :param validation_split: float - percentual number to take from the original dataset and make it into a validation one (default 0.1)
        """
        self.img_path = setup_path(img_path)
        self.batch_size = batch_size
        self.img_size = img_size
        self.images_in_test_split = images_in_test_split
        self.n = len(os.listdir(self.img_path+"data"))
        self.validation_split = images_in_test_split/self.n
        self.horizontal_flip = horizontal_flip
        self.shuffle = shuffle

        self.training_generator = self._get_image_generator(training_subset=True)
        self.validation_generator = self._get_image_generator(training_subset=False)

    def _get_image_generator(self, training_subset=False):
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

    def generate_data(self, split_name="training"):
        ds = tf.keras.preprocessing.image_dataset_from_directory(
            os.path.join(os.getcwd(), self.img_path),
            labels='inferred',  # inters the labels from the given directories
            label_mode="int",  # can also be int, binary etc.
            color_mode='rgb',
            batch_size=self.batch_size,
            image_size=self.img_size,  # reshape if not in the wanted size
            shuffle=False,
            validation_split=self.validation_split,  # split the dataset for validation and training
            subset=split_name,
            seed=123
        )
        # applies augmentation to the images in the dataset normalisation, random_brightness usw.)
        ds = ds.map(self.augment)
        # caching to same the augmented images to save time in the next step
        # ds = ds.cache()
        # # shuffles the dataset (done after caching so that the dataset doesn't freeze and just the same data is returned
        # ds = ds.shuffle(buffer_size=self.batch_size*10)
        # prefetch data so that if you forward_step in one training step you can already apply augmentation for the next
        # time step
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        # maybe add tf.data.Dataset.from_tensors(ds) to make it a dataset the current output type is a BatchDataset
        return ds

    def augment(self, image, label):
        # add here what ever data augmentation you like
        image = tf.image.random_brightness(image, max_delta=0.05)
        # image = tf.image.flip_left_right(image)
        image = tf.image.random_contrast(image, 0.2, 0.5)
        # image = tf.image.per_image_standardization(image)
        image = tf.cast(image / 255., tf.float32)
        return image, label

    # def visualization(self):
    #     for images_folder in self.trainings_data:
    #         for images in images_folder:
    #             for image in images_folder:
    #                 plt.axis("off")
    #                 plt.imshow((image.numpy() * 255).astype("int32")[0])
    #                 # print(image)
    #                 break


if __name__ == "__main__":
    from utils import save_images
    data = DataGenerator("./preprocessing/data64", batch_size=32*8, images_in_test_split=20, shuffle=True)
    print(type(data.validation_generator))
    for i, img in enumerate(data.training_generator):
        if img.shape[0] != 256:
            print("Fail")
        if i == 30000:
            break


