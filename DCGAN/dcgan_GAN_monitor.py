from tensorflow import keras
import tensorflow as tf


class GANMonitor(keras.callbacks.Callback):
    """
    Manages what to do with the GAN and it's result after each epoch
    (make a checkpoint and save the generated images.
    """

    def __init__(self, num_img=3, latent_dim=128,
                 checkpoint_path="/content/drive/MyDrive/pokemaenner/model/training_1/cp.ckpt",
                 IMG_PATH="results/generated_img_{epoch}_{i}.png", epoch_counter_begin=0):
        """
        :param num_img = how many images to save
        :param latent_dim = size of the latent space
        :param checkpoint_path = path for saving the epoch checkpoint
        :param epoch_counter_begin = where the training was left off (to save the images under the correct file name)
        """
        super(GANMonitor, self).__init__()
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.checkpoint_path = checkpoint_path
        self.epoch_counter_begin = epoch_counter_begin
        self.img_path = IMG_PATH

    def on_epoch_end(self, epoch, logs=None):
        # make Checkpoint
        tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                           save_weights_only=True,
                                           verbose=1)
        # generate images from random noise with the generator
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        # denormalize the images to get the full color coding
        generated_images *= 255
        # as numpy
        generated_images.numpy()
        for i in range(self.num_img):
            # save images
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save(self.img_path.format(epoch=(epoch + self.epoch_counter_begin), i=i))
