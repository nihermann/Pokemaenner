from tensorflow import keras
import tensorflow as tf


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=6, latent_dim=128, starter_count=0, img_path="results_all/generated_img_{epoch}_{i}.png",
                 model=None, model_path="models/wgan_model", save_model_every=3):
        """
        Initialization of a Monitor for the gan managing what to do after an epoch (save model, pictures etc.)
        :param num_img = how many images to save per epoch
        :param latent_dim = size of latent space
        :param starter_count = number of epoch of last training session (where training was left off)
        :param img_path = path for saving the resulting images
        :param model = current model to save weights
        :param model_path = path for saving the model
        :param save_model_every = how frequently the models weights should be saved
        """
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.starter_count = starter_count
        self.img_path = img_path
        self.model = model
        self.model_path = model_path
        self.save_model_every = save_model_every

    def on_epoch_end(self, epoch, logs=None):
        """
        :param epoch = the current epoch as prefix for the file name
        """
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        if epoch % self.save_model_every == 0:
            self.model.save_weights(self.model_path, save_format='tf')
            print("Model was saved.")

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.preprocessing.image.array_to_img(img)
            img.save(self.img_path.format(epoch=(epoch + self.starter_count), i=i))
