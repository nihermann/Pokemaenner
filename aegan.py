import tensorflow as tf
from tensorflow.keras.metrics import Mean
import os
import models
from utils import (to_grid, save_images, setup_path, transfer_method, remainder_is_0)
from contextlib import suppress
import re


class AEGAN(tf.keras.Model):
    def __init__(
            self,
            image_shape,
            latentspace,
            batch_size,
            noise_generating_fn=None,
            continue_from_saved_models=False,
            path="./models"
    ):
        """
        This AEGAN class in itself is an empty keras Model but it holds five functional models as attributes:
        *Important* - In each training step the AEGAN needs eight batches of images, so make sure to supply an eight times bigger batch than you intended to use in training.
        Generator(Decoder): Is used in two ways. (1) to generate images and (2) to be part of an autoencoder.
        Encoder: Together with the Generator it forms an autoencoder.
        Image Discriminator: Which is trained to predict the realness of a given image.
        Latent Discriminator: Which is used measure judge the realness of a given latent vector.
        AEGAN: The AEGAN is composed of the a.m. although the two Discriminators are decoupled and are constant throughout the AEGAN training.

        :param image_shape: tuple - specifying the shape of the images w/o the batch size dimension. It has to match the supplied images.
        :param latentspace: int - non negative int specifying the dimensionality of the embedding.
        :param batch_size: int - non negative int which has to be eight times bigger than you want in to be during training.
        :param noise_generating_fn: function - which has one argument specifying the number of the batch size dimension d and returns a noise vector with shape (d, latentspace).
        :param continue_from_saved_models: bool - if True it will search in 'path' for weights for every model and start training from this state on.
        :param path: str - where the weights can be found.
        """
        super(AEGAN, self).__init__()
        self.compile()  # empty compilation
        self._initialize_metrics()  # initializes all metrics which are needed

        assert batch_size % 8 == 0, "batch size needs to be divisible by 8 with no remainder."

        self.batch_size = batch_size // 8
        self.noise_generating_fn = noise_generating_fn

        self.initial_epoch = 0
        message = "Build models: [{}]"
        print(message.format("...."), end="\r")
        self.generator = models.Decoder(
            latentspace,
            first_reshape_shape=[4, 4, 64],
            hidden_activation="relu",
            output_activation="tanh",
            name="generator"
        )
        print(message.format(">..."), end="\r")

        self.encoder = models.Encoder(
            image_shape,
            latentspace,
            hidden_activation="relu",
            output_activation="linear",
            name="encoder"
        )
        print(message.format("=>.."), end="\r")

        self.discriminator_image = models.Encoder(
            image_shape,
            latentspace=1,
            kernel_widths=[4, 4, 4, 4, 4, 4],
            hidden_activation="leaky_relu",
            output_activation="sigmoid",
            name="image_discriminator"
        )
        print(message.format("==>."), end="\r")

        self.discriminator_latent = models.DiscriminatorLatent(
            latentspace,
            num_blocks=16,
            neurons_per_layer=16,
            hidden_activation="relu",
            output_activation="sigmoid",
            name="latent_discriminator"
        )
        print(message.format("===>"), end="\r")

        self.aegan = self._build_aegan()
        print(message.format("===="))

        if continue_from_saved_models:
            self.try_load_all_weights(path)

        self._compile()
        print("Build successfully")

    def _initialize_metrics(self):
        self.dis_image_loss = Mean(name="dis_image_loss")
        self.dis_latent_loss = Mean(name="dis_latent_loss")
        self.aegan_loss = Mean(name="aegan_loss")

    @property
    def metrics(self):
        return [self.dis_image_loss, self.dis_latent_loss, self.aegan_loss]

    def try_load_all_weights(self, path):
        files = os.listdir(path)
        files = [f for f in files if f.endswith(".h5") or f.endswith(".tf")]
        print("<FollowingWeightsFound> ::: ", end="")

        num_loaded = 0
        found_g, found_e, found_l, found_i = False, False, False, False

        for model in reversed(sorted(files, key=len)):
            if not found_g and "generator" in model:
                found_g, num_loaded = True, num_loaded + 1
                print(model, end=" ::: ")
                self.generator.load_weights(os.path.join(path, model))
                self.initial_epoch = int(re.findall(r'\d+', model)[0])

            elif not found_e and "encoder" in model:
                found_e, num_loaded = True, num_loaded + 1
                print(model, end=" ::: ")
                self.encoder.load_weights(os.path.join(path, model))

            elif not found_l and "discriminator_latent" in model:
                found_l, num_loaded = True, num_loaded + 1
                print(model, end=" ::: ")
                self.discriminator_latent.load_weights(os.path.join(path, model))

            elif not found_i and "discriminator_image" in model:
                found_i, num_loaded = True, num_loaded + 1
                print(model, end=" ::: ")
                self.discriminator_image.load_weights(os.path.join(path, model))

        if num_loaded != 4:
            print("\nLoading models was unsuccessful... new models are created...")
        else:
            print(f"\nLoading was successful! Continuing in epoch {self.initial_epoch} with loaded weights...\n")
        return num_loaded == 4

    def _build_aegan(self):
        self.discriminator_image.trainable = False
        self.discriminator_latent.trainable = False

        input_image_shape = self.encoder.input_shape[1:]
        try:
            input_latent_shape = self.generator.input_shape[1:]
        except AttributeError:
            print("Caught AttributeError while extracting the input shape")
            input_latent_shape = self.noise_generating_fn(1).shape[1:]

        # image path
        real_img = tf.keras.layers.Input(input_image_shape, name="image_input")
        embedded_real_img = self.encoder(real_img)

        prediction_real_embedding = self.discriminator_latent(embedded_real_img)
        reconstructed_real_img = self.generator(embedded_real_img)

        prediction_reconstructed_img = self.discriminator_image(reconstructed_real_img)

        # latent path
        real_z = tf.keras.layers.Input(input_latent_shape, name="latentspace_input")
        generated_img = self.generator(real_z)

        prediction_fake_img = self.discriminator_image(generated_img)
        reconstructed_latent_vector = self.encoder(generated_img)

        prediction_reconstructed_embedding = self.discriminator_latent(reconstructed_latent_vector)

        aegan = tf.keras.Model([real_img, real_z], [reconstructed_real_img,
                                                    reconstructed_latent_vector,
                                                    prediction_fake_img,
                                                    prediction_reconstructed_img,
                                                    prediction_real_embedding,
                                                    prediction_reconstructed_embedding], name="AEGAN")
        self.discriminator_latent.trainable = True
        self.discriminator_image.trainable = True
        return aegan

    def _compile(self):
        self.discriminator_latent.compile(
            tf.keras.optimizers.Adam(0.0005, beta_1=0.5, clipnorm=1),
            loss="binary_crossentropy"
        )
        self.discriminator_image.compile(
            tf.keras.optimizers.Adam(0.0005, beta_1=0.5, clipnorm=1),
            loss="binary_crossentropy"
        )
        self.aegan.compile(
            tf.keras.optimizers.Adam(0.0002, beta_1=0.5, clipnorm=1),
            loss=[
                "mae", "mse",
                "binary_crossentropy",
                "binary_crossentropy",
                "binary_crossentropy",
                "binary_crossentropy"
            ],
            loss_weights=[10, 5, 1, 1, 1, 1]
        )

    def train_step(self, real_image_batches):
        data1, data2, data3, data4, data5, data6, data7, data8 = tf.split(real_image_batches, 8, axis=0)

        real_labels_d = tf.ones((self.batch_size, 1)) * 0.95
        fake_labels_d = tf.ones((self.batch_size, 1)) * 0.05

        dis_image_loss = 0
        dis_image_loss += self.discriminator_image.forward_step(
            data1,
            real_labels_d
        )

        generated_images = self.generate_images(self.batch_size)
        dis_image_loss += self.discriminator_image.forward_step(
            generated_images,
            fake_labels_d
        )

        dis_image_loss += self.discriminator_image.forward_step(
            data2,
            real_labels_d
        )

        reconstructed_images = self.autoencode_images(data3)
        dis_image_loss += self.discriminator_image.forward_step(
            reconstructed_images,
            fake_labels_d
        )
        self.dis_image_loss.update_state(dis_image_loss / 4)

        del generated_images, reconstructed_images, data1, data2, data3

        dis_latent_loss = 0
        dis_latent_loss += self.discriminator_latent.forward_step(
            self.noise_generating_fn(self.batch_size),
            real_labels_d
        )

        embedded_image = self.encode(data4)
        dis_latent_loss += self.discriminator_latent.forward_step(
            embedded_image,
            fake_labels_d
        )

        dis_latent_loss += self.discriminator_latent.forward_step(
            self.noise_generating_fn(self.batch_size),
            real_labels_d
        )

        reconstructed_embedding = self.autoencode_latent(self.noise_generating_fn(self.batch_size))
        dis_latent_loss += self.discriminator_latent.forward_step(
            reconstructed_embedding,
            fake_labels_d
        )
        self.dis_latent_loss.update_state(dis_latent_loss / 4)

        del data4, embedded_image, reconstructed_embedding, real_labels_d, fake_labels_d

        labels_g = tf.ones((self.batch_size, 1))
        for data in [data5, data6, data7, data8]:
            latent = self.noise_generating_fn(self.batch_size)

            with tf.GradientTape() as tape:
                all_preds_and_reconstructions = self.aegan([data, latent])
                loss = self.aegan.compiled_loss(
                    [data, latent, labels_g, labels_g, labels_g, labels_g],
                    all_preds_and_reconstructions
                )

            gradients = tape.gradient(loss, self.aegan.trainable_variables)
            self.aegan.optimizer.apply_gradients(zip(gradients, self.aegan.trainable_variables))
            self.aegan_loss.update_state(loss)

        return {m.name: m.result() for m in self.metrics}

    def autoencode_images(self, image):
        encoding = self.encoder(image)
        return self.generator(encoding)

    def autoencode_latent(self, latent_vector):
        image = self.generator(latent_vector)
        return self.encoder(image)

    def encode(self, image):
        return self.encoder(image)

    def generate_images(self, num):
        noise = self.noise_generating_fn(num)
        return self.generator(noise)

    @tf.function
    def call(self, x, training=True):
        data1, _, _, _, _, _, _, _ = tf.split(x, 8, axis=0)
        latent = self.noise_generating_fn(self.batch_size)
        return self.aegan([data1, latent])

    def save_weights(self,
                     filepath,
                     overwrite=True,
                     save_format=None,
                     options=None):
        self.encoder.save(filepath.format("encoder"), overwrite, save_format, options)

        self.generator.save(filepath.format("generator"), overwrite, save_format, options)

        self.discriminator_image.save(filepath.format("discriminator_image"),
                                      overwrite, save_format, options)

        self.discriminator_latent.save(filepath.format("discriminator_latent"),
                                       overwrite, save_format, options)
        print("All Weights Saved")


class SaveAegan(tf.keras.callbacks.Callback):
    def __init__(self, save_images_every: int, save_model_every: int, save_path: str, data_gen,
                 tensorboard_logdir=None):
        super(SaveAegan, self).__init__()
        self.epoch = 0
        self.save_images_every = save_images_every
        self.image_save_path = setup_path(save_path, optional_join="images/")

        self.save_model_every = save_model_every
        self.model_save_path = setup_path(save_path, optional_join="models/")

        self.get_test_images = lambda: data_gen.next()

        self.tbw_generated_imgs = tf.summary.create_file_writer(
            tensorboard_logdir + '/generated_images') if tensorboard_logdir is not None else None
        self.tbw_reconstructed_imgs = tf.summary.create_file_writer(
            tensorboard_logdir + '/reconstructed_images') if tensorboard_logdir is not None else None

    def save(self, images, prefix):
        grid = to_grid(images, 10)
        save_images(grid, save_to=self.image_save_path, prefix=prefix)
        return grid

    def make_and_write_images(self, epoch, logs, suffix=""):
        imgs = self.get_test_images()
        generated_imgs = self.model.generate_images(imgs.shape[0]*2)
        gen_grid = self.save(generated_imgs, f"{epoch}generated_loss{logs['aegan_loss']: .4f}{suffix}_")

        reconstructed_imgs = self.model.autoencode_images(imgs)
        reconstruction_loss = tf.keras.metrics.mean_absolute_error(imgs, reconstructed_imgs)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        combined = tf.stack([imgs, reconstructed_imgs], axis=0)
        rec_grid = self.save(combined, f"{epoch}reconstructed_loss{reconstruction_loss: .4f}{suffix}_")

        if self.tbw_reconstructed_imgs is not None:
            with self.tbw_generated_imgs.as_default():
                tf.summary.image("Generated Images", gen_grid, step=epoch)
            with self.tbw_reconstructed_imgs.as_default():
                tf.summary.image("Reconstructed Images", rec_grid, step=epoch)

    def save_submodels(self, epoch, suffix=""):
        self.model.save_weights(
            filepath=self.model_save_path + str(epoch) + "_{}" + suffix + ".h5",
            save_format="h5"
        )

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch + 1

        if remainder_is_0(self.epoch, self.save_images_every):
            self.make_and_write_images(self.epoch, logs)

        if remainder_is_0(self.epoch, self.save_model_every):
            self.save_submodels(self.epoch)

    def on_train_end(self, logs=None):
        if not remainder_is_0(self.epoch, self.save_images_every):
            self.make_and_write_images(self.epoch, logs, suffix="_F")

        if not remainder_is_0(self.epoch, self.save_model_every):
            self.save_submodels(self.epoch, suffix="_F")


if __name__ == '__main__':
    aegan = AEGAN((64, 64, 3), 10, 16 * 8, lambda b: tf.random.normal((b, 10)), False)
    tb = tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq="batch")
    aegan.fit(tf.ones((16 * 8 * 210, 64, 64, 3)), batch_size=16 * 8, epochs=2, callbacks=[tb])
