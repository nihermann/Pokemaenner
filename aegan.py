import tensorflow as tf
from tensorflow.keras.metrics import Mean
import os
import models
from utils import (to_grid, save_images, setup_path, transfer_method, remainder_is_0)
import re


class AEGAN(tf.keras.Model):
    def __init__(
            self,
            image_shape,
            latentspace,
            batch_size,
            noise_generating_fn=None,
            continue_from_saved_models=False,
            only_weights=True,
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

        loading_successful = False
        if continue_from_saved_models and not only_weights:
            loading_successful = self._try_load_models(path, only_weights)

        message = "Build models: [{}]"
        if not loading_successful:
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
                kernel_widths=[3, 3, 3, 3, 3, 3],
                hidden_activation="leaky_relu",
                output_activation="sigmoid",
                name="image_discriminator"
            )
            print(message.format("==>."), end="\r")

            self.discriminator_latent = models.DiscriminatorLatent(
                latentspace,
                num_blocks=16,
                neurons_per_layer=16,
                hidden_activation="leaky_relu",
                output_activation="sigmoid",
                name="latent_discriminator"
            )
            print(message.format("===>"), end="\r")

        self.aegan = self._build_aegan()
        print(message.format("===="))

        if continue_from_saved_models and only_weights:
            self._try_load_models(path, only_weights)

        self._compile()
        print("Build successfully")
        # self.summarize_to_file()

    def summarize_to_file(self):
        """Writes .summary and the graph structures to file."""
        with open('generator.txt', 'w') as f:
            self.generator.summary(print_fn=lambda x: f.write(x + '\n'))
        with open('encoder.txt', 'w') as f:
            self.encoder.summary(print_fn=lambda x: f.write(x + '\n'))
        with open('discriminator_image.txt', 'w') as f:
            self.discriminator_image.summary(print_fn=lambda x: f.write(x + '\n'))
        with open('discriminator_latent.txt', 'w') as f:
            self.discriminator_latent.summary(print_fn=lambda x: f.write(x + '\n'))

        tf.keras.utils.plot_model(self.generator, "generator.png", show_shapes=True)
        tf.keras.utils.plot_model(self.encoder, "encoder.png", show_shapes=True)
        tf.keras.utils.plot_model(self.discriminator_image, "discriminator_image.png", show_shapes=True)
        tf.keras.utils.plot_model(self.discriminator_latent, "discriminator_latent.png", show_shapes=True)

    def _initialize_metrics(self):
        """
        Initializes the necessary metrics as attributes for easier and cleaner usage later.
        """
        self.dis_image_loss = Mean(name="dis_image_loss")
        self.dis_latent_loss = Mean(name="dis_latent_loss")
        self.aegan_loss = Mean(name="aegan_loss")

    @property
    def metrics(self):
        """
        Overwritten property form super class.
        :return: list of all metrics of this model.
        """
        return [self.dis_image_loss, self.dis_latent_loss, self.aegan_loss]

    def _try_load_models(self, path: str, only_weights: bool):
        """
        Tries to load a generator, encoder, discriminator_image and discriminator_latent from path.
        :param path: str - path where the models/ weights are stored.
        :param only_weights: bool - If True we try to load weights, else models.
        :return: bool - stating whether the loading was successful or not.
        """
        # extract all models/ weights from the specified path.
        files = os.listdir(path)
        files = [f for f in files if f.endswith(".h5") or f.endswith(".tf")]

        models_or_weights = 'Weights' if only_weights else 'Models'
        print(f"<Following{models_or_weights}Found> ::: ", end="")

        # progress tracker
        num_loaded = 0
        found_g, found_e, found_l, found_i = False, False, False, False

        for model in reversed(sorted(files, key=len)):  # iterate from highest to lowest epoch
            if not found_g and "generator" in model:
                found_g, num_loaded = True, num_loaded + 1
                print(model, end=" ::: ")

                if only_weights:
                    self.generator.load_weights(os.path.join(path, model))
                else:
                    self.generator = tf.keras.models.load_model(os.path.join(path, model), compile=False)

                # extract the epoch from the model name to be able to start the training from this epoch. Makes sure
                # that after pausing training it can be continued from there without any further actions necessary.
                self.initial_epoch = int(re.findall(r'\d+', model)[0])

            elif not found_e and "encoder" in model:
                found_e, num_loaded = True, num_loaded + 1
                print(model, end=" ::: ")

                if only_weights:
                    self.encoder.load_weights(os.path.join(path, model))
                else:
                    self.encoder = tf.keras.models.load_model(os.path.join(path, model), compile=False)
                    transfer_method("forward_step", models.Encoder, self.encoder)

            elif not found_l and "discriminator_latent" in model:
                found_l, num_loaded = True, num_loaded + 1
                print(model, end=" ::: ")

                if only_weights:
                    self.discriminator_latent.load_weights(os.path.join(path, model))
                else:
                    self.discriminator_latent = tf.keras.models.load_model(os.path.join(path, model), compile=False)
                    transfer_method("forward_step", models.DiscriminatorLatent, self.discriminator_latent)

            elif not found_i and "discriminator_image" in model:
                found_i, num_loaded = True, num_loaded + 1
                print(model, end=" ::: ")

                if only_weights:
                    self.discriminator_image.load_weights(os.path.join(path, model))
                else:
                    self.discriminator_image = tf.keras.models.load_model(os.path.join(path, model), compile=False)
                    transfer_method("forward_step", models.Encoder, self.discriminator_image)

        if num_loaded != 4:
            print(f"\nLoading {models_or_weights} was unsuccessful...",
                  "" if only_weights else "new models are created...")
        else:
            print(f"\nLoading was successful! Continuing in epoch {self.initial_epoch} with loaded {models_or_weights}...\n")
        return num_loaded == 4

    def _build_aegan(self):
        self.discriminator_image.trainable = False
        self.discriminator_latent.trainable = False

        input_image_shape = self.encoder.input_shape[1:]
        input_latent_shape = self.generator.input_shape[1:]

        # image path
        real_img = tf.keras.layers.Input(input_image_shape, name="image_input")
        embedded_real_img = self.encoder(real_img)

        prediction_real_embedding = self.discriminator_latent(embedded_real_img, training=False)
        reconstructed_real_img = self.generator(embedded_real_img)

        prediction_reconstructed_img = self.discriminator_image(reconstructed_real_img, training=False)

        # latent path
        real_z = tf.keras.layers.Input(input_latent_shape, name="latentspace_input")
        generated_img = self.generator(real_z)

        prediction_fake_img = self.discriminator_image(generated_img, training=False)
        reconstructed_latent_vector = self.encoder(generated_img)

        prediction_reconstructed_embedding = self.discriminator_latent(reconstructed_latent_vector, training=False)

        return tf.keras.Model([real_img, real_z], [reconstructed_real_img,
                                                   reconstructed_latent_vector,
                                                   prediction_fake_img,
                                                   prediction_reconstructed_img,
                                                   prediction_real_embedding,
                                                   prediction_reconstructed_embedding], name="AEGAN")

    def _compile(self):
        """Compile all Models"""
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
        """One forward step - called automatically from .fit()"""
        data1, data2, data3, data4, data5, data6, data7, data8 = tf.split(real_image_batches, 8, axis=0)

        # Discriminator phase
        self.discriminator_latent.trainable = True
        self.discriminator_image.trainable = True

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

        # Generator phase
        self.discriminator_latent.trainable = False
        self.discriminator_image.trainable = False

        # the labels are all "real" because we want our AEGAN to be able to fool the discriminators.
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

    def autoencode_images(self, image, training=False):
        """Autoencodes/Reconstructs an image"""
        encoding = self.encoder(image, training=training)
        return self.generator(encoding, training=training)

    def autoencode_latent(self, latent_vector, training=False):
        """Autoencodes a latent vector."""
        image = self.generator(latent_vector, training=training)
        return self.encoder(image, training=training)

    def encode(self, image, training=False):
        """Encodes the input image"""
        return self.encoder(image, training=training)

    def generate_images(self, num, training=False):
        """
        Generates images from the generator
        :param num: int - how many images should be generated.
        :param training: bool - whether the generator should be in train mode or not.
        :return: the generated images.
        """
        noise = self.noise_generating_fn(num)
        return self.generator(noise, training=training)

    @tf.function
    def call(self, x, training=True):
        data1, _, _, _, _, _, _, _ = tf.split(x, 8, axis=0)
        latent = self.noise_generating_fn(self.batch_size)
        return self.aegan([data1, latent], training=training)

    def save_weights(self, filepath, save_format=None, **kwargs):
        """
        Saves all weights to file.
        """
        self.encoder.save_weights(filepath.format("encoder"), save_format=save_format, **kwargs)

        self.generator.save_weights(filepath.format("generator"), save_format=save_format, **kwargs)

        self.discriminator_image.save_weights(filepath.format("discriminator_image"),
                                              save_format=save_format, **kwargs)

        self.discriminator_latent.save_weights(filepath.format("discriminator_latent"),
                                               save_format=save_format, **kwargs)
        print("All Weights Saved!")

    def save(self, filepath, include_optimizer=False, save_format=None, **kwargs):
        """
        Saves all Models to file.
        """
        self.encoder.save(filepath.format("encoder"), save_format=save_format, include_optimizer=False, **kwargs)

        self.generator.save(filepath.format("generator"), save_format=save_format, include_optimizer=False, **kwargs)

        self.discriminator_image.save(filepath.format("discriminator_image"), save_format=save_format,
                                      include_optimizer=include_optimizer, **kwargs)

        self.discriminator_latent.save(filepath.format("discriminator_latent"), save_format=save_format,
                                       include_optimizer=include_optimizer, **kwargs)
        print("All Models Saved!")


class SaveAegan(tf.keras.callbacks.Callback):
    def __init__(self, save_images_every: int, save_model_every: int, only_weights: bool, save_path: str, data_gen,
                 tensorboard_logdir=None):
        """
        Callback for the AEGAN to save images and the models.
        :param save_images_every: int - after how many epochs images should be saved.
        :param save_model_every: int - after how many epochs the models should be saved.
        :param only_weights: bool - whether only weights or the whole model should be saved.
        :param save_path: str - a base path where the outputs should be saved to.
        :param data_gen: generator - returning real images for the reconstruction images.
        :param tensorboard_logdir: str - if tensorboard is used, it is possible to write the pictures there.
        """
        super(SaveAegan, self).__init__()
        self.epoch = 0
        # Images
        self.save_images_every = save_images_every
        self.static_noise = None
        self.image_save_path = setup_path(save_path, optional_join="images/")
        self.static_image_save_path = setup_path(self.image_save_path, optional_join="static_images/")

        # Models
        self.save_model_every = save_model_every
        self.only_weights = only_weights
        self.model_save_path = setup_path(save_path, optional_join="models/")

        self.get_test_images = lambda: data_gen.next()

        # tensorboard
        self.tbw_generated_imgs = tf.summary.create_file_writer(
            tensorboard_logdir + '/generated_images') if tensorboard_logdir is not None else None
        self.tbw_reconstructed_imgs = tf.summary.create_file_writer(
            tensorboard_logdir + '/reconstructed_images') if tensorboard_logdir is not None else None

    def save_images_to_grid(self, images, prefix: str):
        """
        Produces a grid from the images and saves it to file.
        :param images: tensor of images.
        :param prefix: str - prefix when saving.
        :return: the produced grid.
        """
        grid = to_grid(images, border=10)
        save_images(grid, save_to=self.image_save_path, prefix=prefix)
        return grid

    def make_and_write_images(self, epoch, logs, suffix=""):
        """
        Produces and saves reconstructions and generated images to file and if specified to tensorboard.
        :param epoch: int - the origin epoch of the logs and models.
        :param logs: dict - from event methods. Must have key 'aegan_loss'.
        :param suffix: str - a suffix for saving the images.
        """
        # get many generated images.
        imgs = self.get_test_images()
        generated_imgs = self.model.generate_images(imgs.shape[0] * 8)
        generated_imgs = tf.reshape(generated_imgs, (2, 4*imgs.shape[0], *imgs.shape[1:]))
        gen_grid = self.save_images_to_grid(generated_imgs, f"{epoch}generated_loss{logs['aegan_loss']: .4f}{suffix}_")

        # reconstruct real images and add the MAE to the filename.
        reconstructed_imgs = self.model.autoencode_images(imgs)
        reconstruction_loss = tf.keras.metrics.mean_absolute_error(imgs, reconstructed_imgs)
        reconstruction_loss = tf.reduce_mean(reconstruction_loss)
        combined = tf.stack([imgs, reconstructed_imgs], axis=0)
        rec_grid = self.save_images_to_grid(combined, f"{epoch}reconstructed_loss{reconstruction_loss: .4f}{suffix}_")

        # write the images to tensorboard if specified.
        if self.tbw_reconstructed_imgs is not None:
            with self.tbw_generated_imgs.as_default():
                tf.summary.image("Generated Images", gen_grid, step=epoch)
            with self.tbw_reconstructed_imgs.as_default():
                tf.summary.image("Reconstructed Images", rec_grid, step=epoch)

    def save_submodels(self, epoch, suffix=""):
        """
        Saves all sub models: encoder, generator, discriminator_image, discriminator_latent
        :param epoch: int - the current epoch.
        :param suffix: str - suffix for the file name.
        """
        if self.only_weights:
            self.model.save_weights(
                filepath=self.model_save_path + str(epoch) + "_{}" + suffix + "_w.h5",
                save_format="h5"
            )
        else:
            self.model.save(
                filepath=self.model_save_path + str(epoch) + "_{}" + suffix + "_m.h5",
                save_format="h5"
            )

    def on_train_begin(self, logs=None):
        # obtain static noise when training begins. (Must be here bc the model needs to be initialized.)
        self.static_noise = self.model.noise_generating_fn(9)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch + 1

        # produces images from a static noise vector to see how training develops.
        static_images = self.model.generator(self.static_noise, training=False)
        save_images(static_images, self.static_image_save_path, f"{epoch}_static_")

        if remainder_is_0(self.epoch, self.save_images_every):
            self.make_and_write_images(self.epoch, logs)

        if remainder_is_0(self.epoch, self.save_model_every):
            self.save_submodels(self.epoch)

    def on_train_end(self, logs=None):
        # if the images and models are not saved already, they will in the last epoch.
        if not remainder_is_0(self.epoch, self.save_images_every):
            self.make_and_write_images(self.epoch, logs, suffix="_F")

        if not remainder_is_0(self.epoch, self.save_model_every):
            self.save_submodels(self.epoch, suffix="_F")


if __name__ == '__main__':
    aegan = AEGAN((64, 64, 3), 10, 16 * 8, lambda b: tf.random.normal((b, 10)), False)
    tb = tf.keras.callbacks.TensorBoard(log_dir="./logs", update_freq="batch")
    aegan.fit(tf.ones((16 * 8 * 210, 64, 64, 3)), batch_size=16 * 8, epochs=2, callbacks=[tb])
