import tensorflow as tf
import os
import models


class AEGAN:
    def __init__(
            self,
            image_shape,
            latentspace,
            continue_from_saved_models=False,
            path="./models",
            load_compiled=False,
    ):

        loading_successful = False
        if continue_from_saved_models:
            loading_successful = self.try_load_all_models(path, load_compiled)
        if not loading_successful:
            self.generator = models.Decoder(
                latentspace,
                first_reshape_shape=[4, 4, 64],
                hidden_activation="relu",
                output_activation="tanh",
                name="generator"
            )

            self.encoder = models.Encoder(
                image_shape,
                latentspace,
                hidden_activation="relu",
                output_activation="linear",
                name="encoder"
            )

            self.discriminator_image = models.Encoder(
                image_shape,
                latentspace=1,
                kernel_widths=[4, 4, 4, 4, 4, 4],
                hidden_activation="leaky_relu",
                output_activation="sigmoid",
                name="image_discriminator"
            )

            self.discriminator_latent = models.DiscriminatorLatent(latentspace, name="latent_discriminator")

        self.aegan = self.build_aegan()
        self.compile_aegan()

    def try_load_all_models(self, path, compile=False):
        files = os.listdir(path)
        files = [f for f in files if f.endswith(".h5")]
        print("Models found: ", end="")
        num_loaded = 0
        found_g, found_e, found_l, found_i = False, False, False, False
        for model in reversed(sorted(files, key=len)):
            if not found_g and "generator" in model:
                found_g, num_loaded = True, num_loaded + 1
                print(model, end=" - ")
                self.generator = tf.keras.models.load_model(os.path.join(path, model), compile=compile)
            elif not found_e and "encoder" in model:
                found_e, num_loaded = True, num_loaded + 1
                print(model, end=" - ")
                self.encoder = tf.keras.models.load_model(os.path.join(path, model), compile=compile)
            elif not found_l and "discriminator_latent" in model:
                found_l, num_loaded = True, num_loaded + 1
                print(model, end=" - ")
                self.discriminator_latent = tf.keras.models.load_model(os.path.join(path, model), compile=compile)
            elif not found_i and "discriminator_image" in model:
                found_i, num_loaded = True, num_loaded + 1
                print(model, end=" - ")
                self.discriminator_image = tf.keras.models.load_model(os.path.join(path, model), compile=compile)

        if num_loaded != 4:
            print("\nLoading models was unsuccessful... new models are created...")
        else:
            print("\nLoading was successful! Continuing with loaded models...\n")
        return num_loaded == 4

    def build_aegan(self):
        self.discriminator_image.trainable = False
        self.discriminator_latent.trainable = False
        input_image_shape = self.encoder.input_shape[1:]
        try:
            input_latent_shape = self.generator.input_shape[1:]
        except AttributeError:
            input_latent_shape = self.generator.start_shape[1:]

        x_real = tf.keras.layers.Input(input_image_shape, name="image_input")
        z_real = tf.keras.layers.Input(input_latent_shape, name="latentspace_input")

        z_hat = self.encoder(x_real)
        x_tilde = self.generator(z_hat)
        x_hat = self.generator(z_real)
        z_tilde = self.encoder(x_hat)

        prediction_x_hat = self.discriminator_image(x_hat)
        prediction_x_tilde = self.discriminator_image(x_tilde)

        prediction_z_hat = self.discriminator_latent(z_hat)
        prediction_z_tilde = self.discriminator_latent(z_tilde)

        return tf.keras.Model([x_real, z_real], [x_tilde, z_tilde,
                                                 prediction_x_hat,
                                                 prediction_x_tilde,
                                                 prediction_z_hat,
                                                 prediction_z_tilde],
                                                 name="AEGAN")

    def compile_aegan(self):
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

if __name__ == '__main__':
    aegan = AEGAN((64, 64, 3), 10, True)
    aegan.aegan.summary()
    # tf.keras.utils.plot_model(aegan.aegan, "AEGAN.png", show_shapes=True)

