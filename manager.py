import tensorflow as tf
import GAN
import utils


class GANManager:
    """
    The Manager handles the whole training process for a GAN.
    """

    def __init__(self, kwargs, generator_kwargs, discriminator_kwargs):
        """
        The Manager controls the training of our GAN.
        :param kwargs:
        :param generator_kwargs:
        :param discriminator_kwargs:
        """
        self.generator = GAN.Generator(**generator_kwargs)
        self.discriminator = GAN.Discriminator(**discriminator_kwargs)

        self.batch_size = kwargs["batch_size"]

        # make loss and optimizers as model params?
        self.loss_function = kwargs["loss"]
        self.optimizer = kwargs["optimizer"]

        self.data = tf.ones((128, 32, 32, 3))  # todo add data

    def get_noise(self, batch_size=None, mu=0.5, sd=0.5):
        """
        Creates noise of predefined shape with specified batch size.
        :param batch_size: int - batch size, if None the default batch_size of the manager will be used.
        :param mu: float - mu of the normal distribution.
        :param sd: float - standard deviation of the normal distribution.
        :return: Noise of shape (batch_size, XXX)
        """
        return tf.random.normal(
            shape=(
                self.batch_size if batch_size is None else batch_size,
                self.generator.latentspace
            ),
            mean=mu, stddev=sd
        )

    def generator_loss(self, discriminators_prediction):
        """
        Computes the loss for the Generator.
        :param discriminators_prediction: Obtained predictions from the discriminator for the generated images.
        :return: Loss.
        """
        return self.loss_function(
            tf.ones_like(discriminators_prediction),
            discriminators_prediction
        )

    def discriminator_loss(self, real_images_prediction, generated_images_prediction):
        """
        Computes the loss for the Discriminator.
        :param real_images_prediction: Obtained predictions from the Discriminator for the real images.
        :param generated_images_prediction: Obtained predictions from the Discriminator for the generated images.
        :return: Loss.
        """
        # The Discriminator is supposed to predict the real images as real (1) and the generated ones as fake (0)
        loss_for_real_images = self.loss_function(
            y_true=tf.ones_like(real_images_prediction),
            y_pred=real_images_prediction
        )
        loss_for_generated_images = self.loss_function(
            y_true=tf.zeros_like(generated_images_prediction),
            y_pred=generated_images_prediction
        )
        return loss_for_real_images + loss_for_generated_images

    def forward_step(self, real_images, trainings_frequency, proportion_real=0.5):
        """
        Compute one forward step
        :param real_images: Input one batch of real images to train the Discriminator
        :param partition: tuple(n_g, n_d) - how often will each component be trained before switching.
        :param  proportion_real: float[0:1] - how much percent of the should be replaced by generated images?
        :return: float - Generator loss, float - Discriminator loss
        """
        generator_losses = [self.forward_step_generator() for _ in range(trainings_frequency[0])]
        # gets automatically passed trainings_frequency[1] many partial batches.
        discriminator_losses = [self.forward_step_discriminator(images, proportion_real=proportion_real) for images in
                                real_images]

        return tf.reduce_mean(generator_losses), tf.reduce_mean(discriminator_losses)

    def forward_step_generator(self):
        """
        Computes one forward step for the generator. We generate images and try to fool the discriminator.
        :return: float - loss for the generator.
        """
        with tf.GradientTape() as tape:
            # Generate images from noise
            generated_images = self.generate_images(training=True)

            # try to fool the discriminator and assess our success.
            prediction = self.discriminator(generated_images, training=False)
            loss = self.generator_loss(discriminators_prediction=prediction)

            # improve generator based on its assessed loss.
            gradients = tape.gradient(loss, self.generator.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        return loss

    def forward_step_discriminator(self, real_images, proportion_real=0.5):
        """
        Computes one forward step for the discriminator. We want to predict all real images as real and all the generated ones as fakes.
        :param real_images: tensor - a batch of real images
        :param proportion_real: float[0:1] - how much percent of the batch should be replaced by generated images?
        :return: float - loss for the discriminator.
        """

        with tf.GradientTape() as tape:
            # generate some images to train the discriminator corresponding to the inverse proportion.
            generated_images = self.generate_images(int((1 - proportion_real) * self.batch_size))

            # take the remaining portion from our real images.
            train_batch = tf.concat((real_images, generated_images), axis=0)

            # train the discriminator and assess its performance.
            predictions = self.discriminator(train_batch, training=True)
            loss = self.discriminator_loss(real_images_prediction=predictions, generated_images_prediction=predictions)

            # improve discriminator based on its assessed loss.
            gradients = tape.gradient(loss, self.discriminator.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

        return loss

    def train(
            self,
            epochs,
            samples_per_epoch=1200,
            trainings_frequency=(5, 5),
            proportion_real=0.5,
            print_every=5,
            save_pictures_every=5,
            how_many_pictures_to_save=20,
            save_model_every=5
    ):
        # TODO write docstring
        # TODO eventually sample or save some generated pictures. Maybe sampled ones with download link?

        assert 0 <= proportion_real <= 1, "Proportion must be between 0-1"

        # how much real images we want to take to get the right proportion given the batch size
        take_reals = tf.cast(tf.math.ceil(self.batch_size * proportion_real) * trainings_frequency[1], tf.int32)

        generator_losses_accumulator, discriminator_losses_accumulator = [], []
        try:  # Like this we will have to opportunity to end training early and save the model.
            for epoch in range(epochs):
                count = 0
                while count < samples_per_epoch:
                    count += samples_per_epoch

                    data = self.data.take(take_reals)
                    generator_loss, discriminator_loss = self.forward_step(data, trainings_frequency, proportion_real)

                    generator_losses_accumulator.append(generator_loss)
                    discriminator_losses_accumulator.append(discriminator_loss)
        # ----------- Early Abortion ------------
        except KeyboardInterrupt:
            if "1" == input("Manual abortion.. to save the current model answer '1'"):
                try:
                    self.save_model()
                except Exception as e:
                    print("Saving the Model was unsuccessful with the exception", type(e), e)
        # ---------------------------------------


    def save_model(self):
        # TODO implement save_model function - maybe only save weights?!
        # TODO save one or both models?!
        pass

    def load_best_model(self, name=""):
        """
        Loads the best model so far if no name is specified.
        :param name: string - name of model.
        :return:
        """
        # TODO is there a measure for a good generator which we can later get again?
        # TODO load only one model?!
        pass

    def generate_images(self, number=None, training=False):
        """
        Returns pictures using the Generator.
        :param number: int - number of pictures.
        :param training: bool - is the generator in training?
        :return: The above specified number of generated pictures.
        """
        noise = self.get_noise(number)
        images = self.generator(noise, training=training)
        return images

    def generate_and_save_images(self, number, save_to="", prefix="", return_images=False):
        """
        Generates images with the current managers generator.
        :param number: int - number of images to create.
        :param save_to: string - path to the destination directory. Don't forget the '/' at the end!
        :param prefix: string - name prefix for saved images.
        :param return_images: bool - weather the images should also be returned or not.
        :return: if return_images==True the generated pictures will be returned.
        """
        images = self.generate_images(number, training=False)
        utils.save_images(images, save_to, prefix)

        if return_images:
            return images
