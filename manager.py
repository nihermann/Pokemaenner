import tensorflow as tf
import GAN


class GAN_Manager:
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

        self.loss_function = kwargs["loss"]

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

    def forward_step(self, real_images):
        """ # TODO write docstring
        Compute one forward step
        :param real_images: Input one batch of real images to train the Discriminator
        :return: Generator loss, Discriminator loss
        """
        # TODO implement forward_step
        pass

    def train(self, epochs, print_every=5, save_each=5):
        """ # TODO write docstring

        :param epochs:
        :param print_every:
        :param save_each:
        :return:
        """
        # TODO eventually sample or save some generated pictures. Maybe sampled ones with download link?
        # TODO implement train function.
        pass

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

    def get_pictures(self, number):
        """
        Returns pictures using the Generator.
        :param number: int - number of pictures.
        :return: The above specified number of generated pictures.
        """
        # TODO maybe save also to path?
        noise = self.get_noise(number)
        return self.generator(noise)
