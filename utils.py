import tensorflow as tf
import GAN


class Manager:
    """
    The Manager should handle the whole trainings process for a GAN.
    """

    def __init__(self, kwargs, generator_kwargs, discriminator_kwargs):
        self.generator = GAN.Generator(generator_kwargs)
        self.discriminator = GAN.Discriminator(discriminator_kwargs)

        self.batch_size = kwargs["batch_size"]

        self.loss_function = kwargs["loss"]

    def get_noice(self, batch_size=None, mu=0.5, sd=0.5):
        """
        Creates noise of predefined shape with specified batch size.
        :param batch_size: int - batch size, if None the default batch_size of the manager will be used.
        :return: Noise of shape (batch_size, XXX)
        """
        return tf.random.normal(shape=(batch_size, self.generator.latentspace), mean=mu, stddev=sd)

    def generator_loss(self, discriminators_prediction):
        """
        Computes the loss from the generator.
        :param discriminators_prediction: Obtained predictions from the discriminator.
        :return: Loss
        """
        return self.loss_function(tf.ones_like(discriminators_prediction), discriminators_prediction)

    def forward_step(self, real_images):
        """
        Compute one forward step
        :param real_images: Input one batch of real images to train the Discriminator
        :return: Generator loss, Discriminator loss
        """
        pass

    def get_pictures(self, number):
        """
        Returns pictures using the Generator.
        :param number: int - number of pictures.
        :return: The above specified number of generated pictures.
        """
        noise = self.get_noise(number)
        return self.generator(noise)
