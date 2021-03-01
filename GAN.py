import tensorflow as tf
from tensorflow.keras import layers


class Discriminator(tf.keras.Model):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.layer_list = []

    # @tf.function
    def call(self, x, training=False):
        for layer in self.layer_list:
            x = layer(x, training=training)
        return x


class Generator(tf.keras.Model):
    def __init__(self, latentspace):
        super(Generator, self).__init__()
        self._latentspace = latentspace
        self.layer_list = []

    @property
    def latentspace(self):
        """
        Makes the latentspace (input space) read-only.
        :return: generators latentspace
        """
        return self._latentspace

    # @tf.function
    def call(self, x, training=False):
        for layer in self.layer_list:
            x = layer(x, training=training)
        return x
