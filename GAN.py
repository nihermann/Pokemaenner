import tensorflow as tf
from tensorflow.keras import layers


class Discriminator(tf.keras.Model):
    def __init__(self):
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
        # TODO make latentspace readonly
        self.latentspace = latentspace
        self.layer_list = []

    # @tf.function
    def call(self, x, training=False):
        for layer in self.layer_list:
            x = layer(x, training=training)
        return x
