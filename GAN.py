import tensorflow as tf

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
    def __init__(self):
        super(Generator, self).__init__()
        self.layer_list = []

    # @tf.function
    def call(self, x, training=False):
        for layer in self.layer_list:
            x = layer(x, training=training)
        return x
