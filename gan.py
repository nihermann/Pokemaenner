import tensorflow as tf
from tensorflow.keras import layers
from models import Decoder
tf.keras.backend.set_floatx('float32')


class Discriminator(tf.keras.Model):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()


    @tf.function
    def call(self, x, training=False):
        for layer in self.layer_list:
            x = layer(x, training=training)
        return x


class Generator(tf.keras.Model):
    def __init__(self, latentspace):
        super(Generator, self).__init__()
        self._latentspace = latentspace
        self.layer_list = [
            layers.Dense(8 * 8 * 128, input_shape=(None, latentspace)),
            layers.Reshape((8, 8, 128)),
            layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")
            # layers.Dense(8 * 8 * 128, use_bias=False, input_shape=(None, latentspace)),
            # # layers.Activation('relu'),
            # layers.LeakyReLU(alpha=0.2),
            # layers.Reshape((8, 8, 128)),  # 8x8
            #
            # layers.Conv2DTranspose(256, kernel_size=4, strides=(2, 2), padding="same"),  # 16x16
            # layers.BatchNormalization(),
            # # layers.Activation('relu'),
            # layers.LeakyReLU(alpha=0.2),
            #
            # layers.Conv2DTranspose(128, kernel_size=4, strides=(2, 2), padding="same"),  # 32x32
            # layers.BatchNormalization(),
            # # layers.Activation('relu'),
            # layers.LeakyReLU(alpha=0.2),
            #
            # layers.Conv2DTranspose(64, kernel_size=4, strides=(2, 2), padding="same"),  # 64x64
            # layers.BatchNormalization(),
            # # layers.Activation('relu'),
            # layers.LeakyReLU(alpha=0.2),
            #
            # layers.Conv2DTranspose(32, kernel_size=4, strides=(2, 2), padding="same"),  # 128x138
            # layers.BatchNormalization(),
            # # layers.Activation('relu'),
            # layers.LeakyReLU(alpha=0.2),
            #
            # layers.Conv2DTranspose(16, kernel_size=4, strides=(2, 2), padding="same"),  # 256x256
            # layers.BatchNormalization(),
            # # layers.Activation('relu'),
            # layers.LeakyReLU(alpha=0.2),
            #
            # layers.Conv2D(4, kernel_size=1, padding="same"),
            # layers.Activation('sigmoid')

        ]

    @property
    def latentspace(self):
        """
        Makes the latentspace (input space/dim) read-only.
        :return: generators latentspace
        """
        return self._latentspace

    @tf.function
    def call(self, x, training=False):
        for layer in self.layer_list:
            x = layer(x, training=training)
        return x
