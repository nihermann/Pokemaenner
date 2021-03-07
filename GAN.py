import tensorflow as tf
from tensorflow.keras import layers


class Discriminator(tf.keras.Model):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        self.layer_list = [
            # first conv layer
            layers.Conv2D(filters=32, strides=2, kernel_size=3, padding="same", input_shape=input_shape),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            # second conv layer
            layers.Conv2D(filters=64, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            # third conv layer
            layers.Conv2D(filters=80, kernel_size=3, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            # fourth conv layer
            layers.Conv2D(filters=100, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            # fifth conv layer
            layers.Conv2D(filters=128, kernel_size=3, strides=2, padding="same"),
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Flatten(),
            layers.Dense(1),
            layers.Activation("sigmoid")
        ]

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
            layers.Dense(8 * 8 * 64, use_bias=False, input_shape=(None, latentspace)),
            layers.Activation('relu'),
            layers.Reshape((8, 8, 128)),  # 8x8

            layers.Conv2DTranspose(256, kernel_size=4, strides=(2, 2), padding="same"),  # 16x16
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2DTranspose(128, kernel_size=4, strides=(2, 2), padding="same"),  # 32x32
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2DTranspose(64, kernel_size=4, strides=(2, 2), padding="same"),  # 64x64
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2DTranspose(32, kernel_size=4, strides=(2, 2), padding="same"),  # 128x138
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2DTranspose(16, kernel_size=4, strides=(2, 2), padding="same"),  # 256x256
            layers.BatchNormalization(),
            layers.Activation('relu'),

            layers.Conv2D(1, kernel_size=1, padding="same"),
            layers.Activation('sigmoid')

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
