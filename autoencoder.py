import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Flatten, Concatenate, Reshape, UpSampling2D, Conv2D, Activation,
                                     LeakyReLU, GaussianNoise, GaussianDropout, LayerNormalization)
import numpy as np

tf.keras.backend.set_floatx('float32')


class Encoder(tf.keras.Model):
    def __init__(self, image_shape, latentspace, add_noise=True, **kwargs):
        channels = [64, 64, 128, 128, 128, 256]
        kernel_widths = [4, 4, 4, 4, 4, 4]
        strides = [2, 2, 2, 2, 1, 1]
        hidden_activation = "relu"
        output_activation = "linear"

        initializer = tf.keras.initializers.RandomNormal(0, 0.02)

        inp = Input(image_shape)
        X = inp
        if add_noise:
            X = GaussianNoise(0.01)(X)
        # add more blocks of Conv2D, (noise), LayerNormalization, Activation
        for channel, kernel_width, stride in zip(channels, kernel_widths, strides):
            X = Conv2D(channel, kernel_width, stride, padding="same", kernel_initializer=initializer)(X)
            if add_noise:
                X = GaussianNoise(0.01)(X)
            X = LayerNormalization()(X)
            X = Activation(hidden_activation)(X)

        X = Flatten()(X)
        X = Dense(latentspace, kernel_initializer=initializer)(X)
        out = Activation(output_activation)(X)

        super(Encoder, self).__init__(inp, out, **kwargs)

class Decoder(tf.keras.Model):
    def __init__(self, latentspace, first_reshape_shape, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        channels = [256, 128, 128, 64, 64, 3]
        kernel_widths = [4, 4, 4, 4, 4, 4]
        strides = [1, 1, 1, 1, 1, 1]
        up_sampling = [2, 2, 2, 2, 1, 1]
        hidden_activation = "relu"
        output_activation = "tanh"

        initializer = tf.keras.initializers.RandomNormal(0, 0.02)
        self.layers_A = [
            Dense(np.prod(first_reshape_shape), kernel_initializer=initializer, input_shape=(latentspace,)),
            LayerNormalization(),
            Activation(hidden_activation),
            Reshape(first_reshape_shape)
        ]

        self.layers_B = [
            Dense(64, input_shape=(latentspace,)),
            LayerNormalization(),
            Activation(hidden_activation),
            Reshape((1, 1, 64)),
            UpSampling2D(first_reshape_shape[0])
        ]

        self.layer_blocks = []
        for channel, kernel_width, stride, up in zip(channels[:-1], kernel_widths[:-1], strides[:-1], up_sampling[:-1]):
            self.layer_blocks.append([
                Concatenate(),
                UpSampling2D(up),
                UpSampling2D(up),
                Conv2D(channel, kernel_width, stride,
                       padding="same", kernel_initializer=initializer),
                LayerNormalization(),
                Activation(hidden_activation)
            ])
        self.last_concat = Concatenate()
        self.last_Conv2D = Conv2D(channels[-1], kernel_widths[-1], strides[-1],
                                  padding="same", kernel_initializer=initializer)
        self.last_activation = Activation(output_activation)

        self.build((None, latentspace))

    @tf.function
    def call(self, inputs):
        a = tf.identity(inputs)
        b = tf.identity(inputs)
        for a_layer in self.layers_A:
            a = a_layer(a)

        for b_layer in self.layers_B:
            b = b_layer(b)

        for block in self.layer_blocks:
            a = block[0]([a, b])  # Concatenate
            a = block[1](a)  # UpSampling2D
            b = block[2](b)  # UpSampling2D
            a = block[3](a)  # Conv2D
            a = block[4](a)  # LayerNormalization
            a = block[5](a)  # Activation

        a = self.last_concat([a, b])
        a = self.last_Conv2D(a)
        a = self.last_activation(a)

        return a

class Autoencoder(tf.keras.Model):
    def __init__(self, image_shape, latentspace, add_encoder_noise=True, **kwargs):
        super(Autoencoder, self).__init__(**kwargs)
        self.encoder = Encoder(image_shape, latentspace, add_noise=add_encoder_noise)
        self.decoder = Decoder(latentspace, [4, 4, 64])

        self.build((None, *image_shape))

    @tf.function
    def call(self, x):
        encoding = self.encoder(x)
        return self.decoder(encoding)
    
    def encode_images(self, images):
        return self.encoder.predict(images)
    
    def autoencode_images(self, images):
        return self.predict(images)



if __name__ == "__main__":
    img = tf.ones((1, 64, 64, 3))
    # dec = Decoder(10, [4, 4, 64])
    # print(dec.summary())
    # enc = Encoder((64, 64, 3), 10)
    # print(enc.summary())
    a = Autoencoder((64, 64, 3), 10)
    print(a.summary())
    # print(a.encode(img))
    # print(a.predict(img))