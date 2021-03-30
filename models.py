import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Flatten, Concatenate, Reshape, UpSampling2D, Conv2D, Activation,
                                     LeakyReLU, GaussianNoise, GaussianDropout, LayerNormalization)
import numpy as np
from utils import default_value

tf.keras.backend.set_floatx('float32')


class Encoder(tf.keras.Model):
    def __init__(
            self,
            image_shape,
            latentspace,
            channels=None,
            kernel_widths=None,
            strides=None,
            hidden_activation="relu",
            output_activation="linear",
            add_noise=True,
            **kwargs
    ):
        """
        An Encoder structure which can also be used as Discriminator.
        :param image_shape: tuple - with shape of the images w/o batch dimension.
        :param latentspace: int - how many output neurons.
        :param channels: list - of channel values. Default if left on None: [64, 64, 128, 128, 128, 256]
        :param kernel_widths: list - of kernel widths. Default if left on None: [4, 4, 4, 4, 4, 4]
        :param strides: list - of stride values. Default if left on None: [2, 2, 2, 2, 1, 1]
        :param hidden_activation: string - which activation is used in hidden layers. Default: 'relu'
        :param output_activation: string - which activation is used in the output layer. Default: 'linear'
        :param add_noise: bool - weather noise should be added to the layers computations.
        :param kwargs: additional kwargs for the Model super class.
        """
        self.channels = default_value([64, 64, 128, 128, 128, 256], channels)
        self.kernel_widths = default_value([4, 4, 4, 4, 4, 4], kernel_widths)
        self.strides = default_value([2, 2, 2, 2, 1, 1], strides)

        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        h_Activation = lambda activation: LeakyReLU(0.02) if activation == "leaky_relu" else Activation(activation)

        initializer = tf.keras.initializers.RandomNormal(0, 0.02)

        inp = Input(image_shape, name="Input_encoder_or_discriminator")
        X = inp
        if add_noise:
            X = GaussianNoise(0.01)(X)
        # add more blocks of Conv2D, (noise), LayerNormalization, Activation
        for channel, kernel_width, stride in zip(self.channels, self.kernel_widths, self.strides):
            X = Conv2D(channel, kernel_width, stride, padding="same", kernel_initializer=initializer)(X)
            if add_noise:
                X = GaussianNoise(0.01)(X)
            X = LayerNormalization()(X)
            X = h_Activation(hidden_activation)(X)

        X = Flatten()(X)
        X = Dense(latentspace, kernel_initializer=initializer)(X)
        out = Activation(output_activation, name="encoder_or_discriminator_output")(X)

        super(Encoder, self).__init__(inp, out, **kwargs)

    def forward_step(self, x, target):
        with tf.GradientTape() as tape:
            pred = self(x)
            loss = self.compiled_loss(target, pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss


class Decoder(tf.keras.Model):
    def __init__(
            self,
            latentspace,
            first_reshape_shape,
            channels=None,
            kernel_widths=None,
            strides=None,
            up_sampling=None,
            hidden_activation="relu",
            output_activation="tanh",
            **kwargs):
        """
        A Decoder structure which can also be used as a Generator.
        :param latentspace: int - size of the input embedding.
        :param first_reshape_shape: tuple - contains the shape for the first reshape w/o batch dimension.
        :param channels: list - of channel values. Default if left on None: [256, 128, 128, 64, 64, 3]
        :param kernel_widths: list - of kernel widths. Default if left on None: [4, 4, 4, 4, 4, 4]
        :param strides: list - of stride values. Default if left on None: [1, 1, 1, 1, 1, 1]
        :param up_sampling: list - of up sampling size. Default if left on None: [2, 2, 2, 2, 1, 1]
        :param hidden_activation: string - which activation is used in hidden layers. Default: 'relu'
        :param output_activation: string - which activation is used in the output layer. Default: 'tanh'
        :param kwargs: additional kwargs for the Model super class.
        """
        self.channels = default_value([256, 128, 128, 64, 64, 3], channels)
        self.kernel_widths = default_value([4, 4, 4, 4, 4, 4], kernel_widths)
        self.strides = default_value([1, 1, 1, 1, 1, 1], strides)
        self.up_sampling = default_value([2, 2, 2, 2, 1, 1], up_sampling)

        self.output_activation = output_activation

        h_Activation = lambda activation: LeakyReLU(0.02) if activation == "leaky_relu" else Activation(activation)
        initializer = tf.keras.initializers.RandomNormal(0, 0.02)

        inp = Input((latentspace,))

        # A Block
        A = Dense(np.prod(first_reshape_shape), kernel_initializer=initializer)(inp)
        A = LayerNormalization()(A)
        A = h_Activation(hidden_activation)(A)
        A = Reshape(first_reshape_shape)(A)

        # B Block
        B = Dense(64)(inp)
        B = LayerNormalization()(B)
        B = h_Activation(hidden_activation)(B)
        B = Reshape((1, 1, 64))(B)
        B = UpSampling2D(first_reshape_shape[0])(B)

        for channel, kernel_width, stride, up in zip(self.channels[:-1], self.kernel_widths[:-1], self.strides[:-1],
                                                     self.up_sampling[:-1]):
            A, B = up_sampling_block(
                A, B,
                channel,
                kernel_width,
                stride,
                up,
                h_Activation(hidden_activation),
                initializer
            )

        A = Concatenate()([A, B])
        A = Conv2D(self.channels[-1], self.kernel_widths[-1], self.strides[-1],
                   padding="same", kernel_initializer=initializer)(A)
        out = Activation(self.output_activation)(A)

        super(Decoder, self).__init__(inp, out, **kwargs)

        self.build((None, latentspace))

    #     input_layer = Input((latent_dim,))
    #     X = Dense(np.prod(starting_shape),
    #               kernel_initializer=init)(input_layer)
    #     X = LayerNormalization()(X)
    #     if hidden_activation == 'leaky_relu':
    #         X = LeakyReLU(0.02)(X)
    #     else:
    #         X = Activation(hidden_activation)(X)
    #     X = Reshape(starting_shape)(X)
    #
    #     Y = Dense(64)(input_layer)
    #     Y = LayerNormalization()(Y)
    #     if hidden_activation == 'leaky_relu':
    #         Y = LeakyReLU(0.02)(Y)
    #     else:
    #         Y = Activation(hidden_activation)(Y)
    #     Y = Reshape((1, 1, 64))(Y)
    #     Y = UpSampling2D(np.array(starting_shape[:2]))(Y)
    #
    #     for i in range(len(channels)-1):
    #         X = Concatenate()([X, Y])
    #         X = UpSampling2D(upsampling[i])(X)
    #         Y = UpSampling2D(upsampling[i])(Y)
    #         X = Conv2D(channels[i], kernel_widths[i], strides=strides[i],
    #                    padding='same', kernel_initializer=init)(X)
    #         X = LayerNormalization()(X)
    #         if hidden_activation == 'leaky_relu':
    #             X = LeakyReLU(0.02)(X)
    #         else:
    #             X = Activation(hidden_activation)(X)
    #     else:
    #         X = Concatenate()([X, Y])
    #         X = Conv2D(channels[-1], kernel_widths[-1], strides=strides[-1],
    #                    padding='same', kernel_initializer=init)(X)
    #         output_layer = Activation(output_activation)(X)
    #
    #
    #     return input_layer, output_layer

    # @tf.function
    # def call(self, inputs):
    #     a = tf.identity(inputs)
    #     b = tf.identity(inputs)
    #     for a_layer in self.layers_A:
    #         a = a_layer(a)  # Out 4x4x64 A
    #
    #     for b_layer in self.layers_B:
    #         b = b_layer(b)  # out 4x4x64 B
    #
    #     # Up sampling with skip connection B
    #     for block in self.layer_blocks:
    #         a = block[0]([a, b])  # Concatenate A&B
    #         a = block[1](a)  # UpSampling2D A
    #         b = block[2](b)  # UpSampling2D B
    #         a = block[3](a)  # Conv2D A
    #         a = block[4](a)  # LayerNormalization A
    #         a = block[5](a)  # Activation A
    #
    #     a = self.last_concat([a, b])
    #     a = self.last_Conv2D(a)
    #     a = self.last_activation(a)
    #
    #     return a


def up_sampling_block(A, B, channel, kernel_width, stride, up, hidden_activation, initializer):
    A = Concatenate()([A, B])
    A = UpSampling2D(up)(A)
    B = UpSampling2D(up)(B)
    A = Conv2D(channel, kernel_width, stride,
               padding="same", kernel_initializer=initializer)(A)
    A = LayerNormalization()(A)
    A = hidden_activation(A)
    return A, B


class DiscriminatorLatent(tf.keras.Model):
    def __init__(
            self,
            latentspace,
            num_blocks=16,
            neurons_per_layer=16,
            hidden_activation="relu",
            output_activation="sigmoid",
            add_noise=True,
            **kwargs
    ):
        """
        Residual Dense Network to discriminate latent vectors.
        :param latentspace: int - input embedding.
        :param num_blocks: int - how many Dense block to use.
        :param neurons_per_layer: int - how many neurons per Dense layer are used.
        :param hidden_activation: string - which activation to use in the hidden layers.
        :param output_activation: string - which activation to use in the output layer.
        :param add_noise: bool - weather to apply noise to the layers computations.
        :param kwargs: further keyword arguments for the Model superclass.
        """
        h_Activation = lambda activation: LeakyReLU(0.02) if activation == "leaky_relu" else Activation(activation)

        inp = Input((latentspace,))
        x = inp
        if add_noise:
            x = GaussianNoise(0.01)(x)

        for _ in range(num_blocks):
            x = dense_block(
                x,
                neurons_per_layer,
                h_Activation(hidden_activation),
                add_noise
            )

        x = Dense(128)(x)
        x = h_Activation(hidden_activation)(x)
        x = Dense(1)(x)
        out = Activation(output_activation)(x)

        super(DiscriminatorLatent, self).__init__(inp, out, **kwargs)

        self.build((None, latentspace))

    def forward_step(self, x, target):
        with tf.GradientTape() as tape:
            pred = self(x)
            loss = self.compiled_loss(target, pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss


def dense_block(X, neurons_per_layer, hidden_activation, add_noise=True):
    Y = Dense(neurons_per_layer)(X)

    if add_noise:
        Y = GaussianNoise(0.005)(Y)

    Y = LayerNormalization()(Y)
    Y = hidden_activation(Y)
    return Concatenate()([X, Y])


# class DenseBlock(tf.keras.Model):
#     def __init__(self, neurons_per_layer, hidden_activation_fn, add_noise=True):
#         super(DenseBlock, self).__init__()
#
#         self.block = [Dense(neurons_per_layer)]
#         if add_noise:
#             self.block.append(GaussianNoise(0.005))
#         self.block.append(LayerNormalization())
#         self.block.append(hidden_activation_fn)
#         self.concat = Concatenate()
#
#     @tf.function
#     def call(self, inputs):
#         x = tf.identity(inputs)
#         for layer in self.block:
#             x = layer(x)
#         return self.concat([inputs, x])


# class Autoencoder(tf.keras.Model):
#     def __init__(self, image_shape, latentspace, add_encoder_noise=True, **kwargs):
#         super(Autoencoder, self).__init__(**kwargs)
#
#         self.encoder = Encoder(image_shape, latentspace, add_noise=add_encoder_noise)
#         self.decoder = Decoder(latentspace, [4, 4, 64])
#
#         self.build((None, *image_shape))
#
#     @tf.function
#     def call(self, x):
#         # just encode and decode a batch of images.
#         encoding = self.encoder(x)
#         return self.decoder(encoding)
#
#     def encode_images(self, images):
#         """
#         Translates images to their respective latentspace embedding.
#         :param images: batch of images.
#         :return: embedded images.
#         """
#         return self.encoder.predict(images)
#
#     def autoencode_images(self, images):
#         """
#         Encodes and Decodes a batch of images for predictions. Same as autoencoder.predict(images)
#         :param images: batch of images.
#         :return: the reconstructed images.
#         """
#         return self.predict(images)


if __name__ == "__main__":
    img = tf.ones((1, 10))
    dec = Decoder(10, [4, 4, 64])
    # dec = Decoder
    dec(img)
    dec.summary()
    # enc = Encoder((64, 64, 3), 10)
    # enc.summary()
    # a = Autoencoder((64, 64, 3), 10)
    # a.summary()
    # d_latent = build_discriminator_latent(10, 16, 16, "leaky_relu")
    # d_latent = DiscriminatorLatent(10)
    # print("encoder" in dir(a))
    # tf.keras.utils.plot_model(dec, "Decoder.png", show_shapes=True)
    # print(a.encode_images(img))
    # print(a.predict(img))
