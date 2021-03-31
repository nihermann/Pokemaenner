import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Flatten, Concatenate, Reshape, UpSampling2D, Conv2D, Activation,
                                     LeakyReLU, GaussianNoise, GaussianDropout, LayerNormalization)
import numpy as np
from utils import default_value

tf.keras.backend.set_floatx('float32')


def conv2d_block(X, channel, kernel_width, stride, initializer, add_noise, hidden_activation):
    X = Conv2D(channel, kernel_width, stride, padding="same", kernel_initializer=initializer)(X)
    if add_noise:
        X = GaussianNoise(0.01)(X)
    X = LayerNormalization()(X)
    X = hidden_activation(X)
    return X


def up_sampling_block(A, B, up, channel, kernel_width, stride, initializer, hidden_activation):
    A = Concatenate()([A, B])
    A = UpSampling2D(up)(A)
    B = UpSampling2D(up)(B)
    A = Conv2D(channel, kernel_width, stride,
               padding="same", kernel_initializer=initializer)(A)
    A = LayerNormalization()(A)
    A = hidden_activation(A)
    return A, B


def dense_block(X, neurons_per_layer, add_noise, hidden_activation):
    Y = Dense(neurons_per_layer)(X)

    if add_noise:
        Y = GaussianNoise(0.005)(Y)

    Y = LayerNormalization()(Y)
    Y = hidden_activation(Y)
    return Concatenate()([X, Y])


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

        h_Activation = lambda activation: LeakyReLU(0.02) if activation == "leaky_relu" else Activation(activation)
        initializer = tf.keras.initializers.RandomNormal(0, 0.02)

        inp = Input(image_shape, name="Input_encoder_or_discriminator")
        X = inp
        if add_noise:
            X = GaussianNoise(0.01)(X)

        # add more blocks of Conv2D, (noise), LayerNormalization, Activation
        for channel, kernel_width, stride in zip(self.channels, self.kernel_widths, self.strides):
            X = conv2d_block(
                X,
                channel, kernel_width, stride, initializer,  # Conv2D
                add_noise,  # Gaussian Noise with 0.01
                #  Layer normalization
                h_Activation(hidden_activation)  # Activation
            )  # returns X

        # Final block - Flatten and a Dense as output layer.
        X = Flatten()(X)
        X = Dense(latentspace, kernel_initializer=initializer)(X)
        out = Activation(output_activation, name="encoder_or_discriminator_output")(X)

        # Construct the functional model by calling the constructor of the Model super class.
        super(Encoder, self).__init__(inp, out, **kwargs)

    def forward_step(self, x, target):
        """
        Basic forward step.
        :param x: input tensor.
        :param target: its respective targets to compute the loss and backprop.
        :return: tensor with the loss.
        """
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
        # assign the default values if they were not specified
        channels = default_value([256, 128, 128, 64, 64, 3], channels)
        kernel_widths = default_value([4, 4, 4, 4, 4, 4], kernel_widths)
        strides = default_value([1, 1, 1, 1, 1, 1], strides)
        up_sampling = default_value([2, 2, 2, 2, 1, 1], up_sampling)

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

        for channel, kernel_width, stride, up in zip(channels[:-1], kernel_widths[:-1], strides[:-1], up_sampling[:-1]):
            A, B = up_sampling_block(
                A, B,  # Concat A&B to A
                up,  # Up sampling for A&B
                channel, kernel_width, stride, initializer,  # Conv2D for A
                # Layer normalization
                h_Activation(hidden_activation)  # Activation for A
            )  # returns A, B

        # Final block which finally produces the output image.
        A = Concatenate()([A, B])
        A = Conv2D(channels[-1], kernel_widths[-1], strides[-1],
                   padding="same", kernel_initializer=initializer)(A)
        out = h_Activation(output_activation)(A)

        # Construct the functional model by calling the constructor of the Model super class.
        super(Decoder, self).__init__(inp, out, **kwargs)


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
                neurons_per_layer,  # Dense layer
                add_noise,  # GaussianNoise with 0.005
                # Layer normalization
                h_Activation(hidden_activation)
            )  # returns Concat [X_input, X]

        x = Dense(128)(x)
        x = h_Activation(hidden_activation)(x)
        x = Dense(1)(x)
        out = h_Activation(output_activation)(x)

        # Construct the functional model by calling the constructor of the Model super class.
        super(DiscriminatorLatent, self).__init__(inp, out, **kwargs)

    def forward_step(self, x, target):
        """
        Basic forward step.
        :param x: input tensor.
        :param target: its respective targets to compute the loss and backprop.
        :return: tensor with the loss.
        """
        with tf.GradientTape() as tape:
            pred = self(x)
            loss = self.compiled_loss(target, pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return loss


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
