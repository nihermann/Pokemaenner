
import numpy as np
from PIL import Image

import imageio
import time
import PIL

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys

# from IPython import display
# if not already installed
# import tensorflow_docs.vis.embed as embed

# from keras.utils.vis_utils import plot_model


# load the numpy datasets from the previously saved ones (for checkin and continuity)
images = np.load("data_reshaped_as_array/images_all_64.npy")
labels = np.load("data_reshaped_as_array/labels_all_64.npy")

IMG_SHAPE = (64, 64, 3)
BATCH_SIZE = 32

# Size of the noise vector
noise_dim = 128

train_images = images
train_labels = labels
print(f"Number of examples: {len(train_images)}")
print(f"Shape of the images in the dataset: {train_images.shape[1:]}")

# Reshape each sample to (28, 28, 1) and normalize the pixel values in the [-1, 1] range
train_images = train_images.reshape(train_images.shape[0], *IMG_SHAPE).astype("float32")
train_images = (train_images - 127.5) / 127.5

print(images.shape)


# create block of a convolutional layer for the discriminator
def conv_block(
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=True,
        use_bn=False,
        use_dropout=False,
        drop_value=0.5,
):
    """function to define a convolutional block. Important parameters include
       filters,activation function or wheter or not BatchNormaliaztion/Dropout
       should be used. """
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)
    if use_bn:
        x = layers.BatchNormalization()(x)
    x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


# design the model with convolutional blocks
def get_discriminator_model():
    img_input = layers.Input(shape=IMG_SHAPE)
    x = img_input
    # Zero pad the input to make the input images size to (32, 32, 1).
    # x = layers.ZeroPadding2D(0)(img_input)
    # x = keras.Input(shape=(64, 64, 3))
    x = conv_block(
        x,
        128,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        use_bias=True,
        activation=layers.LeakyReLU(0.2),
        use_dropout=False,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        256,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
    )
    x = conv_block(
        x,
        512,
        kernel_size=(5, 5),
        strides=(2, 2),
        use_bn=False,
        activation=layers.LeakyReLU(0.2),
        use_bias=True,
        use_dropout=True,
        drop_value=0.3,
    )
    # x = conv_block(
    #     #     x,
    #     #     1004,
    #     #     kernel_size=(5, 5),
    #     #     strides=(2, 2),
    #     #     use_bn=True,
    #     #     activation=layers.LeakyReLU(0.2),
    #     #     use_bias=True,
    #     #     use_dropout=False,
    #     #     drop_value=0.3,
    #     # )

    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(1)(x)

    d_model = keras.models.Model(img_input, x, name="discriminator")
    return d_model


d_model = get_discriminator_model()
d_model.summary()


def upsample_block(
        x,
        filters,
        activation,
        kernel_size=(3, 3),
        strides=(1, 1),
        up_size=(2, 2),
        padding="same",
        use_bn=False,
        use_bias=True,
        use_dropout=False,
        drop_value=0.3,
):
    """Function implementing an upsample block which uses transposed convolution,
       to tell wheter or not the activation of a pixel in a random sample.
       :param use_bn = whether or not to use Batch normalization
       :param use_dropout = whether or not to use DropOut"""
    x = layers.UpSampling2D(up_size)(x)
    x = layers.Conv2D(
        filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
    )(x)

    if use_bn:
        x = layers.BatchNormalization()(x)

    if activation:
        x = activation(x)
    if use_dropout:
        x = layers.Dropout(drop_value)(x)
    return x


def get_generator_model():
    noise = layers.Input(shape=(noise_dim,))
    x = layers.Dense(8 * 8 * 256, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Reshape((8, 8, 256))(x)
    x = upsample_block(
        x,
        256,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=False,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        128,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=False,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x, 3, layers.Activation("tanh"), strides=(1, 1), use_bias=False, use_bn=True
    )

    g_model = keras.models.Model(noise, x, name="generator")
    return g_model


g_model = get_generator_model()
g_model.summary()


# define a class object for the complete WGAN combining both the previous discriminator and generator
class WGAN(keras.Model):
    def __init__(
            self,
            discriminator,
            generator,
            latent_dim,
            discriminator_extra_steps=3,
            gp_weight=10.0,
    ):
        """
        :param discriminator = discriminator model
        :param generator = generator model
        :latent_dim = size of the latent space
        :discriminator_extra_steps = how often to train the discriminator before the generator
        """
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        """
        :param d_optimizer, g_optimizer = optimizers for the respective model
        :param d_loss_fn, g_loss_fn = loss for the respective model
        """
        super(WGAN, self).compile()  # compile the model
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated image
        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
        diff = fake_images - real_images
        interpolated = real_images + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated image.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated image.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]

        # Get the batch size
        batch_size = tf.shape(real_images)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )
            with tf.GradientTape() as tape:
                # Generate fake images from the latent vector
                fake_images = self.generator(random_latent_vectors, training=True)
                # Get the logits for the fake images
                fake_logits = self.discriminator(fake_images, training=True)
                # Get the logits for the real images
                real_logits = self.discriminator(real_images, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_images, fake_images)
                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        with tf.GradientTape() as tape:
            # Generate fake images using the generator
            generated_images = self.generator(random_latent_vectors, training=True)
            # Get the discriminator logits for fake images
            gen_img_logits = self.discriminator(generated_images, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_img_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=6, latent_dim=128, starter_count=0, img_path="results_all/generated_img_{epoch}_{i}.png",
                 model=None, model_path="models/wgan_model"):
        """
        :param num_img = how many images to save per epoch
        :param starter_count = call when continuing training so that files match and do not override (use the last epoch
        that was reach of the previous training)
        :param model = model of which to save the weights
        """
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.starter_count = starter_count
        self.img_path = img_path
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        """
        :param epoch = the current epoch as prefix for the file name
        """
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images = (generated_images * 127.5) + 127.5

        if epoch % 10 == 0:
            self.model.save_weights("models/wgan_model_all", save_format='tf')
            print("Model was saved.")

        for i in range(self.num_img):
            img = generated_images[i].numpy()
            img = keras.preprocessing.image.array_to_img(img)
            img.save(self.img_path.format(epoch=(epoch + self.starter_count), i=i))


# Instantiate the optimizer for both networks
# (learning_rate=0.0002, beta_1=0.5 are recommended)
generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)
discriminator_optimizer = keras.optimizers.Adam(
    learning_rate=0.0002, beta_1=0.5, beta_2=0.9
)


# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function
def discriminator_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)


# Set the number of epochs for training.
epochs = 100

# Instantiate the WGAN model.
wgan = WGAN(
    discriminator=d_model,
    generator=g_model,
    latent_dim=noise_dim,
    discriminator_extra_steps=3,
)

# Compile the WGAN model.
wgan.compile(
    d_optimizer=discriminator_optimizer,
    g_optimizer=generator_optimizer,
    g_loss_fn=generator_loss,
    d_loss_fn=discriminator_loss,
)

# if you want to load the saved model:
wgan.load_weights("models/wgan_model_all")

# Instantiate the customer `GANMonitor` Keras callback.
cbk = GANMonitor(num_img=10, latent_dim=noise_dim, starter_count=0,
                 img_path="results_all/generated_img_{epoch}_{i}.png", model=wgan)

if __name__ == '__main__':
    try:
        # Start training the model.
        wgan.fit(train_images, batch_size=BATCH_SIZE, epochs=epochs, callbacks=[cbk])
        # save the model after it successfully trained
        wgan.save_weights("models/wgan_model_all", save_format='tf')
        print("finished and saved")
    except KeyboardInterrupt:
        print('\n Interrupted')
        # save the weights of the gan so you can continue training after the model is
        # finished
        if "y" in input("\n Do you want to save the current model? \n Answer: [y] [n]"):
            wgan.save_weights("models/wgan_model_all", save_format='tf')
            print("saved registered")

        wgan.save_weights("models/model_unregistered_all", save_format='tf')
        print("saved unregistered")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)