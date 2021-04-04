from tensorflow import keras
import tensorflow as tf


class GAN(keras.Model):
    """
    Class to define a generative adversarial network combining both
    discriminator and generator.
    """

    def __init__(self, discriminator, generator, latent_dim):
        """
        discriminator and generator are predefines models and latent_dim
        defines the dimension of the latent space
        """
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def _compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        """Returns the loss metric of both discriminator and the generator"""
        return [self.d_loss_metric, self.g_loss_metric]

    # overriding the train_step
    def train_step(self, real_images):
        """Trains both the discriminator and the generator"""
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Decode the vectors from the latent space to fake images by calling
        # the generator
        generated_images = self.generator(random_latent_vectors)

        # Concatenate the real images from the given dataset and the generated ones
        # from the latent space
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        # generated images are labeled with one while real ones with 0
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )

        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            # get the predictions of each image from both the real and generated
            # images
            predictions = self.discriminator(combined_images)
            # compute the loss of the discriminator for the given loss function
            # (vanilla is binary cross entropy) with the correct target (the labels)
            # and the given classification of the discriminator
            d_loss = self.loss_fn(labels, predictions)
        # train the weights (trainable being the ones from the convolutional layers)
        # and apply the optimizer gradients
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images" (0) (the generator hopes to
        # fool the discriminator)
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            # get the predictions of the discriminator for the generated images
            # (whether or not the discriminator fell for the fake ones)
            predictions = self.discriminator(self.generator(random_latent_vectors))
            # compute the loss for only correct labels and the actual times the
            # generator tricks the discriminator
            g_loss = self.loss_fn(misleading_labels, predictions)
        # train the weights of the transposed convolutional layers with gradient
        # tape and the given optimizer
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Update metrics
        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result(),
        }
