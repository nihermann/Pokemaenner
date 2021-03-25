import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Flatten, Concatenate, Reshape, UpSampling2D, Conv2D, Activation,
                                     LeakyReLU, GaussianNoise, GaussianDropout, LayerNormalization)
import numpy as np


class MyMLP(tf.keras.Model):
    def __init__(self):
        inp = Input((16,))
        out = Dense(2, activation='sigmoid')(inp)
        super(MyMLP, self).__init__(inp, out)

        self.compile(
            "adam",
            loss=["mse", "mse"],
            loss_weights=[5, 1]
        )

    def train(self, x, y):
        with tf.GradientTape() as tape:
            pred = self(x)
            loss = self.compiled_loss(y, pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss


class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = MyMLP()

        self.compile()
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def train_step(self, inp):
        x, y = inp

        mlp_loss = self.mlp.train(x, y)  # Forward pass

        # Update the metrics.
        # Metrics are configured in `compile()`.
        self.loss_tracker.update_state(mlp_loss)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, x):
        return self.mlp(x)


class Manager():
    def __init__(self, model):
        self.model = model

    def train(self, *args, **kwargs):
        self.model.compile()
        self.model.fit(self, *args, **kwargs)


def linear(x, y):
    return y


if __name__ == '__main__':
    mlp = MLP()

    x = np.random.random((1000, 16))
    y = np.ones((1000, 2))
    y[:, 1] = 0

    mlp.fit(x, y, epochs=20)

    # m = Manager(mlp)
    # m.forward_step(x, y, epochs=3)
