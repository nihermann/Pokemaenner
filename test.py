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


class A:
    def __init__(self, a):
        self.a = a

    def add(self, x, y):
        return self(x, y)

    def __call__(self, a, b):
        return a - b + self.a

class B:
    def __init__(self, b=2):
        self.b = b
    def subtract(self, x, y):
        return x - y

    def __call__(self, a, b):
        return a + b + self.b

def add_attr(a, b, name):
    attr = getattr(a, name)
    
    def f(x, y):
        return f.func(b, x, y)
    
    f.__setattr__("func", attr)
    b.__setattr__(name, f)

def add_attr2(a, b, name):
    import types
    attr = getattr(a, name)
    setattr(b, name, types.MethodType(attr, b))


if __name__ == '__main__':
    # mlp = MLP()
    #
    # x = np.random.random((1000, 16))
    # y = np.ones((1000, 2))
    # y[:, 1] = 0
    #
    # mlp.fit(x, y, epochs=20)
    from utils import transfer_method
    a = A(10)
    b = B()
    # attr = getattr(A, "add")
    # b = transfer_method("add", A, b)
    # b.__setattr__("add", attr)
    add_attr2(A, b, "add")
    print()

