from tensorflow import keras
from tensorflow.keras import layers

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

def get_generator_model(noise_dim = 128):
    noise = layers.Input(shape=(noise_dim,))
    x = layers.Dense(8 * 8 * 512, use_bias=False)(noise)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Reshape((8, 8, 512))(x)
    x = upsample_block(
        x,
        512,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        use_bias=False,
        use_bn=False,
        padding="same",
        use_dropout=False,
    )
    x = upsample_block(
        x,
        256,
        layers.LeakyReLU(0.2),
        strides=(1, 1),
        up_size = (1,1),
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
