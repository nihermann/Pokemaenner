from dcgan_GAN_monitor import GANMonitor
from dcgan_GAN import GAN
from dcgan_models import load_models
import tensorflow as tf
from tensorflow import keras

if __name__ == '__main__':

    MODEL_PATH = "models/dcgan_model"
    IMG_PATH = "preprocessing/data"
    IMG_SAVE_PATH = "results/generated_img_{epoch}_{i}.png"

    # Hyperparamters
    IMG_SHAPE = (64, 64, 3)
    BATCH_SIZE = 32
    LATENT_DIM = 128
    EPOCHS = 2

    # preparing the dataset
    dataset = keras.preprocessing.image_dataset_from_directory(IMG_PATH, label_mode=None,
                                                               image_size=(IMG_SHAPE[0], IMG_SHAPE[1]),
                                                               batch_size=BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(lambda x: x / 255.0)

    discriminator, generator = load_models()
    # initialize the gan with the generator and the discriminator as well as the
    # given size of dimension in the latent space
    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=LATENT_DIM)

    # compile the model with Adam optimizer and the Binary Crossentropy loss function
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    # if you want to load the saved model:
    try:
        gan.load_weights(MODEL_PATH)
    except:
        print("No weight available yet")

    # fit the model with the overridden training function
    gan.fit(
        dataset, epochs=EPOCHS, callbacks=[GANMonitor(num_img=10, latent_dim=LATENT_DIM, IMG_PATH=IMG_SAVE_PATH)]
    )

    # save the weights of the gan so you can continue training after the model is
    # finished
    gan.save_weights(MODEL_PATH, save_format='tf')
