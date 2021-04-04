from wgan_loading_data import loading_data
from wgan_generator import get_generator_model
from wgan_discriminator import get_discriminator_model
from wgan_WGAN import WGAN, discriminator_loss, generator_loss
from wgan_GAN_monitor import GANMonitor
from tensorflow import keras
import os,sys

if __name__ == '__main__':
    # Hyperparameters
    IMG_SHAPE = (64, 64, 3)
    BATCH_SIZE = 16
    EPOCHS = 2 # Set the number of epochs for training.
    noise_dim = IMG_SHAPE[0]*2 # Size of the noise vector
    
    parent_path = os.path.dirname(os.getcwd())
    MODEL_PATH = "models/wgan_model"
    IMG_PATH = os.path.join(parent_path,"preprocessing/data")
    IMG_SAVE_PATH = "results/generated_img_{epoch}_{i}.png"

    images,labels = loading_data(IMG_PATH, IMG_SHAPE,IMG_SAVE_PATH="data_reshaped_as_array/images", LBL_SAVE_PATH='data_reshaped_as_array/labels')

    # load the discriminator model
    d_model = get_discriminator_model()
    d_model.summary()

    # load the generator model
    g_model = get_generator_model()
    g_model.summary()

    # Instantiate the optimizer for both networks
    # (learning_rate=0.0002, beta_1=0.5 are recommended)
    generator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )
    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )

    # Instantiate the WGAN model.
    wgan = WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=noise_dim,
        discriminator_extra_steps=4,
    )

    # Compile the WGAN model.
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
    )

    # if you want to load the saved model:
    try:
        wgan.load_weights(MODEL_PATH)
    except:
        print("No weight available yet")

    # Instantiate the customer `GANMonitor` Keras callback.
    cbk = GANMonitor(num_img=10, latent_dim=noise_dim, starter_count=152,
                     img_path= IMG_SAVE_PATH,
                     model_path = MODEL_PATH,
                     save_model_every = 2, model=wgan)

    try:
        # Start training the model.
        wgan.fit(images, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[cbk])
        # save the model after it successfully trained
        wgan.save_weights(MODEL_PATH, save_format='tf')
        print("finished and saved")
    except KeyboardInterrupt:
        print('\n Interrupted')
        # save the weights of the gan so you can continue training after the model is
        # finished
        if "y" in input("\n Do you want to save the current model? \n Answer: [y] [n]"):
            wgan.save_weights(MODEL_PATH, save_format='tf')
            print("saved registered")

        wgan.save_weights(MODEL_PATH+"_unregistered", save_format='tf')
        print("saved unregistered")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
