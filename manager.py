import tensorflow as tf
from utils import default_value

class Manger:
    def __init__(self, model: tf.keras.Model, train_params, data, callbacks):
        self.model = model
        self.train_params = train_params
        self.data = data
        self.try_set_callbacks(callbacks)

    def setup(self):
        pass

    def train(self, x=None, y=None, batch_size=None, epochs=None, verbose=None, callbacks=None):
        self.try_set_callbacks(callbacks)
        self.model.fit(
            x=default_value(self.data.trainings_data, x),
            y=default_value(self.data.validation_data, y),
            batch_size=default_value,
            epochs=default_value,
            verbose=default_value,
            callbacks=self.callbacks
        )

    def try_set_callbacks(self, callbacks):
        # if self.callbacks was never initialized we will do so.
        if not hasattr(self, 'callbacks'):
            self.callbacks = None

        # if no callbacks specified we might want to use the potentially already set callbacks and thus not overwrite them.
        if callbacks is None:
            return

        cb = []
        if "ModelCheckpoint" in callbacks:
            cb.append(
                tf.keras.callbacks.ModelCheckpoint(**callbacks["ModelCheckpoint"])
            )
        if "TensorBoard" in callbacks:
            cb.append(
                tf.keras.callbacks.TensorBoard(**callbacks["TensorBoard"])
            )

        self.callbacks = cb