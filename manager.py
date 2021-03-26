import tensorflow as tf

class Manger:
    def __init__(self, model: tf.keras.Model, train_params, data, callbacks):
        self.model = model
        self.train_params = train_params
        self.data = data
        self.callbacks = self.get_callbacks(callbacks)

    def setup(self):
        pass

    def train(self):
        self.model.fit(
            x=self.data,
            y=None,
            batch_size=None,
            epochs=1,
            verbose=1,
            callbacks=None
        )

    def get_callbacks(self, callbacks):
        cb = []
        if "ModelCheckpoint" in callbacks:
            cb.append(
                tf.keras.callbacks.ModelCheckpoint(**callbacks["ModelCheckpoint"])
            )
        if "TensorBoard" in callbacks:
            cb.append(
                tf.keras.callbacks.TensorBoard(**callbacks["TensorBoard"])
            )
        return cb