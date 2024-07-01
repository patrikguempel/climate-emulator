import tensorflow as tf
print(tf.__version__)
from tensorflow.python import keras


def evaluate(modelName: str):
    model: keras.Model = keras.models.load_model(f"./models/{modelName}/model.h5")

    model.summary()
    # model.evaluate()


evaluate("attempt1")
