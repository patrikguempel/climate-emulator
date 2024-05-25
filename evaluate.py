import keras
from keras.models import load_model


def evaluate(modelName: str):
    model: keras.Model = load_model(f"./models/{modelName}/model.h5")

    model.summary()
    # model.evaluate()


evaluate("attempt1")
