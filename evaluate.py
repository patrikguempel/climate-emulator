import tensorflow as tf
from tensorflow.python import keras
from pathlib import Path


def evaluate(modelName: str):
    model: keras.Model = keras.models.load_model(f"./models/own/{modelName}/model.h5")

    model.summary()
    # model.evaluate()


evaluate("attempt1")

#model_path = Path('models/own/attempt1/model.h5')

# Überprüfen, ob die Datei existiert
#import h5py

#try:
#    with h5py.File(model_path, 'r') as f:
#        print("Die HDF5-Datei ist gültig und kann geöffnet werden.")
#except Exception as e:
#    print(f"Fehler beim Öffnen der HDF5-Datei: {e}")