from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import load_model

def create_model_plot(model_name: str):
    model = load_model(f"models/{model_name}/model.h5")
    plot_model(model, to_file=f"models/{model_name}/model_plot.png", show_shapes=True, show_layer_names=True)

def create_model_plots(models):
    for model in models:
        create_model_plot(model)

create_model_plot("mlp2")