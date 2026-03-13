import sys
import os
import cv2
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# the stuff above is for path finding to the proper imports, don't worry about it too much



from dataset_manager import DatasetManager
from model import GarbageClassificationModel
from transfer_learning_models.resnet50_model import ResNet50Model
from transfer_learning_models.mobilenetv3_model import MobileNetV3Model
from transfer_learning_models.nasnetmobile_model import NASNetMobileModel
from logger import Logger



if __name__ == "__main__":
    DATASET_DIR = "datasets/dataset_vers1"
    NORMALIZED_IMAGE_SIZE = (150, 150)
    MODEL_NAMES = [
        "benchmark_model",
        "resnet50_model",
        "mobilenetv3_model",
        "nasnetmobile_model"
    ] # this acts as the export name
    MODEL_CLASSES = [
        GarbageClassificationModel, 
        ResNet50Model, 
        MobileNetV3Model, 
        NASNetMobileModel
    ]
    NUM_EPOCHS = [50, 25, 25, 25]
    
    for model_name, model_class, num_epochs in zip(MODEL_NAMES, MODEL_CLASSES, NUM_EPOCHS):
        # set up all variables
        # (this order is important!)
        logger = Logger(model_name)
        dl = DatasetManager(DATASET_DIR, NORMALIZED_IMAGE_SIZE, logger=logger)
        m = model_class(dl, logger=logger)

        # prepare the dataset for the model to use
        dl.load_data()
        dl.split_data(60, 20, 20)

        # train the model
        m.load_model_from_file(f"model_registry/{model_name}.keras")
        # model_history = m.train_model(
        #     epochs=num_epochs,
        #     batch_size=32,
        #     export_path=f"model_registry/{model_name}.keras"
        # )
        # m.plot_history(model_history)
        m.measure_metrics()
        # m.predict_img("datasets/benchmark_dataset/organics/organics_1.jpg")
        m.random_preds(3, 6)
        m.random_preds(8, 4)
        m.random_preds(10, 10)
    
    # TODO: for tomorrow, try loading in all the trained models and make sure that's actually possible to do