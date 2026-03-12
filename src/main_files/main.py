import sys
import os
import cv2
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# the stuff above is for path finding to the proper imports, don't worry about it too much




from dataset_manager import DatasetManager
from model import GarbageClassificationModel
from transfer_learning_models.resnet50_model import ResNet50Model
from logger import Logger



if __name__ == "__main__":
    # DATASET_DIR_1 = "datasets/benchmark_dataset/"
    # NORMALIZED_IMAGE_SIZE_1 = (150, 150)

    # using the base model architecture =============================================================================
    # initialize the dataset loader and get the split datasets
    # dl = DatasetManager(DATASET_DIR_1, NORMALIZED_IMAGE_SIZE_1)
    # dl.load_data()
    # dl.split_data(60, 20, 20)

    # # the base model (not good but also not bad, ~60% accuracy at 25 epochs)
    # m = GarbageClassificationModel(dl)
    # model_history = m.train_model(
    #     epochs=25,
    #     batch_size=32,
    #     export_path="model_registry/benchmark_model_2.keras"
    # )
    # m.plot_history(model_history)
    # m.measure_metrics()

    # using transfer learning (ResNet50 in this example) =============================================================================
    # DATASET_DIR_2 = "datasets/dataset_vers1/"
    # NORMALIZED_IMAGE_SIZE_2 = (150, 150)

    # MODEL_PATH_1 = "model_registry/resnet_model.keras"
    
    # dl2 = DatasetManager(DATASET_DIR_2, NORMALIZED_IMAGE_SIZE_2)
    # dl2.load_data()
    # dl2.split_data(60, 20, 20)

    # m2 = ResNet50Model(dl2)
    # model_history = m2.train_model(
    #     epochs=10,
    #     batch_size=32,
    #     export_path=MODEL_PATH_1
    # )
    # m2.plot_history(model_history)
    # m2.measure_metrics()

    # # reload the model
    # m22 = GarbageClassificationModel(dl2)
    # m22.load_model_from_file(MODEL_PATH_1)
    # m22.measure_metrics()

    DATASET_DIR = "datasets/dataset_vers1/"
    NORMALIZED_IMAGE_SIZE = (150, 150)

    logger = Logger("resnet50_model")
    dl = DatasetManager(DATASET_DIR, NORMALIZED_IMAGE_SIZE, logger=logger)
    dl.load_data()
    dl.split_data(60, 20, 20)
    m = ResNet50Model(dl, logger=logger)
    m.load_model_from_file("model_registry/resnet50_model.keras")
    # model_history = m.train_model(
    #     epochs=10,
    #     batch_size=32,
    #     export_path="model_registry/resnet50_model.keras"
    # )
    # m.plot_history(model_history)
    m.measure_metrics()
    m.predict_img("datasets/benchmark_dataset/organics/organics_1.jpg")
    m.random_preds(3, 6)
    m.random_preds(8, 4)