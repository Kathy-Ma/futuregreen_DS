import sys
import os
import cv2
import numpy as np
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
    
    # ================================================================ TRAINING THE MODELS ================================================================
    
    DATASET_DIR = "datasets/dataset_vers2/"
    NORMALIZED_IMAGE_SIZE = (200, 200)

    model_classes = [MobileNetV3Model, NASNetMobileModel]
    image_sizes = [224]
    
    for model_class in model_classes:
        for image_dim in image_sizes: # decided on 224x224 for the image size (it works the best)
            img_size = (image_dim, image_dim)
            export_name = f"{model_class.__name__}_imgsize_{img_size}"
            logger = Logger(export_name)
            dl = DatasetManager(DATASET_DIR, img_size, logger=logger)
            m = model_class(dl, logger=logger)
            dl.load_data()
            dl.split_data(60, 20, 20)
            model_history = m.train_model(
                epochs=25,
                batch_size=32,
                export_path=f"model_registry/{export_name}.keras"
            )
            m.plot_history(model_history)
            m.measure_metrics()
            m.predict_img("datasets/dataset_vers2/organic/organic_1.jpg")
            m.random_preds(3, 6)
            m.random_preds(8, 4)



    # ================================================================ SETTING UP BAR GRAPHS ================================================================

    # # set up the metrics for the model
    # all_metrics = {}
    # model_classes = [MobileNetV3Model, NASNetMobileModel] # change this as needed
    # image_sizes = [100, 125, 150, 175, 200, 225, 250, 275, 300]
    # for model_class in model_classes:
    #     model_metrics = {}
    #     for image_size in image_sizes:
    #         img_size = (image_size, image_size)
    #         export_name = f"{model_class.__name__}_imgsize_{img_size}"
    #         # logger = Logger(export_name)
    #         dl = DatasetManager(DATASET_DIR, img_size)
    #         m = model_class(dl)
    #         dl.load_data()
    #         dl.split_data(60, 20, 20)
    #         m.load_model_from_file(f"model_registry/{export_name}.keras")

    #         metrics_by_category = {}
    #         for cat in dl.get_categories():
    #             correct, wrong = m.metrics_for_category(cat)
    #             total = correct + wrong
    #             metrics_by_category[cat] = {
    #                 "correct": correct,
    #                 "wrong": wrong,
    #                 "accuracy": correct / total * 100,
    #             }
    #         model_metrics[img_size] = metrics_by_category
    #     all_metrics[model_class.__name__] = model_metrics

    # # create the bar chart: categories on x-axis, grouped bars for each image size
    # logger = Logger("bar_graphs")
    # for model_class in model_classes:
    #     model_metrics = all_metrics[model_class.__name__]
    #     categories = list(next(iter(model_metrics.values())).keys())
    #     img_sizes = list(model_metrics.keys())
    #     n_categories = len(categories)
    #     n_sizes = len(img_sizes)

    #     x = np.arange(n_categories)
    #     width = 0.8 / n_sizes

    #     fig, ax = plt.subplots(figsize=(12, 6))
    #     for i, img_size in enumerate(img_sizes):
    #         accuracies = [model_metrics[img_size][cat]["accuracy"] for cat in categories]
    #         offset = (i - n_sizes / 2 + 0.5) * width
    #         ax.bar(x + offset, accuracies, width, label=f"{img_size[0]}×{img_size[1]}")

    #     ax.set_xlabel("Category")
    #     ax.set_ylabel("Accuracy (%)")
    #     ax.set_title(f"Accuracy by Category and Image Size (for {model_class.__name__})")
    #     ax.set_xticks(x)
    #     ax.set_xticklabels(categories, rotation=45, ha="right")
    #     ax.legend(title="Image size")
    #     ax.set_ylim(0, 105)
    #     plt.tight_layout()
    #     logger.save_figure(f"{model_class.__name__}_bar_graph.png")
