import sys
import os
import cv2
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
# the stuff above is for path finding to the proper imports, don't worry about it too much




from dataset_manager import DatasetManager
from model import GarbageClassificationModel
from transfer_learning_models.resnet50_model import ResNet50Model



if __name__ == "__main__":
    # DATASET_DIR_1 = "datasets/benchmark_dataset/"
    # NORMALIZED_IMAGE_SIZE_1 = (150, 150)

    # using thebase model architecture =============================================================================
    # initialize the dataset loader and get the split datasets
    # dl = DatasetManager(DATASET_DIR_1, NORMALIZED_IMAGE_SIZE_1)
    # dl.load_data()
    # dl.split_data(80, 10, 10)

    # # # the base model (not good but also not bad, ~60% accuracy at 25 epochs)
    # m = GarbageClassificationModel(dl)
    # model_history = m.train_model(
    #     epochs=25,
    #     batch_size=32,
    #     export_path="model_registry/benchmark_model_2.keras"
    # )
    # m.plot_history(model_history)
    # m.measure_metrics()

    # using transfer learning (ResNet50 in this example) =============================================================================
    DATASET_DIR_2 = "datasets/benchmark_dataset/"
    NORMALIZED_IMAGE_SIZE_2 = (150, 150)
    
    dl2 = DatasetManager(DATASET_DIR_2, NORMALIZED_IMAGE_SIZE_2)
    dl2.load_data()
    dl2.split_data(80, 10, 10)

    m2 = ResNet50Model(dl2)
    model_history = m2.train_model(
        epochs=10,
        batch_size=32,
        export_path="model_registry/resnet_model.keras"
    )
    m2.plot_history(model_history)
    m2.measure_metrics()

    # print smallest and highest aspect ratio in each category
    # EXPLORE_DATASET_DIR = "datasets/benchmark_dataset/"
    # EXPLORE_DATASET_DIR = "datasets/dataset_vers1/"
    # EXPLORE_NORMALIZED_IMAGE_SIZE = (150, 150)
    # dl = DatasetManager(EXPLORE_DATASET_DIR, EXPLORE_NORMALIZED_IMAGE_SIZE)
    # for category in dl.get_categories():
    #     category_path = os.path.join(EXPLORE_DATASET_DIR, category)
    #     if not os.path.isdir(category_path):
    #         continue
    #     images = [f for f in os.listdir(category_path) if not f.startswith('.')]
    #     aspect_ratios = []
    #     for img_name in images:
    #         img_path = os.path.join(category_path, img_name)
    #         img = cv2.imread(img_path)
    #         if img is not None:
    #             h, w = img.shape[:2]
    #             aspect_ratios.append((w / h, w, h, img_name))
    #     if aspect_ratios:
    #         min_ar = min(aspect_ratios, key=lambda x: x[0])
    #         max_ar = max(aspect_ratios, key=lambda x: x[0])
    #         print(f"{category}:")
    #         print(f"\tmin: {min_ar[1]}x{min_ar[2]} (aspect ratio {min_ar[0]:.2f}) [file name: {min_ar[3]}]")
    #         print(f"\tmax: {max_ar[1]}x{max_ar[2]} (aspect ratio {max_ar[0]:.2f}) [file name: {max_ar[3]}]")



    # d = DatasetManager("datasets/dataset_vers1/", (150, 150))

    # def standardize_image_1(img_array):
    #     return cv2.resize(img_array, (150, 150))

    # i = d.standardize_image("datasets/dataset_vers1/cardboard/cardboard_9.jpg", standardize_image_1)
    # plt.imshow(i)
    # plt.axis("off")
    # plt.title("Image")
    # plt.show()