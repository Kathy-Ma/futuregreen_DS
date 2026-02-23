import numpy as np
import random
import IPython.display as display
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm



from dataset_manager import DatasetManager
from model import GarbageClassificationModel
from transfer_learning_models.transfer_learning_model import TransferLearningModel

# ----------------------------------------------------------------
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
# heatmap of confusion matrix
import seaborn as sn
import pandas as pd
import utils



if __name__ == "__main__":
    DATASET_DIR = "datasets/benchmark_dataset/"
    NORMALIZED_IMAGE_SIZE = (150, 150)

    # initialize the dataset loader and get the split datasets
    # dl = DatasetManager(DATASET_DIR, NORMALIZED_IMAGE_SIZE)
    # dl.load_data()
    # dl.split_data(80, 10, 10)

    # base model architecture (not that good, ~40% accuracy)
    # m = GarbageClassificationModel(dl)
    # model_history = m.train_model(epochs=20, export_path="model_registry/test.keras")
    # m.plot_history(model_history)
    # m.measure_metrics()

    # using transfer learning
    dl2 = DatasetManager(DATASET_DIR, (224, 224))
    dl2.load_data()
    dl2.split_data(80, 10, 10)
    m2 = TransferLearningModel(dl2)