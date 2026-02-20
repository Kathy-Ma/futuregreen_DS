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
    DATASET_DIR = "TrashNet/"
    NORMALIZED_IMAGE_SIZE = (150, 150)

    # initialize the dataset loader and get the split datasets
    dl = DatasetManager(DATASET_DIR, NORMALIZED_IMAGE_SIZE)
    dl.load_data()
    dl.split_data(50, 25, 25)


    m = GarbageClassificationModel(dl, 0.0001)

    model_history = m.train_model(epochs=1, export_path="model_directory/test.keras")
    m.plot_history(model_history)
    m.measure_metrics()