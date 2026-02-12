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


# ----------------------------------------------------------------
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
# heatmap of confusion matrix
import seaborn as sn
import pandas as pd


NORMALIZED_IMAGE_SIZE = (150, 150)
DATA_DIR = "../TrashNet/"
CATEGORIES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def standardized_image(img_path, img_size):
    """
    Given an image path, standardize the dimensions of an image
    Returns the image as an array with dimensions img_size and 3 color channels (RGB)
    """
    img_array = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
    new_array = cv2.resize(img_array, img_size)
    return new_array

    
def create_data(img_size):
    data_array_x, data_array_y = [], []
    for category in tqdm(CATEGORIES):
        category_path = os.path.join(DATA_DIR, category)
        category_num = CATEGORIES.index(category) # get unique number for each category
        for img in os.listdir(category_path):
            img_path = os.path.join(category_path, img)
            standardized_img_array = standardized_image(img_path, img_size)
            data_array_x.append(standardized_img_array)
            data_array_y.append(category_num)
    return data_array_x, data_array_y




if __name__ == "__main__":
    data_array_x, data_array_y = create_data(NORMALIZED_IMAGE_SIZE)

    # encode class values (data_array_y)
    encoder = LabelEncoder()
    encoder.fit(data_array_y_150) #learn the mapping between class names and integers
    encoded_Y = encoder.transform(data_array_y_150)
    dummy_y = to_categorical(encoded_Y)


    # using sklearn to split data into Train/Val/Test
    xTrain150, xTest150, yTrain, yTest = train_test_split(
        np.asarray(data_array_x_150),  # convert to NumPy array
        dummy_y,
        test_size = 0.2,
        shuffle=True,
        random_state = 42)

    xTrain150, xVal150, yTrain, yVal = train_test_split(
        xTrain150,
        yTrain,
        test_size = 0.2,
        shuffle=True,
        random_state = 42)


    train_datagen = ImageDataGenerator(
        rescale=1./255,     #rescale the data in the range [0,1] from the varying range of each picture such that contribution in the loss value due to each picture can be equicalent
        shear_range=0.2,    #Basically think of it as converting a square to a rhombus
        zoom_range=0.2,     #Randomly zooming inside pictures
        # width_shift_range=0.2, #It actually shift the image to the left or right(horizontal shifts). If the value is float and <=1 it will take the percentage of total width as range. Suppose image width is 100px. if width_shift_range = 1.0 it will take -100% to +100% means -100px to +100px. It will shift image randomly between this range. Randomly selected positive value will shift the image to the right side and negative value will shift the image to the left side.
        rotation_range=10,
        # height_shift_range=0.2, #shifts the image vertically
        # horizontal_flip=True, #flips both rows and columns horizontally
        # vertical_flip=True, #flips both rows and columns vertically
    )

    train_generator_150 = train_datagen.flow(
        xTrain150,
        yTrain,
        batch_size=32, #No. of images to be yielded from the generator per batch
        shuffle=True
    )

    val_generator_150 = ImageDataGenerator(rescale=1./255).flow(xVal150, yVal, batch_size=32)
    test_generator_150 = ImageDataGenerator(rescale=1./255)

    # plot training and validation curves (Accuracy and Loss)
    def plot_history(model_history):
        acc = model_history.history['acc']
        val_acc = model_history.history['val_acc']
        loss = model_history.history['loss']
        val_loss = model_history.history['val_loss']

        epoch_num = range(1, len(acc) +1)

        plt.figure(figsize = (10, 3))
        #Train and validation accuracy
        plt.subplot(121)
        plt.plot(epoch_num, acc, 'b', label='Training accurarcy')
        plt.plot(epoch_num, val_acc, 'r', label='Validation accurarcy')
        plt.title('Training and Validation accurarcy')
        plt.legend()

        plt.subplot(122)
        #Train and validation loss
        plt.plot(epoch_num, loss, 'b', label='Training loss')
        plt.plot(epoch_num, val_loss, 'r', label='Validation loss')
        plt.title('Training and Validation loss')
        plt.legend()

        plt.show()


# Collect predictions and true labels
    def measure_model(model, xTest, yTest):
        '''
        This function applies 4 tasks:
        1. Output Accuracy
        2. Print Classification Report
        3. Print Confusion Matrix
        4. Plot Confusion Matrix HeatMap

        Args:
            model: The model to be measured.
            xTest: Test data (input features).
            yTest: True one-hot encoded labels for the test data.
        '''
        y_true = np.argmax(yTest, axis=1)
        xTest = xTest / 255.0
        prediction = model.predict(xTest)
        y_pred = np.argmax(prediction, axis=1)

        # Print metrics
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("\n Classification Report:\n", classification_report(y_true, y_pred))
        print("\n Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

        df_cm = pd.crosstab([CATEGORIES[i] for i in y_true],
                            [CATEGORIES[i] for i in y_pred],
                            rownames=['label'],
                            colnames=['predict'])
        plt.figure(figsize = (6,4))
        sn.heatmap(df_cm, annot=True, cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()