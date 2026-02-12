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
DATA_DIR = "TrashNet/"
CATEGORIES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

def standardized_image(img_path, img_size):
    """
    Given an image path, standardize the dimensions of an image

    Args:
        img_path: str - the path to the image to be standardized
        img_size: tuple(int, int) - a tuple of (width, height) for the standardized image dimensions
    Returns:
        new_array: list - the standardized image as an array with dimensions img_size
    """
    img_array = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

    # TODO: replace this functionality
    # right now, we are just using cv2 to resize the image via stretching, but try experimenting with different image standardization techniques
    # some examples: cropping, padding
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



def split_data(train_ratio, test_ratio, val_ratio, data_x, data_y):
    """
    Given train/test/validation ratios, split data_x and data_y according to those ratios

    Args:
        train_ratio, test_ratio, val_ratio: int - percentage of data to be split into training, testing, and validation
        data_x: list - list of input features (images)
        data_y: list - list of the image's corresponding labels
    Returns:
        xTrain, xTest, xVal, yTrain, yTest, yVal: list - the splitted datasets
    """
    assert(train_ratio >= 0 and test_ratio >= 0 and val_ratio >= 0)
    assert(train_ratio + test_ratio + val_ratio == 100)

    # TODO: split the data according to the ratios, and return the split data as shown in the return statement
    # the train_test_split function only splits into two groups, so we ned to call this function twice to split into three groups
    # this fix just requires a bit of math
    # alternatively, you can implement the splitting yourself without the sklearn function (remember to shuffle the data before splitting!)

    state_num = 42

    # split between train and test
    xTrain, xTest, yTrain, yTest = train_test_split(
        data_x,
        data_y,
        test_size = 0.2, # TODO: fix here
        shuffle=True,
        random_state = state_num
    )

    # TODO: split between train and validation (or test and validation, whichever you see fit)
    if val_ratio != 0: # val_ratio = 0 => no validation set needed
        xTest, xVal, yTest, yVal = train_test_split(
            xTest,
            yTest,
            test_size = 0.5, # fix here
            shuffle=True,
            random_state = state_num
        )
    else:
        xVal, yVal = [], []

    return xTrain, xTest, xVal, yTrain, yTest, yVal

def augment_data(xTrain, yTrain):
    '''
    Given training data, augment the training data with a data generator's transformations

    Args:
        xTrain: list - list of input features (images) for the training set
        yTrain: list - list of the image's corresponding labels for the training set
    '''


if __name__ == "__main__":
    data_array_x, data_array_y = create_data(NORMALIZED_IMAGE_SIZE)

    # complete the conversion from category names -> numeric labels (integers) -> one-hot-encoding (0s and 1s)
    encoder = LabelEncoder()
    encoder.fit(data_array_y)
    encoded_Y = encoder.transform(data_array_y)
    dummy_y = to_categorical(encoded_Y) # convert to one-hot-encoding labels


    # split data into train/test/val sets
    xTrain, xTest, xVal, yTrain, yTest, yVal = split_data(
        80, 10, 10,
        np.asarray(data_array_x),
        dummy_y
    )

    def create_data_generator(data_x, data_y, batch_size=32, only_normalize=False):
        '''
        Given data_x and data_y, set up a data generator for the data

        Args:
            data_x: list - list of input features (images)
            data_y: list - list of the image's corresponding labels
            batch_size: int - the number of images to be yielded from the generator per batch
            only_normalize: bool - whether to only apply rescaling to the data, or to apply other transformations as well
        Returns:
            datagen: ImageDataGenerator - the data generator with the specified transformations, after attaching the datasets to it
        '''

        if only_normalize:
            # only normalize the data (used for testing/validation data)
            datagen = ImageDataGenerator(rescale=1./255)
        else:
            # else, apply other transformations
            datagen = ImageDataGenerator(
                # TODO: experiment with different transformations and their parameters, and see what works best
                # you can also try other transformations not listed here

                rescale=1./255,     #rescale the data in the range [0,1] from the varying range of each picture such that contribution in the loss value due to each picture can be equicalent
                shear_range=0.2,    #Basically think of it as converting a square to a rhombus
                zoom_range=0.2,     #Randomly zooming inside pictures
                # width_shift_range=0.2, #It actually shift the image to the left or right(horizontal shifts). If the value is float and <=1 it will take the percentage of total width as range. Suppose image width is 100px. if width_shift_range = 1.0 it will take -100% to +100% means -100px to +100px. It will shift image randomly between this range. Randomly selected positive value will shift the image to the right side and negative value will shift the image to the left side.
                rotation_range=10,
                # height_shift_range=0.2, #shifts the image vertically
                # horizontal_flip=True, #flips both rows and columns horizontally
                # vertical_flip=True, #flips both rows and columns vertically
            )
        
        # flow() hooks up the training data, which tells the data generator which data to apply transformations to
        generator = datagen.flow(data_x, data_y, batch_size=batch_size, shuffle=True)
        return generator

    # set up generators to use later for training and evaluation
    train_generator = create_data_generator(xTrain, yTrain, batch_size=32, only_normalize=False)
    test_generator = create_data_generator(xTest, yTest, batch_size=32, only_normalize=True)
    val_generator = create_data_generator(xVal, yVal, batch_size=32, only_normalize=True)

# up to here now

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