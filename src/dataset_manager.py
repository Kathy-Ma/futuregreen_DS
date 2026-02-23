import os
from sklearn.calibration import LabelEncoder
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np


class DatasetManager:
    '''
    This class takes in a path to the dataset and a target image size, and outputs a standardized dataset
    '''
    def __init__(self, dataset_path, img_size):
        # set up properties/fields
        self.dataset_path: str = dataset_path
        self.img_size: tuple[int, int] = img_size

        # stores the image data and their corresponding labels (initialized as empty lists)
        self.data_x, self.data_y = [], []

        # same thing for the train/test/val datasets
        self.train_data_x, self.test_data_x, self.val_data_x = [], [], []
        self.train_data_y, self.test_data_y, self.val_data_y = [], [], []
    

    def update_dataset_path(self, new_dataset_path):
        """
        Sets a new path for the dataset manager to load data from
        Only needed if you want to reuse the same dataset manager for multiple datasets
        Alternatively, you can just initialize a new dataset manager for each dataset
        """
        self.dataset_path = new_dataset_path

    

    def get_categories(self):
        """
        Returns an array of category names based on the folder names in the dataset_path
        """
        # TODO: replace this hardcoded return value with one that dynamically searches the dataset folder
        # the os library functions could be useful (e.g. os.listdir, os.path.join, e.t.c.)
        return ["cardboard", "glass", "metal", "paper", "plastic", "trash", "organics", "rejected"]

    

    def standardize_image(self, img_path):
        """
        Given an image path, standardize the dimensions of an image to img_size

        Args:
            img_path: str - the path to the image to be standardized
        Returns:
            new_array: list - the standardized image as an array with dimensions img_size
        """
        img_array = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)

        # TODO: replace this functionality
        # right now, we are just using cv2 to resize the image via stretching, but try experimenting with different image standardization techniques
        # some examples: cropping, padding, other things inside the image standardization research doc
        new_array = cv2.resize(img_array, self.img_size)
        return new_array
    


    def load_data(self):
        """
        Given the dataset_path, standardize the images and load them with their labels into two arrays (data_x and data_y)
        """
        data_array_x, data_array_y = [], []
        dataset_categories = self.get_categories()
        for category in tqdm(dataset_categories):
            category_path = os.path.join(self.dataset_path, category)
            category_num = dataset_categories.index(category) # get unique number for each category
            for img in os.listdir(category_path):
                img_path = os.path.join(category_path, img)
                standardized_img_array = self.standardize_image(img_path)
                data_array_x.append(standardized_img_array)
                data_array_y.append(category_num)
        
        # convert from category names -> numeric labels (integers) -> one-hot-encoding (0s and 1s)
        encoder = LabelEncoder()
        encoder.fit(data_array_y)
        encoded_y = encoder.transform(data_array_y)
        one_hot_encoded_y = to_categorical(encoded_y) # convert to one-hot-encoding labels
        
        # load in the data from dataset_path
        # data_y is one-hot-encoded (0s and 1s)
        self.data_x = data_array_x
        self.data_y = one_hot_encoded_y
    


    def split_data(self, train_ratio, test_ratio, val_ratio, state_num=42):
        """
        Given train/test/validation ratios, split data_x and data_y according to those ratios
        We require that train_ratio + test_ratio + val_ratio = 100

        Args:
            train_ratio, test_ratio, val_ratio: int - percentage of data to be split into training, testing, and validation
        Returns:
            xTrain, xTest, xVal, yTrain, yTest, yVal: list - the splitted datasets
        """
        assert(train_ratio >= 0 and test_ratio >= 0 and val_ratio >= 0)
        assert(train_ratio + test_ratio + val_ratio == 100)

        # TODO: split the data according to the ratios, and return the split data as shown in the return statement
        # the train_test_split function only splits into two groups, so we ned to call this function twice to split into three groups
        # this fix just requires a bit of math
        # alternatively, you can implement the splitting yourself without the sklearn function

        # split between train and test
        xTrain, xTest, yTrain, yTest = train_test_split(
            np.asarray(self.data_x),
            self.data_y,
            test_size=0.8, # TODO: fix here
            shuffle=True,
            random_state = state_num
        )

        # TODO: split between train and validation (or test and validation, whichever you see fit)
        if val_ratio != 0: # val_ratio = 0 means that no validation set needed
            xTest, xVal, yTest, yVal = train_test_split(
                xTest,
                yTest,
                test_size=0.5, # fix here
                shuffle=True,
                random_state = state_num
            )
        else:
            xVal, yVal = [], []
        
        # assign the split data to the dataset manager's variables
        self.train_data_x = xTrain
        self.test_data_x = xTest
        self.val_data_x = xVal
        self.train_data_y = yTrain
        self.test_data_y = yTest
        self.val_data_y = yVal