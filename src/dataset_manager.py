import os
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import numpy as np
from collections import Counter
from pathlib import Path
from logger import NullLogger


class DatasetManager:
    '''
    This class takes in a path to the dataset and a target image size, and outputs a standardized dataset
    '''

    def __init__(self, dataset_path, img_size, logger=None):
        # set up properties/fields
        self.dataset_path: str = dataset_path
        self.img_size: tuple[int, int] = img_size

        # data_x and data_y are arrays of images and their corresponding labels: {category_name: [images]}, labels_by_category: {category_name: [one-hot labels]}
        self.data_x = {}
        self.data_y = {}

        # same thing for the train/test/val datasets
        self.train_data_x, self.test_data_x, self.val_data_x = [], [], []
        self.train_data_y, self.test_data_y, self.val_data_y = [], [], []

        self.logger = logger or NullLogger()
        self.logger.log_message(f"Dataset is set to {Path(self.dataset_path).name}")
        self.logger.log_message(f"Dimensions of the dataset images will be {self.img_size}")

    def update_dataset_path(self, new_dataset_path):
        """
        Sets a new path for the dataset manager to load data from
        Only needed if you want to reuse the same dataset manager for multiple datasets
        Alternatively, you can just initialize a new dataset manager for each dataset
        """
        self.dataset_path = new_dataset_path
        self.logger.log_message(f"Dataset has been updated to {Path(self.dataset_path).name}")

    

    def get_category_name_from_index(self, index):
        """
        Returns the name of the category at the given index
        Since one hot encoding is used, model predictions use the index of the category instead of the category name itself
        This function returns the category name for easier interpretability
        Args:
            index: the index of the category
        Returns:
            category_name: the name of the category
        """
        return self.get_categories()[index]



    def get_categories(self):
        """
        Returns an array of category names based on the folder names in the dataset_path
        """

        categories = []
        for i in os.listdir(self.dataset_path):
            # if the file is hidden, skip it
            if i.startswith('.'):
                continue
            # else, the file is a directory, so add it to the categories list
            path = os.path.join(self.dataset_path, i)
            if os.path.isdir(path) and i not in categories:
                # add the category to the categories list
                categories.append(i)
        return categories

    

    def standardize_image(self, img_path, standardization_func=None):
        """
        Given an image path, standardize the dimensions of an image to img_size

        Args:
            img_path: str - the path to the image to be standardized
            standardization_func: function - the function to be used to standardize the image
            standardization_func should take in an image array and return a standardized image array
        Returns:
            new_array: list - the standardized image as an array with dimensions img_size
        """
        img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # normalize to [0, 1] for consistent training
        # img_array = img_array.astype(np.float32) / 255.0
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        img_array = cv2.resize(img_array, self.img_size)

        if standardization_func is not None:
            img_array = standardization_func(img_array)

        # TODO: replace/update this functionality
        # right now, we are just using cv2 to resize the image via stretching, but try experimenting with different image standardization techniques
        # some examples: cropping, padding, other things inside the image standardization research doc
        return img_array
    


    def load_data(self):
        """
        Given the dataset_path, standardize the images and load them with their labels into two dicts (data_x and data_y)
        """
        dataset_categories = self.get_categories()

        # reset the data
        self.data_x, self.data_y = {}, {}
        encoder = LabelEncoder()
        encoder.fit(range(len(dataset_categories)))
        for category in tqdm(dataset_categories):
            category_path = os.path.join(self.dataset_path, category)
            category_num = dataset_categories.index(category) # get a unique number for each category
            data_array_x, data_array_y = [], []
            for img in os.listdir(category_path):
                img_path = os.path.join(category_path, img)
                try:
                    standardized_img_array = self.standardize_image(img_path)
                except:
                    print(f"Error loading image from: {img_path}, skipping this...")
                    continue
                data_array_x.append(standardized_img_array)
                data_array_y.append(category_num)
            self.data_x[category] = np.asarray(data_array_x)
            encoded_array_y = encoder.transform(data_array_y)
            one_hot_encoded_array_y = to_categorical(encoded_array_y, num_classes=len(dataset_categories))
            self.data_y[category] = one_hot_encoded_array_y

        total_images = sum(len(imgs) for imgs in self.data_x.values())
        self.logger.log_message(f"\nLoaded {total_images} images from {Path(self.dataset_path).name}")
        parts = []
        for i in range(len(dataset_categories)):
            count = len(self.data_x[self.get_category_name_from_index(i)])
            parts.append(f"\t{self.get_category_name_from_index(i)}: {count} images")
        self.logger.log_message(f"Number of images per category:\n" + "\n".join(parts))
    


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

        # reset arrays
        xTrain, xTest, xVal, yTrain, yTest, yVal = [], [], [], [], [], []
        for category in self.get_categories():
            category_data_x = self.data_x.get(category, np.array([]))
            category_data_y = self.data_y.get(category, np.array([]))
            if len(category_data_x) == 0:
                continue

            xTrain_cat, xTest_cat, yTrain_cat, yTest_cat = train_test_split(
                category_data_x, category_data_y,
                test_size=(test_ratio + val_ratio) / 100,
                shuffle=True,
                random_state=state_num
            )

            if val_ratio != 0: # val_ratio = 0 means that no validation set needed
                xTest_cat, xVal_cat, yTest_cat, yVal_cat = train_test_split(
                    xTest_cat, yTest_cat,
                    test_size=val_ratio / (test_ratio + val_ratio),
                    shuffle=True,
                    random_state=state_num
                )
                xVal.append(xVal_cat)
                yVal.append(yVal_cat)

            xTrain.append(xTrain_cat)
            xTest.append(xTest_cat)
            yTrain.append(yTrain_cat)
            yTest.append(yTest_cat)

        self.train_data_x = np.concatenate(xTrain)
        self.test_data_x = np.concatenate(xTest)
        self.train_data_y = np.concatenate(yTrain)
        self.test_data_y = np.concatenate(yTest)
        self.val_data_x = np.concatenate(xVal)
        self.val_data_y = np.concatenate(yVal)

        # log any important information
        self.logger.log_message(f"\nData is split with train/test/validation ratio of {train_ratio}/{test_ratio}/{val_ratio}")
        self.logger.log_message(f"\tTraining data: {len(self.train_data_x)} images")
        self.logger.log_message(f"\tTesting data: {len(self.test_data_x)} images")
        self.logger.log_message(f"\tValidation data: {len(self.val_data_x)} images")

    

    def calculate_raw_aspect_ratios(self):
        '''
        This function calculates the original aspect ratio of each image in the dataset
        Args:
            None
        Returns:
            None
        '''
        # print smallest and highest aspect ratio in each category
        for category in self.get_categories():
            category_path = os.path.join(self.dataset_path, category)
            if not os.path.isdir(category_path):
                continue
            # only include non-hidden files
            images = [f for f in os.listdir(category_path) if not f.startswith('.')]
            aspect_ratios = []
            for img_name in images:
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    aspect_ratios.append((w / h, w, h, img_name))
            if aspect_ratios:
                min_ar = min(aspect_ratios, key=lambda x: x[0])
                max_ar = max(aspect_ratios, key=lambda x: x[0])
                print(f"{category}:")
                print(f"\tmin: {min_ar[1]}x{min_ar[2]} (aspect ratio {min_ar[0]:.2f}) [file name: {min_ar[3]}]")
                print(f"\tmax: {max_ar[1]}x{max_ar[2]} (aspect ratio {max_ar[0]:.2f}) [file name: {max_ar[3]}]")