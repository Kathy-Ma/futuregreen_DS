import os
from sklearn.preprocessing import LabelEncoder
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
        return sorted(categories)

    

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
        Given the dataset_path, standardize the images and load them with their labels into two arrays (data_x and data_y)
        """

        # TODO: split the data evenly between each category
        data_array_x, data_array_y = [], []
        dataset_categories = self.get_categories()
        for category in tqdm(dataset_categories):
            category_path = os.path.join(self.dataset_path, category)
            category_num = dataset_categories.index(category) # get unique number for each category
            for img in os.listdir(category_path):
                img_path = os.path.join(category_path, img)
                try:
                    standardized_img_array = self.standardize_image(img_path)
                except:
                    print(f"Error loading image from: {img_path}, skipping this...")
                    continue
                data_array_x.append(standardized_img_array)
                data_array_y.append(category_num)
        
        # convert from category names -> numeric labels (integers) -> one-hot-encoding (0s and 1s)
        encoder = LabelEncoder()
        encoded_y = encoder.fit_transform(data_array_y)
        one_hot_encoded_y = to_categorical(encoded_y) # convert to one-hot-encoding labels
        
        # load in the data from dataset_path
        # data_y is one-hot-encoded (0s and 1s)
        # Images are normalized in standardize_image()
        self.data_x = np.asarray(data_array_x)
        self.data_y = np.asarray(one_hot_encoded_y)
    


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

        # split between train and test
        xTrain, xTest, yTrain, yTest = train_test_split(
            np.asarray(self.data_x),
            self.data_y,
            test_size=(test_ratio + val_ratio)/100,
            shuffle=True,
            random_state = state_num
        )

        if val_ratio != 0: # val_ratio = 0 means that no validation set needed
            xTest, xVal, yTest, yVal = train_test_split(
                xTest,
                yTest,
                test_size=val_ratio/(test_ratio+val_ratio), # finds ratio between both values
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