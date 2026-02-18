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
import time


from dataset_manager import DatasetManager
import utils

# ----------------------------------------------------------------
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
# heatmap of confusion matrix
import seaborn as sn
import pandas as pd



class GarbageClassificationModel:
    def __init__(self, dataset_manager):
        self.dataset_manager: DatasetManager = dataset_manager
        self.learning_rate: float = 0.0001
        self.model: models.Sequential = self.create_model()
    

    def create_model(self):
        '''
        This function creates a CNN model for trash classification

        Returns:
            model: the CNN model
        '''

        # define the model's architecture    
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Conv2D(64, (3,3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.5))
        model.add(layers.Conv2D(128, (3,3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(6, activation='softmax'))

        # configure the model
        model.compile(
            optimizer=optimizers.RMSprop(self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['acc']
        )
    
        return model

    def train_model(self, epochs=50, export_path=None):
        early_stop = EarlyStopping(
            monitor='val_loss',     # Monitor validation loss
            patience=5,             # Number of epochs with no improvement after which training will be stopped
            restore_best_weights=True # Restore weights from the epoch with the best value of the monitored quantity
        )

        # set up generators to use later for training and evaluation
        dm = self.dataset_manager
        train_generator = utils.create_data_generator(dm.train_data_x, dm.train_data_y, only_normalize=False)
        test_generator = utils.create_data_generator(dm.test_data_x, dm.test_data_y)
        val_generator = utils.create_data_generator(dm.val_data_x, dm.val_data_y)

        start_time = time.time()
        model_history_es = self.model.fit(train_generator,
                                steps_per_epoch=len(dm.train_data_x)//32,
                                epochs=epochs,
                                validation_data=val_generator,
                                validation_steps=len(dm.val_data_x)//32,
                                callbacks = [early_stop])
        end_time = time.time()
        print(f'Training time: {end_time - start_time:.2f} seconds')
        print(f'Training time: {(end_time - start_time)/60:.2f} mins')

        if export_path is not None:
            self.model.save(export_path)

        return model_history_es
    


    def predict_img(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR_RGB)
        # reshape
        img = cv2.resize(img, (150, 150))
        # normalize
        img = img / 255.0

        # img.reshape(1, 150, 150, 3) does not work
        img_for_prediction = np.expand_dims(img, axis=0)

        # predict
        prediction = self.model.predict(img_for_prediction)
        y_pred = np.argmax(prediction, axis=1)
        cat_pre = self.dataset_manager.get_categories()[y_pred[0]] # why we use pred[0]?

        plt.figure(figsize=(6, 4))

        # format prediction text
        plt.title(f'Pred: {cat_pre.upper()}', fontsize=10)

        # Display image (Note: image is already scaled to 0-1)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return



    def plot_history(self, model_history):
        # plot training and validation curves (Accuracy and Loss)
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
    def measure_metrics(self):
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
        dm = self.dataset_manager
        y_true = np.argmax(dm.test_data_y, axis=1)
        x_test = dm.test_data_x / 255.0
        prediction = self.model.predict(x_test)
        y_pred = np.argmax(prediction, axis=1)

        # Print metrics
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("\n Classification Report:\n", classification_report(y_true, y_pred))
        print("\n Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

        dataset_catgories = self.dataset_manager.get_categories()
        df_cm = pd.crosstab([dataset_catgories[i] for i in y_true],
                            [dataset_catgories[i] for i in y_pred],
                            rownames=['label'],
                            colnames=['predict'])
        plt.figure(figsize = (6,4))
        sn.heatmap(df_cm, annot=True, cmap="Blues")
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()