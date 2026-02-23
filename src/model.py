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
        
        # set up the model to start training process
        self.compile_model()
    

    def create_model(self):
        '''
        This function creates a CNN model for trash classification
        It assigns the created model to the class's "model" property

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
        model.add(layers.Dense(8, activation='softmax'))

        self.model = model


    
    def compile_model(self):
        '''
        This function creates and compiles a model, preparing it for training and testing

        Returns:
            model: the CNN model
        '''

        # define the model's architecture
        self.create_model()

        # configure the model
        self.model.compile(
            optimizer=optimizers.Adam(self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['acc']
        )



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
    
    

    def plot_history(self, model_history):
        # plot training and validation curves (Accuracy and Loss)
        acc = model_history.history['acc']
        val_acc = model_history.history['val_acc']
        loss = model_history.history['loss']
        val_loss = model_history.history['val_loss']
        epoch_num = range(1, len(acc) + 1)
        plt.figure(figsize = (10, 3))
        
        # train and validation accuracy
        plt.subplot(121)
        plt.plot(epoch_num, acc, 'b', label='Training accuracy')
        plt.plot(epoch_num, val_acc, 'r', label='Validation accuracy')
        plt.title('Training and Validation accuracy')
        plt.legend()

        plt.subplot(122)

        # train and validation loss
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

        # print metrics
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