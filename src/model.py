import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model
import time
from pathlib import Path

from dataset_manager import DatasetManager
from logger import NullLogger
# ----------------------------------------------------------------

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import load_model
# heatmap of confusion matrix
import seaborn as sn
import pandas as pd



class GarbageClassificationModel:
    def __init__(self, dataset_manager, logger=None):
        self.dataset_manager: DatasetManager = dataset_manager
        self.learning_rate: float = 0.001
        
        # set up the model to start training process
        self.compile_model()

        self.logger = logger or NullLogger()
        self.logger.log_message(f"\nThe model's name is {self.__class__.__name__}")
    

    def create_model(self):
        '''
        This function creates a CNN model for trash classification
        It assigns the created model to the class's "model" property

        Returns:
            model: the CNN model
        '''

        dm = self.dataset_manager
        num_categories = len(dm.get_categories())

        # define the model's architecture
        model = models.Sequential()
        model.add(layers.Input(shape=(dm.img_size[0], dm.img_size[1], 3)))

        model.add(layers.Conv2D(32, (3,3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(64, (3,3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Conv2D(128, (3,3), activation='relu'))
        model.add(layers.MaxPooling2D((2,2)))
        model.add(layers.Dropout(0.5))

        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(0.5))
        model.add(layers.Dense(num_categories, activation='softmax'))

        self.model = model



    def load_model_from_file(self, model_path):
        '''
        This function loads a model from a file, and assigns it to the class's "model" property
        Args:
            model_path: the path to the model file
        Returns:
            None
        '''
        self.model = load_model(model_path)

        self.logger.log_message(f"\nModel is loaded from {Path(model_path).name}")


    
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
            metrics=['accuracy']
        )



    def train_model(self, epochs=50, batch_size=32, export_path=None):
        # if the file already exists, 
        if export_path is not None and os.path.exists(export_path):
            self.logger.log_message(f"The model could not be trained...")
            self.logger.log_message(f"The model will be saved at {export_path} but this file path already exists, so try again with a different location.")
            raise FileExistsError(f"Your model will be saved at the location {export_path}, but it already exists. Please choose a different location.")

        # if the validation loss doesn't improve after 5 epochs, stop training
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # set up generators to use later for training and evaluation
        dm = self.dataset_manager

        start_time = time.time()
        # train the model
        model_history = self.model.fit(
            dm.train_data_x,
            dm.train_data_y,
            validation_data=(dm.val_data_x, dm.val_data_y),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop]
        )
        end_time = time.time()
        print(f'Training time: {end_time - start_time:.2f} seconds')
        print(f'Training time: {(end_time - start_time)/60:.2f} mins')
        
        self.logger.log_message(f"\nModel is trained for {epochs} epochs")
        total_seconds = end_time - start_time
        minutes, seconds = divmod(int(total_seconds), 60)
        self.logger.log_message(f"\tTraining time: {minutes}min {seconds}sec")

        if export_path is not None:
            self.model.save(export_path)
            self.logger.log_message(f"\nModel is saved at {Path(export_path).name}")

        return model_history
    
    

    def plot_history(self, model_history):
        # plot training and validation curves (Accuracy and Loss)
        # TensorFlow 2.x uses 'accuracy' not 'acc' - use fallbacks for compatibility
        acc = model_history.history.get('accuracy', model_history.history.get('acc', []))
        val_acc = model_history.history.get('val_accuracy', model_history.history.get('val_acc', []))
        loss = model_history.history.get('loss', [])
        val_loss = model_history.history.get('val_loss', [])
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

        self.logger.save_figure("training_and_validation_history.png")
    


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
        x_test = dm.test_data_x

        start_time = time.time()
        prediction = self.model.predict(x_test)
        y_pred = np.argmax(prediction, axis=1)
        end_time = time.time() - start_time

        # log metrics about the predictions
        self.logger.log_message(f"\nPrediction summary:")
        self.logger.log_message(f"\tTotal images predicted: {len(x_test)}")
        self.logger.log_message(f"\tCorrect: {np.sum(y_true == y_pred)}")
        self.logger.log_message(f"\tWrong: {np.sum(y_true != y_pred)}")
        self.logger.log_message(f"\tTime: {end_time:.2f}s ({end_time/len(x_test):.2f}s per image)")

        # print metrics (or log if possible)
        self.logger.log_message(f"\tAccuracy: {accuracy_score(y_true, y_pred)}")
        self.logger.log_message(f"\nClassification Report:\n{classification_report(y_true, y_pred, target_names=dm.get_categories())}")
        print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
        print(f"\nClassification Report:\n{classification_report(y_true, y_pred, target_names=dm.get_categories())}")


        dataset_catgories = self.dataset_manager.get_categories()
        cm = confusion_matrix(y_true, y_pred, labels=range(len(dataset_catgories)))

        # use dataframe to plot the confusion matrix
        df_cm = pd.DataFrame(cm, index=dataset_catgories, columns=dataset_catgories)
        plt.figure(figsize=(8, 6))
        sn.heatmap(df_cm, annot=True, cmap="Blues")
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')

        self.logger.save_figure("confusion_matrix.png")
    


    def predict_img(self, img_path):
        '''
        This function predicts the class of an image
        Args:
            img_path: the path to the image
        Returns:
            None
        '''

        try:
            standardized_img = self.dataset_manager.standardize_image(img_path)
        except Exception as e:
            print(f"Error: we could not standardize the image at {img_path}")
            print(f"Error: {e}")
            return None
        
        prediction = self.model.predict(standardized_img)
        print(prediction)
        