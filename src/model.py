import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
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
    # Override in subclasses; used when creating DatasetManager so data is preprocessed at load time
    preprocess_input_func = None

    def __init__(self, dataset_manager, logger=None):
        self.dataset_manager: DatasetManager = dataset_manager
        
        # assign model-specific preprocessing to dataset manager (used during load_data)
        # we assign this model's preprocess_input_func to the dataset manager's preprocess_input_func
        self.dataset_manager.preprocess_input_func = self.__class__.preprocess_input_func

        self.learning_rate: float = 0.001
        
        # set up the model to start training process
        self.compile_model()

        self.logger = logger or NullLogger()
        self.logger.log_message(f"\nThe model's name is {self.__class__.__name__}")
        if self.dataset_manager.preprocess_input_func is not None:
            self.logger.log_message(f"Using builtin, model-specific preprocess_input function to pre-process images")
        else:
            self.logger.log_message(f"Pixel values will be in range [0, 255]")

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
        # train the model (data already preprocessed in DatasetManager)
        model_history = self.model.fit(
            dm.train_data_x,
            dm.train_data_y,
            validation_data=(dm.val_data_x, dm.val_data_y),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop]
        )
        end_time = time.time()
        
        # log if early stopping happened
        actual_epochs_run = len(model_history.history['loss'])
        if actual_epochs_run < epochs:
            self.logger.log_message(f"\nModel was planned to be trained for {epochs} epochs, but stopped at {actual_epochs_run} epochs to avoid overfitting")
        else:
            self.logger.log_message(f"\nModel is trained for {epochs} epochs")

        # also log the time it took to train the model
        total_seconds = end_time - start_time
        minutes, seconds = divmod(int(total_seconds), 60)
        self.logger.log_message(f"\tTraining time: {minutes} min {seconds} sec")
        secs_per_epoch = total_seconds / actual_epochs_run
        mins_per_epoch, secs_per_epoch = divmod(int(secs_per_epoch), 60)
        self.logger.log_message(f"\t{mins_per_epoch}min {secs_per_epoch}sec per epoch")

        # export the model if a path is defined
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
        self.logger.log_message(f"\tTime: {end_time:.2f}s ({end_time/len(x_test):.4f}s per image)")

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

    def metrics_for_category(self, category_name):
        """
        Get correct and wrong prediction counts for a single category.
        Returns (correct, wrong) for images whose true label is category_name.
        """
        dm = self.dataset_manager
        categories = dm.get_categories()
        if category_name not in categories:
            raise ValueError(f"Unknown category: {category_name}. Available: {categories}")
        cat_index = categories.index(category_name)

        y_true = np.argmax(dm.test_data_y, axis=1)
        mask = y_true == cat_index
        if not np.any(mask):
            return 0, 0

        x_subset = dm.test_data_x[mask]
        prediction = self.model.predict(x_subset, verbose=0)
        y_pred = np.argmax(prediction, axis=1)

        correct = np.sum(y_pred == cat_index)
        wrong = len(y_pred) - correct
        return int(correct), int(wrong)

    def predict_img_batch(self, images):
        """Predict on images. Returns (predictions array, list of predicted category names)."""
        predictions = self.model.predict(images, verbose=0)
        pred_indices = np.argmax(predictions, axis=1)
        pred_labels = [self.dataset_manager.get_category_name_from_index(i) for i in pred_indices]
        return predictions, pred_labels

    def _get_text_colour(self, pred_label, true_label):
        color = 'green' if pred_label == true_label else 'red'
        return color

    def predict_img(self, img_path):
        '''
        This function predicts the class of an image
        Args:
            img_path: the path to the image
        Returns:
            None
        '''
        # first, try to load the raw image (pre-processed)
        raw_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if raw_img is None:
            print(f"Error: could not load image at {img_path}")
            return
            # convert to RGB
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        # the true label comes from the parent folder
        true_label = Path(img_path).parent.name

        # next, try to standardize the image (post-processed)
        try:
            standardized_img = self.dataset_manager.standardize_image(img_path)
        except Exception as e:
            print(f"Error: we could not standardize the image at {img_path}")
            print(f"Error: {e}")
            return
        # add extra dimension to the image for prediction (doesn't work otherwise)
        batch_img = np.expand_dims(standardized_img, axis=0)


        # do the actual prediction
        prediction = self.model.predict(batch_img)
        predicted_index = np.argmax(prediction, axis=1)[0]
        pred_label = self.dataset_manager.get_category_name_from_index(predicted_index)
        categories = self.dataset_manager.get_categories()
        probabilities = prediction[0]
        result = {cat: float(probabilities[i]) for i, cat in enumerate(categories)}

        text_colour = self._get_text_colour(pred_label, true_label)
        fig, axes = plt.subplots(1, 2, figsize=(10, 8))
        # display the raw image (original from file)
        axes[0].imshow(raw_img)
        axes[0].set_title(f'Raw Image: ({raw_img.shape[0]} x {raw_img.shape[1]})')
        axes[0].axis('off')
        # display the standardized image (resized, no model preprocessing - for correct display)
        display_img = self.dataset_manager.manually_standardize_image(img_path)
        axes[1].imshow(display_img)
        axes[1].set_title(f'Standardized Image ({display_img.shape[0]} x {display_img.shape[1]})')
        axes[1].axis('off')

        fig.suptitle(f"Image: {Path(img_path).name}", fontsize=12)
        fig.text(0.5, 0.10, f"True Label: {true_label}", ha='center', fontsize=10, color='black')
        fig.text(0.5, 0.06, f"Predicted Label: {pred_label}", ha='center', fontsize=10, color=text_colour)

        pred_str = ", ".join(f"{cat}: {p:.1%}" for cat, p in sorted(result.items(), key=lambda x: -x[1]))
        fig.text(0.5, 0.02, f"Predictions: {pred_str}", ha='center', fontsize=9)
        plt.tight_layout(rect=[0, 0.2, 1, 0.95])
        
        self.logger.save_figure(f"pred_{Path(img_path).name}")


    def random_preds(self, rows=2, columns=5):
        """
        Display predictions on random test images. Green title = correct, red = incorrect.
        """
        dm = self.dataset_manager
        x_test = dm.test_data_x
        y_test = dm.test_data_y

        if len(x_test) == 0:
            print("No test data available.")
            return

        num_samples = min(rows * columns, len(x_test))
        random_indices = random.sample(range(len(x_test)), num_samples)

        sample_images = x_test[random_indices]
        sample_true_labels = np.argmax(y_test[random_indices], axis=1)
        sample_paths = [dm.test_data_paths[j] for j in random_indices] if getattr(dm, 'test_data_paths', None) else None
        predictions, pred_labels = self.predict_img_batch(sample_images)

        fig, axes = plt.subplots(rows, columns, figsize=(4 * columns, 4 * rows))
        if rows == 1 and columns == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif columns == 1:
            axes = axes.reshape(-1, 1)

        for i in range(num_samples):
            row, col = i // columns, i % columns
            ax = axes[row, col]
            true_cat = dm.get_category_name_from_index(sample_true_labels[i])
            color = self._get_text_colour(pred_labels[i], true_cat)
            top2_idx = np.argsort(predictions[i])[-2:][::-1]
            top2_str = ", ".join(f"{dm.get_category_name_from_index(j)}: {predictions[i][j]:.0%}" for j in top2_idx)
            img_name = Path(sample_paths[i]).name if sample_paths else f"Image {i}"
            ax.set_title(
                f"{img_name}\nPred: {pred_labels[i]} | True: {true_cat}\n{top2_str}",
                color=color,
                fontsize=10
            )
            # use raw image for display (preprocessed images look wrong in imshow)
            ax.imshow(dm.manually_standardize_image(sample_paths[i]) if sample_paths else sample_images[i])
            ax.axis('off')

        for i in range(num_samples, rows * columns):
            axes.flat[i].axis('off')

        fig.suptitle("Predictions on Random Test Images", fontsize=14)
        plt.tight_layout()
        self.logger.save_figure(f"random_preds_{rows}x{columns}.png")