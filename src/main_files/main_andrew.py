import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset_manager import DatasetManager
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score,  ConfusionMatrixDisplay
from transfer_learning_models.nasnetmobile_model import NASNetMobileModel

if __name__ == "__main__":
    DATASET_DIR = "datasets/benchmark_dataset/"
    NORMALIZED_IMAGE_SIZE = (224, 224)

    dl = DatasetManager(DATASET_DIR, NORMALIZED_IMAGE_SIZE)
    dl.load_data()
    dl.split_data(80, 10, 10)

    m = NASNetMobileModel(dl)
    model_history = m.train_model(
        epochs=10,
        batch_size=32,
        export_path="model_registry/nasnetmobile_model.keras"
    )
    m.plot_history(model_history)
    m.measure_metrics()

    # NUM_CLASSES = dl.train_data_y.shape[1]
    # print(f"Number of classes detected: {NUM_CLASSES}")

    # base_model = NASNetMobile(
    #     input_shape=(*NORMALIZED_IMAGE_SIZE, 3),
    #     include_top=False,
    #     weights="imagenet"
    # )
    # base_model.trainable = False

    # model = models.Sequential([
    #     base_model,
    #     layers.GlobalAveragePooling2D(),
    #     layers.Dense(128, activation="relu"),
    #     layers.Dropout(0.3),
    #     layers.Dense(NUM_CLASSES, activation="softmax")
    # ])

    # model.compile(
    #     optimizer="adam",
    #     loss="categorical_crossentropy",
    #     metrics=["accuracy"]
    # )

    # history = model.fit(
    #     dl.train_data_x,
    #     dl.train_data_y,
    #     validation_data=(dl.val_data_x, dl.val_data_y),
    #     epochs=10,
    #     batch_size=32
    # )

    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(history.history["accuracy"], label="Train Accuracy")
    # plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    # plt.title("Accuracy")
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(history.history["loss"], label="Train Loss")
    # plt.plot(history.history["val_loss"], label="Val Loss")
    # plt.title("Loss")
    # plt.legend()

    # # Get predicted and true labels from test set
    # y_pred = model.predict(dl.test_data_x)
    # y_pred_classes = np.argmax(y_pred, axis=1)
    # y_true_classes = np.argmax(dl.test_data_y, axis=1)

    # # Get category names from dataset folder for readable axis labels
    # class_names = dl.get_categories()

    # # Build and plot the confusion matrix
    # cm = confusion_matrix(y_true_classes, y_pred_classes)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # print("Accuracy:", accuracy_score(y_true_classes, y_pred_classes))
    # print("\nClassification Report:\n", classification_report(y_true_classes, y_pred_classes))
    # print("\nConfusion Matrix:\n", cm)

    # fig, ax = plt.subplots(figsize=(10, 10))
    # disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
    # plt.title("Confusion Matrix")
    # plt.tight_layout()
    # plt.show()