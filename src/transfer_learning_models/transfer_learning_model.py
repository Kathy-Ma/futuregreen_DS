from keras.applications import ResNet50
from tensorflow.keras import layers, models, optimizers


from model import GarbageClassificationModel

class TransferLearningModel(GarbageClassificationModel):
    def create_model(self):
        pretrained_model = ResNet50(
            include_top=False,
            weights='imagenet', 
            input_shape=(224, 224, 3)
        )
        pretrained_model.trainable = False

        model = models.Sequential()
        model.add(pretrained_model)
        model.add(layers.Flatten())
        model.add(layers.Dropout(0.5))
        num_categories = len(self.dataset_manager.get_categories())
        model.add(layers.Dense(num_categories, activation="softmax"))

        self.model = model