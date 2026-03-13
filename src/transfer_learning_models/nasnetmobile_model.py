from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras import layers, models

from model import GarbageClassificationModel

class NASNetMobileModel(GarbageClassificationModel):
    def create_model(self):
        dm = self.dataset_manager
        
        pretrained_model = NASNetMobile(
            include_top=False,
            weights='imagenet',
            input_shape=(dm.img_size[0], dm.img_size[1], 3)
        )
        pretrained_model.trainable = False

        model = models.Sequential()
        model.add(pretrained_model)
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))
        num_categories = len(self.dataset_manager.get_categories())
        model.add(layers.Dense(num_categories, activation="softmax"))


        self.model = model