from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras import layers, models

from model import GarbageClassificationModel



class NASNetMobileModel(GarbageClassificationModel):
    def __init__(self, dataset_manager, logger=None):
        super().__init__(dataset_manager, logger=logger)
        # NASNetMobile expects [0,1] as its image data
        self.dataset_manager.rescale_pixel_values = True

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