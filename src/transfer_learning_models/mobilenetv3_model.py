from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras import layers, models, optimizers


from model import GarbageClassificationModel

class MobileNetV3Model(GarbageClassificationModel):
    def create_model(self):
        dm = self.dataset_manager

        # loads MobileNetV3 model
        pretrained_model = MobileNetV3Small(
        include_top=False,
        weights='imagenet',
        input_shape=(dm.img_size[0], dm.img_size[1], 3)
        )
        
        pretrained_model.trainable = False

        model = models.Sequential()
        # MobileNetV3 expects [0,255] + ImageNet preprocessing; data is [0,1] from DatasetManager
        model.add(layers.Lambda(lambda x: preprocess_input(x * 255.0), input_shape=(dm.img_size[0], dm.img_size[1], 3)))
        # add MobileNetV3 to model architecture
        model.add(pretrained_model)
        # flatten output
        model.add(layers.GlobalAveragePooling2D())
        
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dropout(0.5))

        num_categories = len(self.dataset_manager.get_categories())
        model.add(layers.Dense(num_categories, activation="softmax"))

        # assign MobileNetV3 to the class's model property
        self.model = model
