from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers, models

from model import GarbageClassificationModel

class ResNet50Model(GarbageClassificationModel):
    def __init__(self, dataset_manager, logger=None):
        super().__init__(dataset_manager, logger=logger)
        # ResNet50 expects [0,255] as its image data, so don't rescale the image's pixel values
        self.dataset_manager.rescale_pixel_values = False

    def create_model(self):
        dm = self.dataset_manager

        # loads in the pretrained ResNet50 model
        # include_top=False means that we don't want to include the top layer (the fully connected layer)
        # weights='imagenet' means that we want to use the ImageNet weights (we can change this to other weights)
        # input_shape is the image size that the model will take in
        #   remember that the DatasetManager has a property called img_size, which indicates the size that we transform the dataset images to!
        #   so, we will define this as the input size for our transfer learning model
        pretrained_model = ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=(dm.img_size[0], dm.img_size[1], 3)
        )
        pretrained_model.trainable = False



        # define and construct the model architecture
        model = models.Sequential()

        # # ResNet expects [0,255] as its image data
        # so, make sure that the DatasetManager does not divide pixel values by 255

        # we add ResNet50 to the model architecture
        model.add(pretrained_model)

        # this step is important!
        # recall that ResNet50 is a CNN, so its inputs and outputs are tensors (i.e. 3D arrays)
        # since we still want this model to behave like a classifier, we need to flatten the output of ResNet50 (i.e. convert it into a 1D array)
        # you can think of GlobalAveragePooling2D as a type of flattening
        model.add(layers.GlobalAveragePooling2D())
        # this is an intermediate layer, where the model could learn a useful representation before classification
        model.add(layers.Dense(256, activation='relu'))
        # just an extra little fun thing that you can keep or not (it is good for reducing overfitting though)
        model.add(layers.Dropout(0.5))
        # we can also get the number of categories in our dataset by calling the DatasetManager's get_categories() function
        # Important: make sure that the number of categories is the same as the number of categories in the dataset
        # we do this because this output will be our prediction (i.e. is it glass? paper? plastic? e.t.c.)
        num_categories = len(self.dataset_manager.get_categories())
        model.add(layers.Dense(num_categories, activation="softmax"))

        # finally, we assign this ResNet50 pre-trained model to the class's model property
        self.model = model