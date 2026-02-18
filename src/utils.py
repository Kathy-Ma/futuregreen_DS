from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_data_generator(data_x, data_y, batch_size=32, only_normalize=True):
    '''
    Given data_x and data_y, set up a data generator for the data

    Args:
        data_x: list - list of input features (images)
        data_y: list - list of the image's corresponding labels
        batch_size: int - the number of images to be yielded from the generator per batch
        only_normalize: bool - whether to only apply rescaling to the data, or to apply other transformations as well
    Returns:
        datagen: ImageDataGenerator - the data generator with the specified transformations, after attaching the datasets to it
    '''

    if only_normalize:
        # only normalize the data (used for testing/validation data)
        datagen = ImageDataGenerator(rescale=1./255)
    else:
        # else, apply other transformations
        datagen = ImageDataGenerator(
            # TODO: experiment with different transformations and their parameters, and see what works best
            # you can also try other transformations not listed here

            rescale=1./255, # rescale the data in the range [0,1]
            shear_range=0.2, # basically think of it as converting a square to a rhombus
            zoom_range=0.2, # randomly zooming inside pictures
            rotation_range=10,
            # height_shift_range=0.2, # shifts the image vertically
            # horizontal_flip=True, # flips both rows and columns horizontally
            # vertical_flip=True, # flips both rows and columns vertically
        )
    
    # flow() hooks up the data, telling the data generator which data it should apply transformations to
    generator = datagen.flow(data_x, data_y, batch_size=batch_size, shuffle=True)
    return generator