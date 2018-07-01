import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

class Patched_CNN(object):
    """
    Implementation of 'Patched CNN' architecture from
    http://chenpingyu.org/docs/yago_eccv2016.pdf (page 9)

    CNN that accepts 4 channels; Red, Green, Blue and Patch.
    Here, Patch is meant to be a mask which specifies if the pixel is part the
    the segment being classified, or just from a neighbouring segment.
    """
    def __init__(self):
        super(Patched_CNN, self).__init__()
        self.model = Sequential()

    def load_model(self, filepath='model.h5', custom_objects=None, compile=True):
        self.model = keras.models.load_model(filepath, custom_objects, compile)

    def save_model(self, filepath='model.h5'):
        self.model.save(filepath)

    def build_model(self):

        # 32x32x4 -> conv1 -> 30x30x50
        self.model.add( Conv2D( 50, activation="relu", kernel_size=(3, 3),
                                input_shape=(32, 32, 3) ))

        # 30x30x50 -> conv2 -> 28x28x50
        self.model.add( Conv2D( 50, kernel_size=(3, 3), activation="relu" ) )

        # 28x28x50 -> pool1 -> 14x14x50
        self.model.add( MaxPooling2D( pool_size=(2, 2), strides=1))

        # 14x14x50 -> conv3 -> 12x12x50
        self.model.add( Conv2D( 50, kernel_size=(3, 3), activation="relu" ) )

        # 12x12x50 -> conv4 -> 10x10x50
        self.model.add( Conv2D( 50, kernel_size=(3, 3), activation="relu" ) )

        # 10x10x50 -> conv5 -> 10x10x30
        self.model.add( Conv2D( 30, kernel_size=(1, 1), activation="relu" ) )

        # 10x10x30 -> pool2 -> 5x5x30
        self.model.add( MaxPooling2D( pool_size=(2, 2), strides=1 ))

        # 5x5x30 -> conv6 -> 3x3x50
        self.model.add( Conv2D( 50, kernel_size=(3, 3), activation="relu" ) )

        # 3x3x50 -> flatten -> 450
        self.model.add( Flatten() )

        # 450 -> Dense Layer -> 1
        self.model.add( Dense( 450, activation="relu"  ) )
        # self.model.add( Dropout(0.5) )
        self.model.add( Dense( 1, activation="sigmoid") )


        self.model.compile(
                "rmsprop",
                loss=keras.losses.binary_crossentropy,
                metrics=['accuracy']
                )

    def train(self, image_segments, labels, batch_size=32, epochs=100):
        self.model.fit(image_segments, labels, batch_size, epochs)

    def train_generator(self, train_generator):
        self.model.fit_generator(
                train_generator,
                steps_per_epoch=train_generator.n // train_generator.batch_size,
                epochs=50)

if __name__ == '__main__':
    train_data = []
    train_labels = []
    test_data = []

    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    train_generator = datagen.flow_from_directory(
        "./segments",
        target_size=(32, 32),
        classes=("shadows", "non_shadows"),
        class_mode="binary"
        )

    cnn = Patched_CNN()
    cnn.build_model()
    # # cnn.train(train_x, train_y)
    cnn.train_generator(train_generator)

