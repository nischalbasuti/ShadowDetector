#!/usr/bin/env python
import os

import numpy as np
import cv2 as cv

# Disable annoying messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import backend as K

from random import shuffle

class Patched_CNN(object):
    """
    Implementation based on the 'Patched CNN' architecture from
    http://chenpingyu.org/docs/yago_eccv2016.pdf (page 9).
    This implementation differs in that it accepts an arbitrary number of
    channels for the input, which is defined when build_model() is called.
    """
    def __init__(self):
        super(Patched_CNN, self).__init__()
        self.model = Sequential()

    def load_model(self, filepath='model.h5', custom_objects=None, compile=True):
        """ Load a model from 'filepath'. """
        self.model = keras.models.load_model(filepath, custom_objects, compile)

    def save_model(self, filepath='model.h5'):
        """ Save the current model as 'filepath'. """
        self.model.save(filepath)

    def build_model(self, channels=3, padding="valid", size=32, **kwargs):
        """ Initialize an untrained model.
            :channels: number of channels of input, i.e. sizexsizexchannels.
            :padding: valid or same
            :size: input dimension size, like size of image.
        """

        # 32x32x4 -> conv1 -> 30x30x50
        self.model.add( Conv2D( 50, activation="relu", kernel_size=(3, 3),
                                input_shape=(size, size, channels), padding=padding ))

        # 30x30x50 -> conv2 -> 28x28x50
        self.model.add( Conv2D( 50, kernel_size=(3, 3), activation="relu", padding=padding ) )

        # 28x28x50 -> pool1 -> 14x14x50
        self.model.add( MaxPooling2D( pool_size=(2, 2), strides=1))

        # 14x14x50 -> conv3 -> 12x12x50
        self.model.add( Conv2D( 50, kernel_size=(3, 3), activation="relu", padding=padding ) )

        # 12x12x50 -> conv4 -> 10x10x50
        self.model.add( Conv2D( 50, kernel_size=(3, 3), activation="relu", padding=padding ) )

        # 10x10x50 -> conv5 -> 10x10x30
        self.model.add( Conv2D( 30, kernel_size=(1, 1), activation="relu", padding=padding ) )

        # 10x10x30 -> pool2 -> 5x5x30
        self.model.add( MaxPooling2D( pool_size=(2, 2), strides=1 ))

        # 5x5x30 -> conv6 -> 3x3x50
        self.model.add( Conv2D( 50, kernel_size=(3, 3), activation="relu", padding=padding ) )

        # 3x3x50 -> flatten -> 450
        self.model.add( Flatten() )

        # 450 -> Dense Layer -> 1
        # self.model.add( Dense( 450, activation="relu"  ) )
        # self.model.add( Dropout(0.5) )
        self.model.add( Dense(size*size, activation="sigmoid") )

        # self.model.add( Dense(1, activation="sigmoid") )

        self.model.compile(
                # keras.optimizers.SGD(0.01),
                keras.optimizers.RMSprop(0.0001),
                loss=keras.losses.binary_crossentropy,
                metrics=["acc", "mae"]
                )

    def train(self, image_segments, labels,
            batch_size=32, epochs=10, patience=2, prefix="", **kwargs):
        """ Train a model using a arrays of input and output. """
        self.model.fit(
                np.array(image_segments),
                np.array(labels),
                batch_size,
                epochs,
                validation_split=0.20,
                callbacks=[
                        EarlyStopping(patience=patience),
                        ModelCheckpoint(
                            "./checkpoints/"+prefix+"model.{epoch:02d}-{val_acc:.2f}.hdf5",
                            monitor="val_acc",
                            verbose=1,
                            save_best_only=True)
                    ]
                )
        return

    def test(self, image_segments=None, labels=None):
        images = open_images("./data/SBU-Test/ShadowImages", None)
        shadow_masks = open_images("./data/SBU-Test/ShadowMasks", None, True)

        x = [] # input features.
        y = [] # labels

        x.extend(images)
        y.extend(shadow_masks)
        ret = self.model.evaluate(
                x=np.array(x),
                y=np.array(y),
                batch_size=None,
                verbose=1,
                sample_weight=None,
                steps=None)

        print("Evaluation loss, accuracy, mean absolute error:", ret)
        return ret


    def predict(self, img):
        return self.model.predict(img)

def open_images(path, max=None, mask=False, size=(32, 32)):
    """ Read images under the directory given in 'path'. """
    print("- Loading images from", path)
    for (dirpath, dirnames, filenames) in os.walk(path):
        if max is not None:
            # shuffle(filenames)
            filenames = filenames[:max]
            print("-- limited to reading", max, "files.")

        if mask:
            images = [
                        cv.resize(cv.imread(os.path.join(path, fname), cv.IMREAD_UNCHANGED)/255.0, size).flatten()
                        for fname in filenames
                    ]

        else:
            images = [
                        cv.resize(
                            cv.cvtColor(cv.imread(os.path.join(path, fname), cv.IMREAD_UNCHANGED), cv.COLOR_BGR2LAB),
                            size)
                        for fname in filenames
                    ]
    # print(filenames)
    print("- Finished loading images from", path)
    cv.waitKey()
    return images

if __name__ == '__main__':

    # Learn for image.
    images = open_images("./data/SBUTrain4KRecoveredSmall/ShadowImages", 4070)
    shadow_masks = open_images("./data/SBUTrain4KRecoveredSmall/ShadowMasks", 4070, True)

    x = [] # input features.
    y = [] # labels

    x.extend(images)
    y.extend(shadow_masks)

    prior_cnn = Patched_CNN()
    prior_cnn.build_model(channels=3)
    prior_cnn.train(
            x, y,
            batch_size=20,
            epochs=100,
            patience=5,
            prefix="prior")
    prior_cnn.save_model("./prior_model.h5")

#     # Learn for each segment.
#     patch_cnn = Patched_CNN(4)

#     shadows = open_images("./segments/shadows")
#     non_shadows = open_images("./segments/non_shadows", len(shadows))

#     x = [] # input features.
#     y = [] # labels

#     x.extend(shadows)
#     x.extend(non_shadows)

#     y.extend([ 1 for i in range(len(shadows)) ])
#     y.extend([ 0 for i in range(len(non_shadows)) ])

#     patch_cnn = Patched_CNN()
#     patch_cnn.build_model(channels=4)
#     patch_cnn.train(x, y, 100, 10, 2, "patch")
#     patch_cnn.save_model("patch_model.h5")

