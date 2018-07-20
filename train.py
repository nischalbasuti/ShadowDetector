#!/usr/bin/env python
from patched_cnn import *

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
        prefix="complete_images_")
