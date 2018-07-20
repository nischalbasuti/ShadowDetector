#!/usr/bin/env python
from patched_cnn import *
import patched_cnn as p

# TODO: Train with equal number of shadow and non-shadow segments.

# segments = open_images("./segments/images", lab_color=False)
segments = open_images("./segments/images", lab_color=True)
shadow_mask = open_images("./segments/shadow_masks", mask=True)

x = [] # input features.
y = [] # shadow maps.

x.extend(segments)
y.extend(shadow_mask)

cnn = Patched_CNN()
# cnn.build_model(channels=4)
cnn.build_model(channels=3)

cnn.train(x, y, batch_size=300, epochs=100, patience=3, prefix="segment_lab_")
cnn.save_model()
