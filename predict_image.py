#!/usr/bin/env python
import os
import numpy as np
import cv2 as cv

import patched_cnn as pcnn
import image_processor as ip

img = cv.imread("./img/static_outdoor_anchorage_alaska_usa-img_8999.jpg")
img_processed = ip.ImageProcessed(img, "flickr-4292988620_cjon.jpg")
img_processed._segment()

cnn = pcnn.Patched_CNN()
cnn.load_model("./checkpoints/weights.01-0.95.hdf5")
cnn.load_model("./model.h5")
segments = np.array(img_processed.get_segment_images())
print(segments.shape)
pred = cnn.predict(segments)

for i in range(len(img_processed.segments)):
    img_processed.segments[i]['is_shadow'] = pred[i] == 1

img_processed.showShadows()
