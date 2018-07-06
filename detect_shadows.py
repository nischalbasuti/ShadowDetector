#!/usr/bin/env python
import os
import argparse

import numpy as np
import cv2 as cv

import patched_cnn as pcnn
import image_processor as ip

parser = argparse.ArgumentParser()
parser.add_argument("--image", help="path to image.")
parser.add_argument("--model", help="path to trained model h5 file.")

args = parser.parse_args()
print(args)

image_source = args.image
model_source = args.model

img = cv.imread(image_source)
img_processed = ip.ImageProcessed(img, "image_name")
img_processed._segment()

cnn = pcnn.Patched_CNN()
cnn.load_model(model_source)
segments = np.array(img_processed.get_segment_images())
print(segments.shape)
pred = cnn.predict(segments)

for i in range(len(img_processed.segments)):
    img_processed.segments[i]['is_shadow'] = pred[i] == 1

img_processed.showShadows()
