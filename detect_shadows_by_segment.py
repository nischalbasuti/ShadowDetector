#!/usr/bin/env python
import os
import argparse

import numpy as np
import cv2 as cv

import patched_cnn as pcnn
import image_processor as ip

import copy


parser = argparse.ArgumentParser()
parser.add_argument("--image", help="path to image.")
parser.add_argument("--model", help="path to trained model h5 file.")
args = parser.parse_args()

image_source = args.image
model_source = args.model


img = cv.imread(image_source)

imgLab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
input = np.array([ cv.resize( imgLab, (32, 32) ) ]) # input to nn should be (1, 32, 32, 3)

cnn = pcnn.Patched_CNN()
cnn.load_model(model_source)
print("...initialized model...")

# pred = cnn.predict(input)
pred = np.zeros(32*32)
shadow_mask = cv.resize( pred.reshape(32, 32), (img.shape[1], img.shape[0]) )

outImg = copy.deepcopy(img)

imgp = ip.ImageProcessed(img, "image_name")
imgp._segment(False)

for segment in  imgp.segments:
    imgLab = cv.cvtColor(segment["image"], cv.COLOR_BGR2LAB)
    # imgLab = segment["image"]
    input = np.array([ cv.resize( imgLab, (32, 32) ) ]) # input to nn should be (1, 32, 32, 3)
    segment_shadow_map = cv.resize(
                cnn.predict(input).reshape(32, 32),
                (segment["image"].shape[1], segment["image"].shape[0])
            )

    print("segment shadow map shape:", segment_shadow_map.shape)
    for col in range(segment["minPoint"][0], segment["maxPoint"][0]):
        for row in range(segment["minPoint"][1], segment["maxPoint"][1]):
            shadow_mask[col, row] = segment_shadow_map[
                                        col - segment["minPoint"][0],
                                        row - segment["minPoint"][1]
                                    ]
imgp.shadow_mask = shadow_mask
imgp.label_shadow_segments(0.1)
imgp.showShadows()

for col in range(outImg.shape[0]):
    for row in range(outImg.shape[1]):
        # if shadow_mask[col, row] > 0.5:
            # outImg[col, row][0] = 255.0 * shadow_mask[col, row]
            # outImg[col, row][0] = outImg[col, row][0] * (1-shadow_mask[col, row])
            outImg[col, row][1] = 255 * (shadow_mask[col, row])
            # outImg[col, row][2] = 255.0 * shadow_mask[col, row]

# cv.namedWindow("original", cv.WINDOW_NORMAL)
cv.namedWindow("mask", cv.WINDOW_NORMAL)

# cv.imshow("original", img)
cv.imshow("mask", outImg)
# # cv.imshow("predicted shadow mask", shadow_mask)
cv.waitKey()

# for i in range(len(img_processed.segments)):
#     img_processed.segments[i]['is_shadow'] = pred[i] == 1

# img_processed.showShadows()
