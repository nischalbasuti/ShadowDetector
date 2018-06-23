import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import xml.etree.ElementTree
import copy
# import pymeanshift.build.lib.linux-x86_64-3.6.pymeanshift
import pymeanshift as pms

class ImageProcessor(object):
    """docstring for ImageProcessor"""
    def __init__(self, image, filename = None):
        super(ImageProcessor, self).__init__()
        self.image = image
        self.imageShadow = copy.deepcopy(image)
        self.shadowPoints = []
        self.segments = []
        
        if filename is not None:
            # Find the shadow points from the corresponding xml file.
            # TODO: use regex to change extension
            xmlFileName = str(filename[:-3]) + "xml"
            print("xml file:", xmlFileName)
            e = xml.etree.ElementTree.parse("./xml/"+xmlFileName).getroot()

            for tag in e.findall("pt"):
                point = ( int(float(tag.get("x"))), int(float(tag.get("y"))) )
                cv.circle(self.imageShadow, point, 3, (0, 255, 0))
                self.shadowPoints.append( point )
            print("shadow points found:", len(self.shadowPoints))
            # cv.imshow("img shadow", self.imageShadow)
            # cv.imshow("img", self.image)

        # segment the image.
        cv.imshow("img", self._segment(1))
        cv.waitKey()

    def _segment(self, flag=0):
        if flag == 0:
            # code from: 
            #     https://docs.opencv.org/3.3.1/d3/db4/tutorial_py_watershed.html

            segmented_image = copy.deepcopy(self.image)

            gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
            ret, thresh = cv.threshold(gray, 0, 255,
                    cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
            # noise removal
            kernel = np.ones((3,3), np.uint8)
            opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations = 2)
            # sure background area
            sure_bg = cv.dilate(opening, kernel, iterations=3)
            # Finding sure foreground area
            dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
            ret, sure_fg = cv.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
            # Finding unknown region
            sure_fg = np.uint8(sure_fg)
            unknown = cv.subtract(sure_bg, sure_fg)

            # Marker labelling
            ret, markers = cv.connectedComponents(sure_fg)
            # Add one to all labels so that sure background is not 0, but 1
            markers = markers+1
            # Now, mark the region of unknown with zero
            markers[unknown==255] = 0

            markers = cv.watershed(segmented_image, markers)
            segmented_image[markers == -1] = [255,0,0]
        elif flag == 1:
            # Using mean shift implementation from https://github.com/fjean/pymeanshift

            (segmented_image, labels_image, number_regions) = pms.segment(
                    self.image, spatial_radius = 6, range_radius = 4.5, min_density = 50
                    )
        elif flag == 2:
            segmented_image = cv.pyrMeanShiftFiltering(self.image, 50, 50)

        # for point in labels_image:
        #     print(point[0])
        for col in range(self.image.shape[0]):
            for row in range(self.image.shape[1]):
                if labels_image[col, row] == labels_image[450, 200]:
                    segmented_image[col, row][0] = 0 # b 
                    segmented_image[col, row][1] = 255 # g
                    segmented_image[col, row][2] = 0 # r

        print("number_regions: ", number_regions)
        print("source image:", self.image.shape)
        return segmented_image[:,:]
        
        
# Read images from directory
images = []
path = "./img"
for (dirpath, dirnames, filenames) in os.walk(path):
    images.extend([[cv.imread(os.path.join(path, fname)), fname] for fname in filenames])
    break

for image in images:
    print("size: %s" % str(image[0].shape))
    print("name: %s" % str(image[1]))
    processed = ImageProcessor(image[0], image[1])
    break # break after one iteration

# cv.imshow("img", images[0][0])
# cv.waitKey()
