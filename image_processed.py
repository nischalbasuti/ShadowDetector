import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import xml.etree.ElementTree
import copy
import pymeanshift as pms
import scipy.io as sio
import time

class ImageProcessed(object):
    """Opens and image along with it's annotations/labels."""
    def __init__(self, image, filename = None):
        super(ImageProcessed, self).__init__()
        self.filename = filename
        self.image = image
        self.shadowPoints = []
        self.segments = {}          # [ { image_of_segment, is_shadow }, ... ]
        
    def openWithMAT(self):
        # Find the shadow points from the corresponding mat file.
        # TODO: use regex to change extension
        # "./data/train/annt_1.mat"
        matFileName = "./data/train/annt_"+str(self.filename[:-3]) + "mat"
        print("mat file:", matFileName)
        mat = sio.loadmat(matFileName)
        self.image_mask = mat['seg'] # Each point is assigned it's segment's label here.

        # Resizing image because the mask is using different dimensions for some reason.
        self.image = cv.resize( self.image,
                (np.array(mat['im']).shape[1], np.array(mat['im']).shape[0]) )

        self.shadow_regions = mat['allshadow']
        # non_shadow_regions = mat['allnonshadow']
        # numlabel = mat['numlabel'] # TODO: find out what this is.

    def showShadows(self):
        pass

    def show(self):
        # segment the image.
        cv.imshow("original image %s" % self.filename, self.image)
        cv.waitKey()

    def _segment(self):
         ##
         # Segment the image.
        start_time_seconds = time.time()
        print("...started segmenting", self.filename, "...")
        # Using mean shift implementation from https://github.com/fjean/pymeanshift

        segmented_image, labels_image, number_regions = pms.segment(
                                                            self.image,
                                                            spatial_radius = 6,
                                                            range_radius = 4.5,
                                                            min_density = 50)
        # segmented_image, labels_image, number_regions = pms.segment(
        #                                                     self.image,
        #                                                     spatial_radius = 10,
        #                                                     range_radius = 10,
        #                                                     min_density = 300)
        end_time_seconds = time.time()
        print("...finished segmenting", self.filename, "...",
                end_time_seconds - start_time_seconds)


        ##
        # Add pixel points of each segment into the list self.segments.
        start_time_seconds = time.time()
        print("...started gathering points of each segment", self.filename, "...")

        # points are a dict, with columns as keys and list of rows as values.
        #   segments =  [ 
        #                   {
        #                       "points":   {
        #                                       "col": [ row0, row1,... ]
        #                                   }...,
        #                       "maxPoint": [ minX, minY ], 
        #                       "minPoint": [ maxX, maxY ] 
        #                   } ...  
        #               ]
        self.segments = [ {"points": {},"maxPoint": [0,0], "minPoint": [1000,1000]} for i in range(number_regions)]
        # points with 1's are shadows, 0's are not shadows.
        shadow_mask = np.zeros((self.image.shape[0], self.image.shape[1]))
        for col in range(self.image.shape[0]):
            for row in range(self.image.shape[1]):
                if self.image_mask[col, row] in self.shadow_regions:
                    shadow_mask[col, row] = 1
                # store the segments in a list.

                # Find the minimum and maximum row and col values.
                if self.segments[labels_image[col, row]]["minPoint"][0] > col:
                    self.segments[labels_image[col, row]]["minPoint"][0] = col
                if self.segments[labels_image[col, row]]["minPoint"][1] > row:
                    self.segments[labels_image[col, row]]["minPoint"][1] = row

                if self.segments[labels_image[col, row]]["maxPoint"][0] < col:
                    self.segments[labels_image[col, row]]["maxPoint"][0] = col
                if self.segments[labels_image[col, row]]["maxPoint"][1] < row:
                    self.segments[labels_image[col, row]]["maxPoint"][1] = row 

                if col not in self.segments[labels_image[col, row]]['points'].keys():
                    self.segments[labels_image[col, row]]['points'][col] = []

                self.segments[labels_image[col, row]]['points'][col].append(row)

                # TODO: check if shadow by ratio of ones:zeros in the whole segment.
                self.segments[labels_image[col, row]]["is_shadow"] = shadow_mask[col, row] == 1
        end_time_seconds = time.time()
        print("...finished gathering points for each segment...",
                end_time_seconds - start_time_seconds)
        ##
        # Make a seperate segmented images using the points in self.segements.
        start_time_seconds = time.time()
        print("...started constructing segment images...")
        for i in range(len(self.segments)):
            minX = self.segments[i]['minPoint'][0]
            minY = self.segments[i]['minPoint'][1]
            maxX = self.segments[i]['maxPoint'][0]
            maxY = self.segments[i]['maxPoint'][1]

            newImage = np.zeros((maxX-minX, maxY-minY, 3), dtype=self.image.dtype)
            # print("minX, minY:", minX, minY)
            # print("maxX, maxY:",maxX, maxY)
            print()
            for row in range(minY, maxY):
                for col in range(minX, maxX):
                    # if col in self.segments[i]['points'].keys():
                    if row in self.segments[i]['points'][col]:
                            newImage[col-minX, row-minY] = segmented_image[col, row]
                            # # Uncomment to make segment area green.
                            # newImage[col-minX, row-minY] = [0, 255, 0]
                    else:
                        newImage[col-minX, row-minY] = self.image[col, row]

            self.segments[i]["image"] = newImage
            # print(i, self.segments[i]["is_shadow"])
            cv.imshow("individual seg",self.segments[i]["image"])
            cv.waitKey()
        print("...finished constructing segment images...",
                end_time_seconds - start_time_seconds)

        cv.imshow("shadow mask", shadow_mask)
        cv.imshow("segmented_image", segmented_image)

        print("number_regions: ", number_regions)
        print("source image:", self.image.shape)
        return segmented_image[:,:]
        
if __name__ == '__main__':
    processed = ImageProcessed(cv.imread("./data/train/2.png"), "2.png")
    processed.openWithMAT()
    processed._segment()
    processed.show()
