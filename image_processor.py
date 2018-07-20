#!/usr/bin/env python
import numpy as np
import cv2 as cv
import xml.etree.ElementTree
import pymeanshift as pms
import scipy.io as sio
import copy
import os
import time

class ImageProcessed(object):
    """Opens and image along with it's annotations/labels."""
    def __init__(self, image, filename = None):
        super(ImageProcessed, self).__init__()
        self.filename = filename
        self.image = image
        self.shadow_points = []
        self.segments = {}          # [ { image_of_segment, is_shadow }, ... ]

        self.mode = "" # String specifying if opening with mat or tappen

    def open_with_mat(self):
        self.mode = "mat"

        # Find the shadow points from the corresponding mat file.
        # TODO: use regex to change extension
        # mat files in the form of: "./data/train/annt_1.mat"
        mat_file_name = "./data/train/annt_"+str(self.filename[:-3]) + "mat"
        print("mat file:", mat_file_name)
        mat = sio.loadmat(mat_file_name)
        self.image_mask = mat['seg'] # Each point is assigned it's segment's label here.

        # Resizing image because the mask is using different dimensions for some reason.
        self.image = cv.resize( self.image,
                (np.array(mat['im']).shape[1], np.array(mat['im']).shape[0]) )

        self.shadow_regions = mat['allshadow']
        self._segment()

    def open_with_tappen(self):
        self.mode="tappen"

        # "./data/tappen/gt/image_name.png"
        self.shadow_mask = cv.imread("./data/tappen/gt/"+str(self.filename[:-3]) + "png")[:,:,1] / 255
        print(self.shadow_mask.shape)
        print("./data/tappen/gt/"+self.filename)
        # print("gt file:", self.shadow_mask)

        # Make sure image and mask are of the same size
        if self.image.shape != self.shadow_mask.shape:
            self.image = cv.resize( self.image, ( self.shadow_mask.shape[1], self.shadow_mask.shape[0] ) )
        self._segment()

    def showShadows(self):
        shadow_image = copy.deepcopy(self.image)

        for i in range(len(self.segments)):
            if self.segments[i]["is_shadow"] == True:
                for col in self.segments[i]["points"].keys():
                    rows = self.segments[i]["points"][col]
                    for row in rows:
                        shadow_image[col, row] = (0, 255, 0)
        cv.imshow("shadow image", shadow_image)
        cv.imshow("original image", self.image)
        cv.imshow("segmented image", self.segmented_image)
        cv.waitKey()

    def show(self):
        # segment the image.
        cv.imshow("original image %s" % self.filename, self.image)
        cv.waitKey()

    def _segment(self, label_shadows=True , threshold=0):
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
        # Gather points of each segment.
        self._set_segment_points(labels_image, number_regions)
        # Construct image of the segment and save to disk.
        self._make_segment_images(segmented_image, label_shadows)

        self.segmented_image = segmented_image

        return segmented_image

    def _set_segment_points(self, labels_image, number_regions):
        """
        Add pixel points of each segment into the list self.segments.
        Also constructs self.shadow_mask.
        """

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
        self.segments = [ { "points": {},
                            "maxPoint": [0,0],
                            "minPoint": [1000,1000]
                          } for i in range(number_regions)]
        # Initialize  a shadow mask, points with shadows will be 1 and points 
        # without shadows will be 0.
        if self.mode == "mat":
            self.shadow_mask = np.zeros((self.image.shape[0], self.image.shape[1]))

        # This loop will:
        # 1. Construct the shadow mask.
        # 2. Find the minimum and maximum row and column values.
        # 3. Find the points that are inside the segment.
        for col in range(self.image.shape[0]):
            for row in range(self.image.shape[1]):

                if self.mode == "mat":
                    # Make the shadow mask if mat.
                    if self.image_mask[col, row] in self.shadow_regions:
                        self.shadow_mask[col, row] = 1
                # store the segments in a list.

                # Find the minimum row and col values.
                if self.segments[labels_image[col, row]]["minPoint"][0] > col:
                    self.segments[labels_image[col, row]]["minPoint"][0] = col
                if self.segments[labels_image[col, row]]["minPoint"][1] > row:
                    self.segments[labels_image[col, row]]["minPoint"][1] = row

                # Find the maximum row and col values.
                if self.segments[labels_image[col, row]]["maxPoint"][0] < col:
                    self.segments[labels_image[col, row]]["maxPoint"][0] = col
                if self.segments[labels_image[col, row]]["maxPoint"][1] < row:
                    self.segments[labels_image[col, row]]["maxPoint"][1] = row

                # Initialize the first time the col comes up.
                if col not in self.segments[labels_image[col, row]]['points'].keys():
                    self.segments[labels_image[col, row]]['points'][col] = []

                # Add the row to the corresponding list of rows for col.
                self.segments[labels_image[col, row]]['points'][col].append(row)

    def _is_shadow_point(self, col, row):
        # return self.shadow_mask[col, row] != 0
        return self.shadow_mask[col, row] >= 0.5

    def label_shadow_segments(self, threshold=0.5):
        """
        Labels a segment as a shadow if ratio of shadow points to non-shadow
        points is over the threshold(default 0.5)
        """
        for i in range(len(self.segments)):
            shadow_point_count = 0
            total_point_count = 0
            for col in self.segments[i]["points"].keys():
                rows = self.segments[i]["points"][col]
                for row in rows:
                    total_point_count += 1
                    if self._is_shadow_point(col, row):
                        shadow_point_count += 1
            print(shadow_point_count/total_point_count)

            # if shadow_point_count > 0:
            #     self.segments[i]["is_shadow"] = True

            if shadow_point_count/total_point_count > threshold:
                self.segments[i]["is_shadow"] = True
            else:
                self.segments[i]["is_shadow"] = False

    def _make_segment_images(self, segmented_image, label_shadows):
        """
        Make a seperate segmented images using the points in self.segements.
        """

        if self.mode != "" and label_shadows:
            self.label_shadow_segments()

        totalWidth = 0
        totalHeight = 0
        delBuffer = []
        for i in range(len(self.segments)):
            minX = self.segments[i]['minPoint'][0]
            minY = self.segments[i]['minPoint'][1]
            maxX = self.segments[i]['maxPoint'][0]
            maxY = self.segments[i]['maxPoint'][1]

            # Shitty hack, I feel dirty.
            if maxX - minX == 0:
                maxX += 1
            if maxY - minY == 0:
                maxY += 1

            newImage = np.zeros((maxX-minX, maxY-minY, 4), dtype=self.image.dtype)
            segmentShadowMask = np.zeros((maxX-minX, maxY-minY), dtype=self.image.dtype)
            # print("minX, minY:", minX, minY)
            # print("maxX, maxY:",maxX, maxY)
            for row in range(minY, maxY):
                for col in range(minX, maxX):
                    # if col in self.segments[i]['points'].keys():
                    if row in self.segments[i]['points'][col]:
                        # When col, row is in the current segment.
                        newImage[col-minX, row-minY] = [
                                    self.image[col, row][0],
                                    self.image[col, row][1],
                                    self.image[col, row][2],
                                    255, # point is in segment
                                ]
                        if label_shadows:
                            if self._is_shadow_point(col, row):
                                segmentShadowMask[col-minX, row-minY] = 255
                    else:
                        # When col, row is NOT in the current segment.

                        # newImage[col-minX, row-minY] = [ 0, 0, 0, 0 ]

                        newImage[col-minX, row-minY] = [
                                    self.image[col, row][0],
                                    self.image[col, row][1],
                                    self.image[col, row][2],
                                    0, # point is not in segment
                                ]

            # if newImage.
            if newImage.shape[0] > newImage.shape[1]:
                totalWidth += newImage.shape[0]
                totalHeight += newImage.shape[1]
            else:
                totalWidth += newImage.shape[1]
                totalHeight += newImage.shape[0]

            if newImage.shape[0] > 0 and newImage.shape[1] > 0:
                self.segments[i]["image"] = newImage
            else:
                print(newImage.shape)

            if self.mode != "":
                write_file = os.path.join("segments", "shadows",
                                          self.filename+"_"+str(i)+".png")
                if self.segments[i]["is_shadow"] == False:
                    write_file = os.path.join("segments","non_shadows",
                                              self.filename+"_"+ str(i)+".png")
                print("writing", write_file)
                cv.imwrite(
                        os.path.join("segments", "shadow_masks",
                            self.filename+"_"+ str(i)+".png"),
                        segmentShadowMask
                        )
                cv.imwrite(write_file, self.segments[i]["image"])

            # print(i, self.segments[i]["is_shadow"])
            # cv.imshow("individual seg",self.segments[i]["image"])
            # cv.waitKey()

        self.avgSize = (totalWidth/len(self.segments), totalHeight/len(self.segments))
        print("avg width: ", self.avgSize[0])
        print("avg height: ", self.avgSize[1])

    def get_segment_images(self, shape=(32, 32)):
        return [ cv.resize(segment["image"], shape)  for segment in self.segments ]

def process_mat_files():
    with open("./data/train/filelist.txt") as f:
        filenames = [ file.strip("\n") for file in  f.readlines() ]

    print(filenames)
    avgWidth = 0
    avgHeight = 0
    for filename in filenames:
        processed = ImageProcessed(cv.imread("./data/train/%s" % filename), filename)
        processed.open_with_mat()
        avgWidth += processed.avgSize[0]
        avgHeight+= processed.avgSize[1]
        # processed.show()
        # break
    print("Average size:", avgWidth/len(filenames), avgHeight/len(filenames))

def process_tappen_files():
    path = "./data/tappen/original"
    avgWidth = 0
    avgHeight = 0
    for (dirpath, dirnames, filenames) in os.walk(path):
        print(filenames)
        for filename in filenames:
            processed = ImageProcessed(cv.imread("./data/tappen/original/%s" % filename), filename)
            processed.open_with_tappen()
            avgWidth += processed.avgSize[0]
            avgHeight+= processed.avgSize[1]

            avgWidth = 0
            avgHeight = 0
    print("Average size:", avgWidth/len(filenames), avgHeight/len(filenames))
