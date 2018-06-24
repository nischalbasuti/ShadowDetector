import tensorflow as tf
import numpy as np
import cv2 as cv
import os
import xml.etree.ElementTree
import copy
import pymeanshift as pms
import scipy.io as sio

class ImageProcessed(object):
    """Opens and image along with it's annotations/labels."""
    def __init__(self, image, filename = None):
        super(ImageProcessed, self).__init__()
        self.filename = filename
        self.image = image
        self.shadowPoints = []
        self.segments = []          # [ { image_of_segment, is_shadow }, ... ]
        
    # TODO: use the shadow points to divide into segments with labels.
    # The XML data is only for edges of shadows, so isn't useful.
    # def openWithXML(self):
    #     if self.filename is not None:
    #         # Find the shadow points from the corresponding xml file.
    #         # TODO: use regex to change extension
    #         xmlFileName = str(self.filename[:-3]) + "xml"
    #         print("xml file:", xmlFileName)
    #         e = xml.etree.ElementTree.parse("./xml/"+xmlFileName).getroot()

    #         for tag in e.findall("pt"):
    #             point = ( int(float(tag.get("x"))), int(float(tag.get("y"))) )
    #             cv.circle(self.imageShadow, point, 3, (0, 255, 0))
    #             self.shadowPoints.append( point )
    #         print("shadow points found:", len(self.shadowPoints))
    #     pass

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
        print("...started segmenting", self.filename, "...")
            # Using mean shift implementation from https://github.com/fjean/pymeanshift

        segmented_image, labels_image, number_regions = pms.segment(
                                                            self.image,
                                                            spatial_radius = 6,
                                                            range_radius = 4.5,
                                                            min_density = 50)

        # for col in range(self.image.shape[0]):
        #     for row in range(self.image.shape[1]):
        #         if labels_image[col, row] == labels_image[200, 271]:
        #             segmented_image[col, row][0] = 0    # b 
        #             segmented_image[col, row][1] = 255  # g
        #             segmented_image[col, row][2] = 0    # r

        shadow_mask = np.zeros((self.image.shape[0], self.image.shape[1], self.image.shape[2]))
        for col in range(self.image.shape[0]):
            for row in range(self.image.shape[1]):
                if self.image_mask[col, row] in self.shadow_regions:
                    shadow_mask[col, row][0]= 255
                    shadow_mask[col, row][1]= 255
                    shadow_mask[col, row][2]= 255

        cv.imshow("shadow mask", shadow_mask)
        cv.imshow("segmented_image", segmented_image)

        print("number_regions: ", number_regions)
        print("source image:", self.image.shape)
        print("...finished segmenting", self.filename, "...")
        return segmented_image[:,:]
        
if __name__ == '__main__':

    # Read images from directory
    # images = []
    # path = "./img"
    # for (dirpath, dirnames, filenames) in os.walk(path):
    #     images = [[cv.imread(os.path.join(path, fname)), fname] for fname in filenames]
    #     break

    # for image in images:
    #     print("size: %s" % str(image[0].shape))
    #     print("name: %s" % str(image[1]))
    #     processed = ImageProcessed(image[0], image[1])
    #     processed.openWithXML()
    #     processed._segment()
    #     processed.show()
    #     break # break after one iteration
    processed = ImageProcessed(cv.imread("./data/train/2.png"), "2.png")
    processed.openWithMAT()
    processed._segment()
    processed.show()
