from image_processer import ImageProcessed
import cv2

if __name__ == '__main__':
    with open("./data/train/filelist.txt") as f:
        filenames = [ file.strip("\n") for file in  f.readlines() ]

    print(filenames)
    avgWidth = 0
    avgHeight = 0
    for filename in filenames:
        processed = ImageProcessed(cv.imread("./data/train/%s" % filename), filename)
        processed.openWithMAT()
        processed._segment()
        avgWidth += processed.avgSize[0]
        avgHeight+= processed.avgSize[1]
        # processed.show()
    print("Average size:", avgWidth/len(filenames), avgHeight/len(filenames))

