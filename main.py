from image_processed import ImageProcessed
import cv2

image = ImageProcessed(cv2.imread("./data/train/2.png"), "2.png")
image.openWithMAT()
image._segment()
image.show()
