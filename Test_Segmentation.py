import cv2
import matplotlib.pyplot as plt
from White_Crop import getWhiteBox
from Cropping import cropping_horizontal, cropping_vertical

image = cv2.imread("data_uji/scale_analysis/scale (4).jpg")

thresh, new_image = getWhiteBox(image)
crop_sentences = cropping_horizontal(new_image)

for i in crop_sentences:
    plt.imshow(i)
    plt.show()