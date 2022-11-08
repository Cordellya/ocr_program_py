import functools

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
import argparse
from scipy.ndimage import rotate
from PIL import Image, ImageChops
from Cropping import cropping_horizontal


def get_box(img):

    # GREY, BLUR, EDGES
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (3, 3), 0)

    max = np.asarray(blur_gray, dtype='float64').max()
    min = np.asarray(blur_gray, dtype='float64').min()
    T = (max + min) / 2
    ret, thresh = cv2.threshold(blur_gray, T, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    edges = cv2.Canny(thresh, 50, 150, apertureSize = 3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bunch_images = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        bunch_images.append(thresh[y:y + h, x:x + w])

    cv2.imshow("img contour", cv2.resize(img, (500, 500)))
    cv2.waitKey(0)

    lens = [len(i) for i in bunch_images]
    largest = np.asarray(lens).max()
    index = lens.index(largest)
    new_images = bunch_images[index]

    # y_top = 1500
    # x_top = x_bottom = 900
    #
    # y_bottom = 2700
    #
    # count_white = 0
    # get_index_top = 0
    # for i, color in enumerate(thresh[y_top, x_top:]):
    #     if color == 255:
    #         get_index_top = i+x_top
    #         break
    #
    # get_index_bottom = 0
    # for i, color in enumerate(thresh[y_bottom, x_bottom:]):
    #     if color == 255:
    #         get_index_bottom = i+x_bottom
    #         break
    #
    # # print(count_white)
    # print(get_index_top)
    # print(get_index_bottom)
    #
    # new_image = thresh[y_top:y_bottom, x_top:get_index_top]

    plt.imshow(new_images)
    plt.show()




image = cv2.imread("data_uji/whitebox/whitebox (1).jpg")
get_box(image)
