import functools

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
import argparse
from scipy.ndimage import rotate
from PIL import Image, ImageChops


def image_processing_first(img):
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # max = gray.max()
    # min = gray.min()
    # T = (max + min) / 2
    # print(T)
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray1, (5, 5), 0)
    ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 200, 250, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # plt.imshow(thresh)
    # plt.show()
    #
    # plt.imshow(edges)
    # plt.show()
    # blur = cv2.GaussianBlur(gray1, (5, 5), 0)
    # edges = cv2.Canny(blur, 100, 200, apertureSize=3)
    bunch_images = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        bunch_images.append(thresh[y:y + h, x:x + w])

    cv2.imshow("image", cv2.resize(img, (500, 500)))
    cv2.imshow("thresh", cv2.resize(thresh, (500, 500)))
    cv2.imshow("edges", cv2.resize(edges, (500, 500)))
    cv2.waitKey(0)
    # for px in thresh:
    # new_thresh = []
    # x1 = []
    # for i, px_row in enumerate(gray1):
    #     for j, px_col in enumerate(px_row):
    #         # print(px_col)
    #         if px_col == 255:
    #             x1.append((i,j))
    #             break

    # for i, first_white in enumerate(x1[0]):

    return bunch_images


def image_processing_second(bunch_images):
    # # image = cv2.imread(img)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # print(gray)
    # thresh2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    #
    # # edges = cv2.Canny(gray, 100, 200, apertureSize=3)

    lens = [len(i) for i in bunch_images]
    largest = max(lens)
    index = lens.index(largest)
    # print(len(bunch_images[index]))
    cv2.imshow("bunch_images1", cv2.resize(bunch_images[index], (500, 500)))
    cv2.waitKey(0)

    new_images = bunch_images[index]

    for i, px_row in enumerate(new_images):
        for j, px_col in enumerate(px_row):
            if px_col == 0:
                break
            if px_col == 255:
                new_images[i][j] = 0

    cv2.imshow("new image 2", cv2.resize(new_images, (500, 500)))

    for m, px_col in enumerate(new_images.T):
        for n, px_row in enumerate(px_col):
            # print(px_row)
            if px_row == 0:
                break
            if px_row == 255:
                new_images.T[m][n] = 0

    # print(new_images)

    cv2.imshow("new image transpose", cv2.resize(new_images.T.T, (500, 500)))
    cv2.waitKey(0)

    return new_images


def get_box(img):
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray1, (5, 5), 0)
    max = np.asarray(blur, dtype='float64').max()
    min = np.asarray(blur, dtype='float64').min()
    T = (max + min)/2

    print(max, min, T)
    ret, thresh = cv2.threshold(blur, T, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(thresh, 200, 250, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bunch_images = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        bunch_images.append(thresh[y:y + h, x:x + w])

    lens = [len(i) for i in bunch_images]
    print(lens)
    largest = np.asarray(lens).max()
    index = lens.index(largest)

    print(largest, index)


    cv2.imshow("image", cv2.resize(img, (500, 500)))
    cv2.imshow("thresh", cv2.resize(thresh, (500, 500)))
    cv2.imshow("edges", cv2.resize(edges, (500, 500)))
    cv2.imshow("bunch_images1", cv2.resize(bunch_images[index], (500, 500)))
    cv2.waitKey(0)
    return


image = cv2.imread("data_uji/boxcrop/data_uji_boxcrop (5).jpg")
box = get_box(image)
# image = cv2.imread("data_uji/dataset9.jpg")
# print(image)

# bunch_images = image_processing_first(image)
# crop = image_processing_second(bunch_images)

# thresh2, gray = image_processing_second(thresh)

# print(thresh2)
# cv2.imshow("first gray", cv2.resize(gray1, (500, 500)))
# cv2.imshow("first thresh", cv2.resize(thresh, (500, 500)))
# cv2.imshow("second thresh", cv2.resize(thresh2, (500, 500)))
# cv2.waitKey(0)
