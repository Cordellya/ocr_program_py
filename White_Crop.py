import functools

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
import argparse

from transform import four_point_transform
from skimage.filters import threshold_local
from scipy.ndimage import rotate
from PIL import Image, ImageChops
import imutils

def edgeDetection(img):
    image = imutils.resize(img, height=500)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    # cv2.imshow("Image", image)
    # cv2.imshow("Edged", edged)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return edged


def findContour(edged):
    cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    # cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    # cv2.imshow("Outline", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return screenCnt


def scan(screenCnt, image):
    ratio = image.shape[0] / 500.0

    warped = four_point_transform(image, screenCnt.reshape(4, 2) * ratio)

    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255

    # kernel = np.ones((1,5), np.uint8)  # note this is a HORIZONTAL kernel
    # kernel = np.array([(0,1,0),(1,1,1),(0,1,0)])
    # e_im = cv2.dilate(warped, kernel, iterations=1)
    # e_im = cv2.erode(e_im, kernel, iterations=2)

    # cv2.imshow("Original", imutils.resize(orig, height = 650))
    # cv2.imshow("Scanned", imutils.resize(warped, height = 650))
    # cv2.imshow("Scanne", imutils.resize(e_im, height = 650))
    # cv2.waitKey(0)

    return warped


image = cv2.imread("data_uji/coba/coba1.jpg")
# getWhiteBox(image)
edged = edgeDetection(image)
plt.imshow(edged)
plt.show()

screenCnt = findContour(edged)
plt.imshow(screenCnt)
plt.show()

scannedImage = scan(screenCnt, image)
plt.imshow(scannedImage)
plt.show()

# GA DIPAKE

# def getWhiteBox(img):
#     # resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
#     _, thresh = cv2.threshold(blur_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

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

    # plt.imshow(thresh)
    # plt.show()

    # plt.imshow(new_image)
    # plt.show()

    # cv2.imshow("thresh", cv2.resize(thresh, (500, 500)))
    # cv2.imshow("edges", cv2.resize(edges, (500, 500)))
    # cv2.waitKey(0)

    # return thresh, new_image


# def getWhiteBox(img):
#     # resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)
#
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray, (5, 5), 0)
#
#     max = np.asarray(blur, dtype='float64').max()
#     min = np.asarray(blur, dtype='float64').min()
#     T = (max + min) / 2
#
#     _, thresh = cv2.threshold(blur, T, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
#     plt.imshow(thresh)
#     plt.show()
#
#     edges = cv2.Canny(thresh, 200, 255, apertureSize=3)
#
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     contoursThresh = []
#     countoursImg = []
#     for i, contour in enumerate(contours):
#         x, y, w, h = cv2.boundingRect(contour)
#         # print(x,y,w,h)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
#         contoursThresh.append(thresh[y:y + h, x:x + w])
#         countoursImg.append(img[y:y + h, x:x + w])
#
#     lens = [len(i) for i in contoursThresh]
#     # print(lens)
#     largest = np.asarray(lens).max()
#     index = lens.index(largest)
#
#     new_thresh = contoursThresh[index]
#
#     plt.imshow(new_thresh)
#     plt.show()
#
#     am_white_row = []
#     for i, row in enumerate(new_thresh):
#         count = 0
#         for j, col in enumerate(row):
#             if col == 255:
#                 count += 1
#         am_white_row.append(count)
#
#     max_white_row = np.asarray(am_white_row).max()
#     get_condition_row = [i for i in am_white_row if i > 1500] if max_white_row > 2000 else [i for i in am_white_row if
#                                                                                            i > 1000]
#     # get_condition_row = [i for i in am_white_row if i > 1000]
#     index_row_start = am_white_row.index(get_condition_row[0])
#     index_row_end = am_white_row.index(get_condition_row[len(get_condition_row) - 1])
#     print("row", get_condition_row)
#     print("max col", max_white_row)
#     print("index start", index_row_start)
#     print("index end", index_row_end)
#
#     am_white_col = []
#     for i, col in enumerate(new_thresh.T):
#         count = 0
#         for j, row in enumerate(col):
#             if (row & 255).all():
#                 count += 1
#         am_white_col.append(count)
#
#     max_white_col = np.asarray(am_white_col).max()
#     get_condition_col = [i for i in am_white_col if i > 1000]
#     # get_condition_row = [i for i in am_white_row if i > 1000]
#     index_col_start = am_white_col.index(get_condition_col[0])
#     index_col_end = am_white_col.index(get_condition_col[len(get_condition_col)-1])
#     print("col", get_condition_col)
#     print("max col", max_white_col)
#     print("start col", get_condition_col[0])
#     print("end col", get_condition_col[len(get_condition_col)-1])
#     print("index col start", index_col_start)
#     print("index col end", index_col_end)
#
#     # plt.imshow(thresh)
#     # plt.imshow(img)
#     # plt.imshow(contoursThresh[index])
#
#     # cv2.imshow("image", cv2.resize(img, (500, 500)))
#     # cv2.waitKey(0)
#     cv2.imshow("image", cv2.resize(img, (500, 500)))
#     # cv2.imshow("bunch_images1", cv2.resize(contoursThresh[index], (500, 500)))
#     # cv2.imshow("box row", cv2.resize(new_thresh[indexRowStart:indexRowEnd, :], (500, 500)))
#     # cv2.imshow("box col", cv2.resize(new_thresh[:, index_col:], (500, 500)))
#     # cv2.imshow("box", cv2.resize(new_thresh[indexRowStart:, index_col:], (500, 500)))
#     # cv2.imshow("bunch_images2", cv2.resize(countoursImg[index], (500, 500)))
#     # cv2.waitKey(0)
#
#     plt.imshow(contoursThresh[index])
#     plt.show()
#     plt.imshow(new_thresh[index_row_start:, :])
#     plt.show()
#     plt.imshow(new_thresh[:, index_col_start:index_col_end])
#     plt.show()


# def image_processing_first(img):
#     gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     print(gray1)
#     blur = cv2.GaussianBlur(gray1, (5, 5), 0)
#     # ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     (T, thresh) = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     edges = cv2.Canny(thresh, 200, 250, apertureSize=3)
#
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#
#     bunch_images = []
#     for i, contour in enumerate(contours):
#         x, y, w, h = cv2.boundingRect(contour)
#         # print(x,y,w,h)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
#         bunch_images.append(thresh[y:y + h, x:x + w])
#
#     cv2.imshow("image", cv2.resize(img, (500, 500)))
#     cv2.imshow("blur", cv2.resize(blur, (500, 500)))
#     cv2.imshow("thresh", cv2.resize(thresh, (500, 500)))
#     cv2.imshow("gray", cv2.resize(gray1, (500, 500)))
#     cv2.imshow("edges", cv2.resize(edges, (500, 500)))
#     cv2.waitKey(0)
#
#     return bunch_images


# image = cv2.imread("data_uji/boxcrop/data_uji_boxcrop (3).jpg")
# gray1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# (T, thresh) = cv2.threshold(gray1, 150, 255, cv2.THRESH_BINARY_INV)
#
# print(thresh)
# print(gray1)
#
# cv2.imshow("thresh", cv2.resize(thresh, (500, 500)))
# cv2.imshow("gray1", cv2.resize(gray1, (500, 500)))
# cv2.waitKey(0)
# image = cv2.imread("data_uji/dataset9.jpg")
# print(image)s

# bunch_images = image_processing_first(image)
