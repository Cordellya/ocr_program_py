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
    resized = cv2.resize(img, (500, 500), interpolation=cv2.INTER_AREA)

    # GREY, BLUR, EDGES
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (3, 3), 0)

    max = np.asarray(blur_gray, dtype='float64').max()
    min = np.asarray(blur_gray, dtype='float64').min()
    T = (max + min) / 2
    ret, thresh = cv2.threshold(blur_gray, T, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    edges = cv2.Canny(thresh, 50, 150, apertureSize = 3)

    # plt.imshow(bg)
    # plt.show()
    #
    # bg = cv2.blur(bg, (3, 3))
    # edges = cv2.Canny(bg, 50, 150, apertureSize=3)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    bunch_images = []
    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        bunch_images.append(thresh[y:y + h, x:x + w])

    cv2.imshow("img contour", cv2.resize(img, (500, 500)))
    cv2.waitKey(0)
    #
    # print(bunch_images)
    lens = [len(i) for i in bunch_images]
    # largest = np.asarray(lens).max()
    largest = np.asarray(lens).max()
    index = lens.index(largest)
    new_images = bunch_images[index]

    plt.imshow(new_images)
    plt.show()
    cv2.imshow("new_images", cv2.resize(new_images, (500, 500)))
    cv2.waitKey(0)

    h, w = new_images.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    cv2.floodFill(new_images, mask, (0, 0), 123)
    floodfill = new_images.copy()

    cv2.imshow("flood fill", cv2.resize(floodfill, (500, 500)))
    cv2.waitKey(0)

    bg = new_images
    bg[new_images == 123] = 0
    #
    # h2, w2 = new_images.shape[:2]
    # mask2 = np.zeros((h2 + 2, w2 + 2), np.uint8)
    # cv2.floodFill(new_images, mask2, (0, 0), 0)
    # floodfill2 = new_images.copy()

    # cv2.imshow("img", resized)
    # cv2.imshow("thresh", thresh)
    cv2.imshow("new img",  cv2.resize(new_images, (500, 500)))
    cv2.imshow("mask",  cv2.resize(bg, (500, 500)))
    # cv2.imshow("flood fill", floodfill)
    # cv2.imshow("flood fill2", floodfill2)
    # cv2.imshow("edges", edges)
    cv2.waitKey(0)

    # plt.imshow(bg)
    # plt.show()


image = cv2.imread("data_uji/whitebox/whitebox (1).jpg")
get_box(image)
#
# import sys
# import inspect
#
# # sys.setrecursionlimit(4000)
#
# # print(sys.getrecursionlimit())
#
# def get_box(img):
#     gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur = cv2.GaussianBlur(gray1, (5, 5), 0)
#     max = np.asarray(blur, dtype='float64').max()
#     min = np.asarray(blur, dtype='float64').min()
#     T = (max + min) / 2
#     ret, thresh = cv2.threshold(blur, T, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     edges = cv2.Canny(thresh, 200, 250, apertureSize=3)
#
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     bunch_images = []
#     for i, contour in enumerate(contours):
#         x, y, w, h = cv2.boundingRect(contour)
#         # print(x,y,w,h)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
#         bunch_images.append(thresh[y:y + h, x:x + w])
#
#     lens = [len(i) for i in bunch_images]
#     largest = np.asarray(lens).max()
#     index = lens.index(largest)
#     new_images = bunch_images[index]
#
#     cv2.imshow("edges1", cv2.resize(img, (500, 500)))
#     cv2.waitKey(0)
#
#     h, w = new_images.shape[:2]
#     mask = np.zeros((h + 2, w + 2), np.uint8)
#     cv2.floodFill(new_images, mask, (0, 0), 0)
#     # Show the flood fill image
#     floodfill = new_images.copy()
#
#     # cv2.imshow("image", cv2.resize(img, (500, 500)))
#     cv2.imshow("new_images", cv2.resize(new_images, (500, 500)))
#     # cv2.imshow("thresh", cv2.resize(thresh, (500, 500)))
#     # cv2.imshow("flood", cv2.resize(floodfill, (500, 500)))
#     # cv2.imshow("edges", cv2.resize(edges, (500, 500)))
#     cv2.waitKey(0)
#
#     return new_images
#
# # def isValid(img, m, n, x, y, prevC, newC):
# #     if x < 0 or x >= m or y < 0 or y >= n or img[x][y] != prevC or img[x][y] == newC:
# #         return False
# #     return True
#
#
# def isValid(screen, m, n, x, y, prevC, newC):
#     if x < 0 or x >= m \
#             or y < 0 or y >= n or \
#             (screen[x][y] != prevC).any() \
#             or (screen[x][y] == newC).any():
#         return False
#     return True
#
#
# # FloodFill function
# def floodFill(screen,
#               m, n, x,
#               y, prevC, newC):
#     queue = []
#
#     # Append the position of starting
#     # pixel of the component
#     queue.append([x, y])
#
#     # Color the pixel with the new color
#     screen[x][y] = newC
#
#     # While the queue is not empty i.e. the
#     # whole component having prevC color
#     # is not colored with newC color
#     while queue:
#
#         # Dequeue the front node
#         currPixel = queue.pop()
#
#         posX = currPixel[0]
#         posY = currPixel[1]
#
#         # Check if the adjacent
#         # pixels are valid
#         if isValid(screen, m, n,
#                    posX + 1, posY,
#                    prevC, newC):
#             # Color with newC
#             # if valid and enqueue
#             screen[posX + 1][posY] = newC
#             queue.append([posX + 1, posY])
#
#         if isValid(screen, m, n,
#                    posX - 1, posY,
#                    prevC, newC):
#             screen[posX - 1][posY] = newC
#             queue.append([posX - 1, posY])
#
#         if isValid(screen, m, n,
#                    posX, posY + 1,
#                    prevC, newC):
#             screen[posX][posY + 1] = newC
#             queue.append([posX, posY + 1])
#
#         if isValid(screen, m, n,
#                    posX, posY - 1,
#                    prevC, newC):
#             screen[posX][posY - 1] = newC
#             queue.append([posX, posY - 1])
#
#     return screen
#
# image = cv2.imread("data_uji/boxcrop/data_uji_boxcrop (6).jpg")
# new_images = get_box(image)
#
# m = len(new_images)
#
# # Column of the display
# n = len(new_images[0])
#
# # Co-ordinate provided by the user
# x = 0
# y = 0
#
# screen = floodFill(new_images, m, n, x, y, new_images[0][0], 0)
# # print(new_images)
# # cv2.imshow("new_images", cv2.resize(new_images, (500, 500)))
# cv2.imshow("final", cv2.resize(screen, (500, 500)))
# cv2.waitKey(0)
#
# # cropping_horizontal(screen)
# #
# # for i, res_sentence in enumerate(cropping_horizontal(screen)):
# #     plt.imshow(res_sentence)
# #     plt.show()
#     # cv2.imshow("res", cv2.resize(res_sentence, (500, 500)))
#     # cv2.waitKey(0)
#
#
# # new_images = get_box(image)
# # new_images2 = new_images[:10, :10]
# # def change_color(img, x, y):
# #     # print(len(inspect.stack()))
# #     try:
# #         if img[x][y] == 0:
# #             return
# #
# #         img[x][y] = 0
# #         # for x_off, y_off in ((0, 1), (0, -1), (1, 0), (-1, 0)):
# #         #     change_color(img, x + x_off, y + y_off)
# #
# #         change_color(img, x + 1, y)
# #         change_color(img, x - 1, y)
# #         change_color(img, x, y + 1)
# #         change_color(img, x, y - 1)
# #     except:
# #         pass
#
#     # if change_color(img, x, y + 1) == 0:
#     #     return
#     #
#     # return change_color(img, x + 1, y)
#     # if temp[x][y] == 255:
#     #     temp[x][y] = 0
#     #     print(x, y, temp[x][y])
#     #     change_color(temp, x + 1, y)
#     #     return change_color(temp, x+1, y)
#     # else:
#     #     return temp
#
#
# # image = cv2.imread("data_uji/boxcrop/data_uji_boxcrop (3).jpg")
# #
# # new_images = get_box(image)
# # new_images2 = new_images[:10, :10]
# # print(new_images)
# # contours2, _ = cv2.findContours(new_images, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# # for i, contour in enumerate(contours2):
# #     x, y, w, h = cv2.boundingRect(contour)
# #     # print(x,y,w,h)
# #     cv2.rectangle(new_images, (x, y), (x + w, y + h), (0, 255, 0), 3)
#     # bunch_images.append(thresh[y:y + h, x:x + w])
#
# # cv2.imshow("new_images2", cv2.resize(new_images, (500, 500)))
#
# # change_color(new_images2, 0, 0)
# # # print(new_images)
# # cv2.imshow("final", cv2.resize(new_images2, (500, 500)))
# # cv2.waitKey(0)
# # print(final_images)
#
# # cv2.imshow("images", cv2.resize(bunch_images,(500, 500)))
# # cv2.waitKey(0)
#
#
#
# # def get_box(img):
# #     gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# #     blur = cv2.GaussianBlur(gray1, (5, 5), 0)
# #     max = np.asarray(blur, dtype='float64').max()
# #     min = np.asarray(blur, dtype='float64').min()
# #     T = (max + min) / 2
# #     ret, thresh = cv2.threshold(blur, T, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# #     edges = cv2.Canny(blur, 200, 250, apertureSize=3)
# #
# #     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# #     bunch_images = []
# #     for i, contour in enumerate(contours):
# #         x, y, w, h = cv2.boundingRect(contour)
# #         # print(x,y,w,h)
# #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
# #         bunch_images.append(thresh[y:y + h, x:x + w])
# #
# #     lens = [len(i) for i in bunch_images]
# #     largest = np.asarray(lens).max()
# #     index = lens.index(largest)
# #     new_images = bunch_images[index]
# #
# #     # cv2.imshow("image", cv2.resize(img, (500, 500)))
# #     cv2.imshow("new_images", cv2.resize(new_images, (500, 500)))
# #     # cv2.imshow("thresh", cv2.resize(thresh, (500, 500)))
# #     # cv2.imshow("edges", cv2.resize(edges, (500, 500)))
# #     cv2.waitKey(0)
# #
# #     return new_images