import functools
import os

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy
import numpy as np
import argparse
from scipy.ndimage import rotate
from PIL import Image, ImageChops
import tensorflow as tf

def preprocessing(im):
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])

    # kernel = np.ones((5, 5), np.float32) / 25
    image_sharp = cv2.filter2D(src=im, ddepth=-1, kernel=kernel)

    blur = cv2.GaussianBlur(image_sharp, (3, 3), 0)
    # blur = cv2.GaussianBlur(image_sharp, (0, 0), sigmaX=33, sigmaY=33)

    # plt.imshow(image_sharp)
    # divide = cv2.divide(im, blur, scale=255)

    # plt.show()
    max = np.asarray(blur, dtype='float64').max()
    min = np.asarray(blur, dtype='float64').min()
    T = (max + min) / 2
    # print(gray)
    thresh = cv2.threshold(blur, T, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    # morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, se)

    # plt.imshow(thresh)
    # plt.show()
    return thresh


def crop_box(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray.shape)

    height = gray.shape[0]
    width = gray.shape[1]

    point_x = width * 0.275
    point_y = height * 0.375

    point_corner_y = (round(0 + point_y), round(height - point_y))
    point_corner_x = (round(0 + point_x), round(width - point_x))

    # print(point_corner_y)
    # print(point_corner_x)

    new_image = gray[point_corner_y[0]:point_corner_y[1], point_corner_x[0]:point_corner_x[1]]

    # plt.imshow(new_image)
    # plt.show()

    thresh = preprocessing(new_image)

    return thresh


def cropping_horizontal(im):
    total_count = []
    for px_row in im:
        count = 0
        for px_col in px_row:
            if px_col == 0:
                count += 1
        total_count.append(count)

    # print(total_count)
    nonzero_to_one = list(map(int, [i != 0 for i in total_count]))
    # print(x)
    idx_range = []  # result list of tuples
    start_idx = 0  # index of first element of each range of zeros or non-zeros
    for n, px in enumerate(nonzero_to_one):
        if (n + 1 == len(nonzero_to_one)) or (nonzero_to_one[n] != nonzero_to_one[n + 1]):
            # Here: EITHER it is the last value of the list
            #       OR a new range starts at index n+1
            if px != 0:
                idx_range.append((start_idx, n))
            start_idx = n + 1

    # print(total_count)
    # print(idx_range)
    cropping_img = []
    for index, range_black in enumerate(idx_range):
        cropping_img.append(im[range_black[0]:range_black[1], :])

    return cropping_img


def cropping_vertical(img_sentence):
    total_count = []
    for i, px_col in enumerate(img_sentence.T):
        count = 0
        for px_row in px_col:
            if px_row == 0:
                count += 1
        total_count.append(count)

    nonzero_to_one = list(map(int, [i != 0 for i in total_count]))

    idx_range = []  # result list of tuples
    start_idx = 0  # index of first element of each range of zeros or non-zeros
    for n, px in enumerate(nonzero_to_one):
        if (n + 1 == len(nonzero_to_one)) or (nonzero_to_one[n] != nonzero_to_one[n + 1]):
            # Here: EITHER it is the last value of the list
            #       OR a new range starts at index n+1
            if px != 0:
                idx_range.append((start_idx, n))
            start_idx = n + 1

    # print(idx_range)
    cropping_img = []
    for index, range_black in enumerate(idx_range):
        cropping_img.append(img_sentence[:, range_black[0]:range_black[1]])
        # length = len(cropping_img[index]) + 1024 - 1024 % len(cropping_img[index])
        cropping_img[index] = np.pad(cropping_img[index], 5, 'constant', constant_values=255)

    # print(cropping_img)
    return cropping_img


def final_crop(im):
    box = crop_box(im)
    crop_sentences = cropping_horizontal(box)

    exp_date_idx = 0
    code_prod_idx = 0 

    arr_char_img = []

    if len(crop_sentences) == 4:
        exp_date_idx = 2
        code_prod_idx = 4
    else:
        exp_date_idx = 3
        code_prod_idx = 5

    for i, sentence in enumerate(crop_sentences[exp_date_idx:code_prod_idx]):
        # plt.imshow(sentence)
        # plt.show()
        crop_char = cropping_vertical(sentence)
        for j, char in enumerate(crop_char):
            # plt.imshow(char)
            # plt.show()
            # cv2.imwrite("Hasil_Cropping/Crop_{}/Char_{}_{}.jpg".format(i, j, k), char)
            arr_char_img.append(char)

    return arr_char_img

# saved_dir = "D:/Aplikasi_Skripsi/Program_OCR/Hasil_Cropping"

# for i in range(1, 36):
#     os.mkdir('Hasil_Cropping/Crop_{}'.format(i))

#     image = cv2.imread("dataset/crop_new/crop ({}).jpg".format(i))

#     crop_img = crop_box(image)
#     crop_sentences = cropping_horizontal(crop_img)

#     sentence1 = 0
#     sentence2 = 0

#     if len(crop_sentences) == 4:
#         sentence1 = 2
#         sentence2 = 4
#     # elif len(crop_sentences) == 3:
#     #     sentence1 = 1
#     #     sentence2 = 3
#     else:
#         sentence1 = 3
#         sentence2 = 5

#     for j, sentence in enumerate(crop_sentences[sentence1:sentence2]):
#         # plt.imshow(sentence)
#         # plt.show()
#         crop_char = cropping_vertical(sentence)
#         for k, char in enumerate(crop_char):
#             # plt.imshow(char)
#             # plt.show()
#             cv2.imwrite("Hasil_Cropping/Crop_{}/Char_{}_{}.jpg".format(i, j, k), char)

# plt.imshow(crop_char[0])
# plt.show()

# def preprocessing(im):
#     blur = cv2.GaussianBlur(im, (3, 3), 0)
#
#     se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
#     bg = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, se)
#     out_gray = cv2.divide(blur, bg, scale=255)
#     # out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
#     # kernel = np.zeros((5, 5), np.uint8)
#
#     out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#     # opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel1)
#
#     plt.imshow(out_binary)
#     plt.show()
#
#     plt.imshow( out_gray)
#     plt.show()
#
#     # plt.imshow(erode)
#     # plt.show()
#     return out_binary

# thresh,_ = image_processing(crop_img)
# crop_final = crop(crop_img)

# true_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
#                 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
#
# letters = []
# model = tf.keras.models.load_model('saved_model/resnet_new/model_resnet2_100_aug_new.h5')
#
# for res in crop_final:
#     ypred = model.predict(res)
#     ypred = np.argmax(ypred, axis=1)
#     # ypred = train_labels[ypred]
#     # ypred = LB.inverse_transform(ypred)
#     print(ypred[0], true_classes[ypred[0]])
#     [x] = true_classes[ypred[0]]
#     letters.append(x)


# def crop(im):
#     # parent_dir = "D:/Aplikasi_Skripsi/Program_OCR/Hasil_Cropping"
#     kernel = np.array([[0, -1, 0],
#                        [-1, 5, -1],
#                        [0, -1, 0]])
#     image_sharp = cv2.filter2D(src=im, ddepth=-1, kernel=kernel)
#     #
#     # plt.imshow(image_sharp)
#     # plt.show()
#     blur = cv2.GaussianBlur(image_sharp, (3, 3), 0)
#
#     max = np.asarray(blur, dtype='float64').max()
#     min = np.asarray(blur, dtype='float64').min()
#     T = (max + min) / 2
#     # print(gray)
#     thresh = cv2.threshold(blur, T, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
#     # print(thresh)
#     crop_final = []
#     crop_sentence = cropping_horizontal(thresh)
#     # print(crop_sentence)
#     for i, res_sentence in enumerate(crop_sentence[3:4]):
#         plt.imshow(res_sentence)
#         plt.show()
#         crop_char = cropping_vertical(res_sentence)
#         for j, res_char in enumerate(crop_char):
#             # cv2.imshow("res", res_char)
#             # cv2.waitKey(0)
#             # plt.imshow(res_char)
#             # plt.show()
#             # print(res_char.shape)
#             res = cv2.resize(res_char, (32, 32))
#             # cv2.imwrite("{}/{}.jpg".format(parent_dir, j), res)
#             # plt.imshow(res)
#             # plt.show()
#             res = res.astype("float32") / 255.0
#             res = np.expand_dims(res, axis=-1)
#             res = res.reshape(1, 32, 32, 1)
#
#             # np.append(crop, res)
#             crop_final.append(res)
#     return crop_final
# def get_box(img):
#
#     # GREY, BLUR, EDGES
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     blur_gray = cv2.GaussianBlur(gray, (3, 3), 0)
#
#     max = np.asarray(blur_gray, dtype='float64').max()
#     min = np.asarray(blur_gray, dtype='float64').min()
#     T = (max + min) / 2
#     ret, thresh = cv2.threshold(blur_gray, T, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#
#     edges = cv2.Canny(thresh, 50, 150, apertureSize = 3)
#
#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#     bunch_images = []
#     for i, contour in enumerate(contours):
#         x, y, w, h = cv2.boundingRect(contour)
#         # print(x,y,w,h)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
#         bunch_images.append(thresh[y:y + h, x:x + w])
#
#     cv2.imshow("img contour", cv2.resize(img, (500, 500)))
#     cv2.waitKey(0)
#
#     lens = [len(i) for i in bunch_images]
#     largest = np.asarray(lens).max()
#     index = lens.index(largest)
#     new_images = bunch_images[index]
#
#     # y_top = 1500
#     # x_top = x_bottom = 900
#     #
#     # y_bottom = 2700
#     #
#     # count_white = 0
#     # get_index_top = 0
#     # for i, color in enumerate(thresh[y_top, x_top:]):
#     #     if color == 255:
#     #         get_index_top = i+x_top
#     #         break
#     #
#     # get_index_bottom = 0
#     # for i, color in enumerate(thresh[y_bottom, x_bottom:]):
#     #     if color == 255:
#     #         get_index_bottom = i+x_bottom
#     #         break
#     #
#     # # print(count_white)
#     # print(get_index_top)
#     # print(get_index_bottom)
#     #
#     # new_image = thresh[y_top:y_bottom, x_top:get_index_top]
#
#     plt.imshow(new_images)
#     plt.show()
