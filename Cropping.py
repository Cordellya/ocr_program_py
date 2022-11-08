import os

import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage import rotate
from PIL import Image, ImageChops
# from Data_Ready import data_ready


def image_processing(img):
    # image = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(gray)
    thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # edges = cv2.Canny(gray, 100, 200, apertureSize=3)

    return thresh1, gray


def image_processing2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max = gray.max()
    min = gray.min()
    T = (max + min) / 2
    print(T)
    # blur = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(gray, T, 255, cv2.THRESH_BINARY)

    return thresh


def cropping_horizontal(img_matrix):
    total_count = []
    for px_row in img_matrix:
        count = 0
        for px_col in px_row:
            if (px_col & 255).all():
                count += 1
        total_count.append(count)

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
        cropping_img.append(img_matrix[range_black[0]:range_black[1], :])

    return cropping_img


def cropping_vertical(img_sentence):
    total_count = []
    for i, px_col in enumerate(img_sentence.T):
        count = 0
        for px_row in px_col:
            if (px_row & 255).all():
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

    # print(cropping_img)
    return cropping_img


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    print(bbox)
    if bbox:
        return im.crop(bbox)

# image = cv2.imread("data_uji/dataset13_copy.jpg")

# thresh, gray = image_processing(image)

# train_images, train_labels, val_images, val_labels = data_ready()


# def get_letters(thresh1):
#     # LB = LabelBinarizer()
#     model = tf.keras.models.load_model('model_50.h5')
#     true_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
#                     'K',
#                     'L',
#                     'M', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
#
#     letters = []
#     crop_sentence = cropping_horizontal(thresh)
#     # print(crop_sentence)
#     for i, res_sentence in enumerate(crop_sentence):
#         # cv2.imshow("res",res_sentence)
#         # cv2.waitKey(0)
#
#         crop_char = cropping_vertical(res_sentence)
#         for j, res_char in enumerate(crop_char):
#             # cv2.imshow("res", res_char)
#             # cv2.waitKey(0)
#             res = cv2.resize(res_char, (224, 224), interpolation=cv2.INTER_CUBIC)
#             res = res.astype("float32") / 255.0
#             res = np.expand_dims(res, axis=-1)
#             res = res.reshape(1, 224, 224, 1)
#             ypred = model.predict(res)
#             ypred = np.argmax(ypred, axis=1)
#             # ypred = train_labels[ypred]
#             # ypred = LB.inverse_transform(ypred)
#             print(j, true_classes[ypred[0]])
#             [x] = ypred
#             letters.append(x)
#
#     return letters
#
#
# letters = get_letters(thresh)
# print(letters)

# TEST USING TESTING DATA
# test_data = "dataset/data/testing_data"
# model = tf.keras.models.load_model('model_50.h5')

# for i in os.listdir(test_data):
#     sub_dir = os.path.join(test_data, i)
#     for j in os.listdir(sub_dir):
#         img = cv2.imread(os.path.join(sub_dir, j), 0)  # to loads an image from the specified file and returns it
#         img = cv2.resize(img, (224, 224))  # changing dimension of the image to the desire size
#         # thresh_test, _ = image_processing(img)  # append into array and add label into each image
#         # res_test = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
#         res_test = img.astype("float32") / 255.0
#         res_test = np.expand_dims(res_test, axis=-1)
#         res_test = res_test.reshape(1, 224, 224, 1)
#         ypred_test = model.predict(res_test)
#         ypred_test = np.argmax(ypred_test, axis=1)
#         print(j, true_classes[ypred_test[0]])
#         break

# GA KEPAKE
# def CannyThreshold(val, src_gray, src):
#     ratio = 3
#     kernel_size = 3
#     low_threshold = val
#     img_blur = cv2.blur(src_gray, (3, 3))
#     detected_edges = cv2.Canny(img_blur, low_threshold, low_threshold * ratio, kernel_size)
#     mask = detected_edges != 0
#     dst = src * (mask[:, :, None].astype(src.dtype))
#     return dst

# def skew_correction(img, img_thresh, delta=1, limit=45):
#     def determine_score(arr, angle):
#         data = rotate(arr, angle, reshape=False, order=0)
#         histogram = np.sum(data, axis=1, dtype=float)
#         score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
#         return histogram, score
#
#     # coords = np.column_stack(np.where(img_thresh > 0))
#     # angles = cv2.minAreaRect(coords)[-1]
#     #
#     # if angles < -45:
#     #     angles = -(90 + angles)
#     # else:
#     #     angles = -angles
#     scores = []
#     angles = np.arange(-limit, limit + delta, delta)
#     for angle in angles:
#         histogram, score = determine_score(img_thresh, angle)
#         scores.append(score)
#
#     best_angle = angles[scores.index(max(scores))]
#
#     image2 = cv2.imread(img)
#
#     (h, w) = image2.shape[:2]
#     print(h, w)
#     center = (w // 2, h // 2)
#     M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
#     rotated = cv2.warpAffine(image2, M, (w, h),
#                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
#
#     return best_angle, rotated


# edges = CannyThreshold(0, gray, image)
# angle_res, skew_res = skew_correction(image, thresh)
# hough_res = houghlines(image)
# print(edges)

# def houghlines(img):
#     # Convert the image to gray-scale
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # Find the edges in the image using canny detector
#     edges = cv2.Canny(gray, 50, 200)
#     # Detect points that form a line
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=10, maxLineGap=250)
#     # Draw lines on the image
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
#
#     # Show result
#     img2 = cv2.resize(img, dsize=(600, 600))
#
#     return img2

# plt.imshow(edges)
# plt.show()

# edg = cv2.resize(edges, (500, 500))
# imS = cv2.resize(image, (500, 500))  # Resize image
#
# # cdst = cv2.cvtColor(edg, cv2.COLOR_GRAY2BGR)
# # cdstP = np.copy(cdst)
#
# linesP = cv2.HoughLinesP(edg, 1, np.pi / 180, 50, None, 50, 10)
#
# if linesP is not None:
#     for i in range(0, len(linesP)):
#         l = linesP[i][0]
#         cv2.line(imS, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)

# minLineLength = 0
# maxLineGap = 0
# lines = cv2.HoughLinesP(edg,1,np.pi/180,150,minLineLength,maxLineGap)
# for line in lines:
#     for x1,y1,x2,y2 in line:
#         cv2.rectangle(imS,(x1,y1),(x2,y2),(0,255,0),2)

# print(linesP)
# cv2.imshow("edges", edg)
# cv2.imshow("lines", cdst)
# cv2.waitKey(0)

# cv2.imwrite(image, edges)

# cv2.imshow("Result Image", hough_res)
# cv2.waitKey(0)

# print('Skew angle:', angle_res)
#
# plt.imshow(skew_res)
# plt.show()
# cv2.imshow("Rotated", skew_res)
# cv2.waitKey(0)

# print(crop_char)
# plt.imshow(crop_char[0])
# plt.show()
# image = cv2.imread("test_3.jpg")
# _, labels = cv2.connectedComponents(thresh1)
# mask = np.zeros(thresh1.shape, dtype="uint8")
# total_pixels = image.shape[0] * image.shape[1]
# lower = total_pixels // 70  # heuristic param, can be fine tuned if necessary
# upper = total_pixels // 20  # heuristic param, can be fine tuned if necessary
#
# for (i, label) in enumerate(np.unique(labels)):
#     # If this is the background label, ignore it
#     if label == 0:
#         continue
#
#     # Otherwise, construct the label mask to display only connected component
#     # for the current label
#     labelMask = np.zeros(thresh1.shape, dtype="uint8")
#     labelMask[labels == label] = 255
#     numPixels = cv2.countNonZero(labelMask)
#
#     # If the number of pixels in the component is between lower bound and upper bound,
#     # add it to our mask
#     if numPixels > lower and numPixels < upper:
#         mask = cv2.add(mask, labelMask)
#
# cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# boundingBoxes = [cv2.boundingRect(c) for c in cnts]
#
#
# def compare(rect1, rect2):
#     if abs(rect1[1] - rect2[1]) > 10:
#         return rect1[1] - rect2[1]
#     else:
#         return rect1[0] - rect2[0]
#
#
# boundingBoxes = sorted(boundingBoxes, key=functools.cmp_to_key(compare))
