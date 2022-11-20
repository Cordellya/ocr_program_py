import os

import cv2
import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from Cropping import cropping_horizontal, image_processing, cropping_vertical


IMG_SIZE = 32

true_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

model = tf.keras.models.load_model('saved_model/resnet_new/model_resnet2_100_aug_new.h5')


# def crop_img(thresh):
#     crop=[]
#     crop_sentence = cropping_horizontal(thresh)
#     print(crop_sentence)
#     kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
#     # print(crop_sentence)
#     for i, res_sentence in enumerate(crop_sentence):
#         # cv2.imshow("res",res_sentence)
#         # cv2.waitKey(0)
#         # plt.imshow(res_sentence)
#         # plt.show()
#         crop_char = cropping_vertical(res_sentence)
#         for j, res_char in enumerate(crop_char):
#             # erode = cv2.erode(res_char, kernel, iterations=3)
#             # cv2.imshow("res", res_char)
#             # cv2.waitKey(0)
#             # plt.imshow(erode)
#             # plt.show()
#             # print(res_char.shape)
#             res = cv2.resize(res_char, (IMG_SIZE, IMG_SIZE))
#             # # plt.imshow(res)
#             # # plt.show()
#             res = res.astype("float32") / 255.0
#             res = np.expand_dims(res, axis=-1)
#             res = res.reshape(1, IMG_SIZE, IMG_SIZE, 1)
#             # np.append(crop, res)
#             crop.append(res)
#     return crop
#
# image = cv2.imread("data_uji/dataset13_copy.jpg")
#
# thresh, gray = image_processing(image)
# crop_res = crop_img(thresh)

# letters=[]
# model = tf.keras.models.load_model('saved_model/resnet_new/model_resnet2_100_aug_new.h5')

# for res in crop_res:
#     thin = np.zeros(res.shape,dtype='uint8')
#     erode = cv2.erode(res, kernel, iterations=3)
#     # Opening on eroded image
#     opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
#     # Subtract these two
#     subset = erode - opening
#
#     # print(thin, subset)
#     # Union of all previous sets
#     thin = cv2.bitwise_or(subset, thin)
#
#     fig, axes = plt.subplots(1, 2, figsize=(10, 10))
#     axes = axes.flatten()
#     axes[0].imshow(erode, cmap="gray")
#     axes[1].imshow(res, cmap='gray')
#     plt.tight_layout()
#     plt.show()

# for res in crop_res:
#     ypred = model.predict(res)
#     ypred = np.argmax(ypred, axis=1)
#     # ypred = train_labels[ypred]
#     # ypred = LB.inverse_transform(ypred)
#     print(ypred[0], true_classes[ypred[0]])
#     [x] = true_classes[ypred[0]]
#     letters.append(x)




# def get_letters(thresh):
#     # LB = LabelBinarizer()
#     model = tf.keras.models.load_model('saved_model/resnet_new/model_resnet_100_aug.h5')
#     true_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
#                     'K',
#                     'L',
#                     'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
#
#     letters = []
#     crop_sentence = cropping_horizontal(thresh)
#     # print(crop_sentence)
#     for i, res_sentence in enumerate(crop_sentence):
#         # cv2.imshow("res",res_sentence)
#         # cv2.waitKey(0)
#         # plt.imshow(res_sentence)
#         # plt.show()
#         crop_char = cropping_vertical(res_sentence)
#         for j, res_char in enumerate(crop_char):
#             # cv2.imshow("res", res_char)
#             # cv2.waitKey(0)
#             # plt.imshow(res_char)
#             # plt.show()
#             res = cv2.resize(res_char, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
#             res = res.astype("float32") / 255.0
#             res = np.expand_dims(res, axis=-1)
#             res = res.reshape(1, IMG_SIZE, IMG_SIZE, 1)
#             ypred = model.predict(res)
#             ypred = np.argmax(ypred, axis=1)
#             # ypred = train_labels[ypred]
#             # ypred = LB.inverse_transform(ypred)
#             print(ypred[0], true_classes[ypred[0]])
#             [x] = true_classes[ypred[0]]
#             letters.append(x)
#
#     return letters


# letters = get_letters(thresh)
# print(letters)