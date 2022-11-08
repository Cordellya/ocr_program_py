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

image = cv2.imread("data_uji/dataset13_copy.jpg")

thresh, gray = image_processing(image)

IMG_SIZE = 32

def get_letters(thresh):
    # LB = LabelBinarizer()
    model = tf.keras.models.load_model('saved_model/model_resnet_50_3.h5')
    true_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K',
                    'L',
                    'M', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    letters = []
    crop_sentence = cropping_horizontal(thresh)
    # print(crop_sentence)
    for i, res_sentence in enumerate(crop_sentence):
        # cv2.imshow("res",res_sentence)
        # cv2.waitKey(0)
        # plt.imshow(res_sentence)
        # plt.show()
        crop_char = cropping_vertical(res_sentence)
        for j, res_char in enumerate(crop_char):
            # cv2.imshow("res", res_char)
            # cv2.waitKey(0)
            # plt.imshow(res_char)
            # plt.show()
            res = cv2.resize(res_char, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
            res = res.astype("float32") / 255.0
            res = np.expand_dims(res, axis=-1)
            res = res.reshape(1, IMG_SIZE, IMG_SIZE, 1)
            ypred = model.predict(res)
            ypred = np.argmax(ypred, axis=1)
            # ypred = train_labels[ypred]
            # ypred = LB.inverse_transform(ypred)
            print(j, true_classes[ypred[0]])
            [x] = true_classes[ypred[0]]
            letters.append(x)

    return letters


letters = get_letters(thresh)
print(letters)