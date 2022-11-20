import math
from shutil import copyfile

import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

dir_data = "dataset/data_new/All"
# train_data = []

for i in os.listdir(dir_data):
    # print(i)
    if not os.path.isdir('dataset/data_new/training/' + i):
        os.mkdir('dataset/data_new/training/' + i)
    if not os.path.isdir('dataset/data_new/testing/' + i):
        os.mkdir('dataset/data_new/testing/' + i)

count = 1
for i in os.listdir(dir_data):
    class_len = len(os.listdir(dir_data + "/" + i))
    # print(class_len)

    train_len = math.floor(class_len * 0.80)
    test_len = math.ceil(class_len * 0.20)
    # print(train_len)

    rand_data = np.random.randint(low=1, high=class_len, size=class_len)
    rand_train = rand_data[:train_len]
    rand_test = rand_data[train_len:]
    print(len(rand_test), len(rand_train))
    # print(len(rand_train))
    for img_no_train in rand_train:
        src = dir_data + "/" + i + '/img' + str(count).zfill(3) + '-' + str(img_no_train).zfill(5) + '.png'
        des = 'D:/Aplikasi_Skripsi/Program_OCR/dataset/data_new/training/' + i + '/img' + str(count).zfill(3) + '-' + str(img_no_train).zfill(
            5) + '.png'
        copyfile(src, des)

    for img_no_test in rand_test:
        # print("masuk")
        src = dir_data + "/" + i + '/img' + str(count).zfill(3) + '-' + str(img_no_test).zfill(5) + '.png'
        des = 'D:/Aplikasi_Skripsi/Program_OCR/dataset/data_new/testing/' + i + '/img' + str(count).zfill(3) + '-' + str(img_no_test).zfill(
            5) + '.png'
        copyfile(src, des)

    count += 1
# for char in range(1, 63):
#   classLen = len(os.listdir(base + str(char).zfill(3)))
#
#   trainLen = math.floor(classLen*0.80)
#   validLen = math.ceil(classLen*0.15)
#
#   randFnt = np.random.randint(low = 1, high = classLen, size = classLen)
#   randTrain = randFnt[:trainLen]
#   randValid = randFnt[trainLen : trainLen+validLen]
#   randTest = randFnt[trainLen+validLen :]
#
#   for imgNo in randTrain:
#     src = base+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
#     des = 'dataset/train/Sample'+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
#     copyfile(src, des)
#
#   for imgNo in randValid:
#     src = base+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
#     des = 'dataset/valid/Sample'+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
#     copyfile(src, des)
#
#   for imgNo in randTest:
#     src = base+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
#     des = 'dataset/test/Sample'+str(char).zfill(3)+'/img'+str(char).zfill(3)+'-'+str(imgNo).zfill(5)+'.png'
#     copyfile(src, des)
# def data_ready():
#     for i in os.listdir(dir_data):
#         sub_dir = os.path.join(dir_data, i)
#         for j in os.listdir(sub_dir):
#             img = cv2.imread(os.path.join(sub_dir, j))
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             img = np.array(gray).reshape(128, 128, 1)
#             train_data.append(img)  # append into array and add label into each image
#
#     x_train, x_test = train_test_split(train_data, test_size=0.2, random_state=42)
#     print(x_train[0].shape)
#     return x_train, x_test

# to read the folder and its file one by one and append it into array
# for i in os.listdir(dir_train_data):
#     sub_dir = os.path.join(dir_train_data, i)
#     # count = 0
#     os.makedirs(path)
#     for j in os.listdir(sub_dir):
#         # count += 1
#         # if count > 1:
#         #     break
#         img = cv2.imread(os.path.join(sub_dir, j), 0)  # to loads an image from the specified file and returns it
#         print(img)
#         img = cv2.resize(img, (img_size, img_size))  # changing dimension of the image to the desire size
#         _, thresh_train = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#         # os.chdir(directory)
#         cv2.imwrite("{}/{}.jpg".format(path, j), thresh_train)
#         train_data.append([thresh_train, i])  # append into array and add label into each image


# train_images = []
# train_labels = []
# for feature, label in train_data:
#     train_images.append(feature)
#     train_labels.append(label)
#
# # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
# LB = LabelBinarizer()
# train_labels = LB.fit_transform(train_labels)
#
# # Split the train and the validaiton for the fitting
# random_seed = 10
# train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2,
#                                                                       random_state=random_seed)
#
# # # Normalization
# # # train_x = train_x / 255.0
# train_images = np.array(train_images) / 255.0
# val_images = np.array(val_images) / 255.0
# #
# # print(train_x)
# #
# # # Reshape
# train_images = train_images.reshape(-1, 32, 32, 1)
# val_images = val_images.reshape(-1, 32, 32, 1)
# #
# train_labels = np.array(train_labels)
# val_labels = np.array(val_labels)

# return train_images, train_labels, val_images, val_labels
