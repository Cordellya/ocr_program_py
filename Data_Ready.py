import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# def data_ready():
    # to read the folder and its file one by one and append it into array
    # from Data_Ready import data_ready
dir_train_data = "dataset/data/training_data"
train_data = []
img_size = 224

parent_dir = "D:/Aplikasi_Skripsi/Program_OCR/dataset/data/training224/"

# to read the folder and its file one by one and append it into array
for i in os.listdir(dir_train_data):
    sub_dir = os.path.join(dir_train_data, i)
    # count = 0
    path = os.path.join(parent_dir, i)
    os.makedirs(path)
    for j in os.listdir(sub_dir):
        # count += 1
        # if count > 1:
        #     break
        img = cv2.imread(os.path.join(sub_dir, j), 0)  # to loads an image from the specified file and returns it
        print(img)
        img = cv2.resize(img, (img_size, img_size))  # changing dimension of the image to the desire size
        _, thresh_train = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # os.chdir(directory)
        cv2.imwrite("{}/{}.jpg".format(path, j), thresh_train)
        train_data.append([thresh_train, i])  # append into array and add label into each image



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
