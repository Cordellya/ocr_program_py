import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import random
import cv2
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from keras.models import Sequential
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
from keras.layers import Dense, Activation, Flatten, Dense, MaxPooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf

dir_data = "dataset/data/training_data"
train_data = []
img_size = 224

# to read the folder and its file one by one and append it into array
for i in os.listdir(dir_data):
    sub_dir = os.path.join(dir_data, i)
    count = 0
    for j in os.listdir(sub_dir):
        count += 1
        if count > 10:
            break
        img = cv2.imread(os.path.join(sub_dir, j), 0)  # to loads an image from the specified file and returns it
        img = cv2.resize(img, (img_size, img_size))  # changing dimension of the image to the desire size
        train_data.append([img, i])  # append into array and add label into each image

# # df_train = pd.DataFrame(np.array(train_data))
# #
# # print(df_train)
#
# # train_y = df_train['label']
# # train_x = df_train.drop(labels=['label'], axis=1)
#
# to seperate features and labels of each images
train_images = []
train_labels = []
for feature, label in train_data:
    train_images.append(feature)
    train_labels.append(label)

# Split the train and the validaiton for the fitting
random_seed = 2
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2,
                                                                      random_state=random_seed)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
LB = LabelBinarizer()
train_labels = LB.fit_transform(train_labels)
val_labels = LB.fit_transform(val_labels)
#
#
#
# # Normalization
# # train_x = train_x / 255.0
train_images = np.array(train_images) / 255.0
val_images = np.array(val_images) / 255.0
#
# print(train_x)
#
# # Reshape
train_images = train_images.reshape(-1, 224, 224, 1)
val_images = val_images.reshape(-1, 224, 224, 1)
#
train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

print(val_images, val_labels)

#
# print(train_images.shape, val_images.shape)
# print(train_labels.shape, val_labels.shape)
# def create_model():
#     model = Sequential()
#
#     model.add(Conv2D(input_shape=(224, 224, 1), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
#     model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
#     model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
#     model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
#     model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
#
#     model.add(Flatten())
#     model.add(Dense(units=4096, activation="relu"))
#     model.add(Dense(units=4096, activation="relu"))
#     model.add(Dense(units=36, activation="softmax"))
#
#     return model
#
#
# model = create_model()
# model.summary()
#
# checkpoint_path = "training_1/cp.ckpt"
# checkpoint_dir = os.path.dirname(checkpoint_path)
#
# cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
#                                                  save_weights_only=True,
#                                                  verbose=1)
#
# model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
# # model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
#
# model.fit(train_images, train_labels,
#                     epochs=50, batch_size=32,
#                     validation_data=(val_images, val_labels),
#                     verbose=1,
#                     callbacks=[cp_callback])
#
# os.listdir(checkpoint_dir)
#
# loss, acc = model.evaluate(val_images, val_labels, verbose=2)
#
# print(loss, acc)
#
# model.save('model_50.h5')
#
