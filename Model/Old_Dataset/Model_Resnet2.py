import os

import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, MaxPooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# from Data_Ready import data_ready

dir_train_data = "../../dataset/data/training_data"
dir_val_data = "../../dataset/data/testing_data"
train_data = []
val_data = []
batch_size = 32
epochs = 150
IMG_HEIGHT = 32
IMG_WIDTH = 32
# data_size = 400

augmented_image_gen = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=2,
    width_shift_range=.1,
    height_shift_range=.1,
    zoom_range=0.1,
    shear_range=2,
    brightness_range=[0.9, 1.1],
    validation_split=0.2,
)

normal_image_gen = ImageDataGenerator(
    rescale=1 / 255.0,
    validation_split=0.2,
)

train_data_gen = augmented_image_gen.flow_from_directory(batch_size=batch_size,
                                                         directory=dir_train_data,
                                                         color_mode="grayscale",
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode="categorical",
                                                         seed=42,  # 65657867
                                                         subset='training')
val_data_gen = normal_image_gen.flow_from_directory(batch_size=batch_size,
                                                    directory=dir_train_data,
                                                    color_mode="grayscale",
                                                    shuffle=True,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    class_mode="categorical",
                                                    seed=42,
                                                    subset='validation')

# print(train_data_gen[0])
resnet_model = Sequential()

pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                  input_shape=(IMG_HEIGHT, IMG_WIDTH, 1),
                                                  classes=36,
                                                  pooling='avg',
                                                  weights=None)

resnet_model.add(pretrained_model)
resnet_model.add(Flatten())
resnet_model.add(Dense(512, activation='relu', input_dim=2048))
resnet_model.add(Dense(256, activation='relu'))
resnet_model.add(Dropout(0.5))
resnet_model.add(BatchNormalization())
resnet_model.add(Dense(512, activation='relu'))
resnet_model.add(Dropout(0.5))
resnet_model.add(BatchNormalization())
resnet_model.add(Dense(128, activation='relu'))
resnet_model.add(Dropout(0.5))
resnet_model.add(BatchNormalization())
resnet_model.add(Dense(36, activation='softmax'))

checkpoint_path = "pretrained_resnet2_100_aug_new/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 mode='min',
                                                 monitor='val_loss',
                                                 verbose=1)

# EarlyStop_callback = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.0001)
# my_callback = [EarlyStop_callback, cp_callback]

resnet_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                     optimizer=Adam(learning_rate=0.001),
                     metrics=['categorical_accuracy'])

# resnet_model.summary()

history = resnet_model.fit(train_data_gen,
                           epochs=epochs,
                           steps_per_epoch=train_data_gen.samples // batch_size,
                           validation_steps=val_data_gen.samples // batch_size,
                           validation_data=val_data_gen,
                           verbose=1,
                           shuffle=True,
                           callbacks=[cp_callback])

os.listdir(checkpoint_dir)

# loss, acc = resnet_model.evaluate(val_images, val_labels, verbose=2)
#
# print(loss * 100, acc * 100)

np.save('history_resnet2_150_aug_new.npy', history.history)

resnet_model.save('model_resnet2_150_aug_new.h5')

# train_images = []
# train_labels = []
# for feature, label in train_data:
#     train_images.append(feature)
#     train_labels.append(label)
#
# # val_images = []
# # val_labels = []
# # for feature, label in val_data:
# #     val_images.append(feature)
# #     val_labels.append(label)
#
# # Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
# LB = LabelBinarizer()
# train_labels = LB.fit_transform(train_labels)
# # val_labels = LB.fit_transform(val_labels)
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

# print(len(val_labels))

# for layer in pretrained_model.layers:
#         layer.trainable=False
