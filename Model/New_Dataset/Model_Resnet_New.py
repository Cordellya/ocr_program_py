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
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# from Data_Ready import data_ready
# from Data_Ready import data_ready

dir_data = "../../dataset/data_new/All"
train_data = []
val_data = []
batch_size = 32
epochs = 100
IMG_HEIGHT = 32
IMG_WIDTH = 32

print(len(os.listdir(dir_data)))
# data_size = 400

# data_train, _ = data_ready()

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
                                                         directory=dir_data,
                                                         color_mode="grayscale",
                                                         shuffle=True,
                                                         target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                         class_mode="categorical",
                                                         seed=42,  # 65657867
                                                         subset='training')

val_data_gen = normal_image_gen.flow_from_directory(batch_size=batch_size,
                                                    directory=dir_data,
                                                    color_mode="grayscale",
                                                    shuffle=True,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    class_mode="categorical",
                                                    seed=42,
                                                    subset='validation')


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

checkpoint_path = "pretrained_resnetnew_100_aug/cp.ckpt"
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

np.save('history_resnetnew_100_aug.npy', history.history)

resnet_model.save('model_resnetnew_100_aug.h5')




# train_data_gen = augmented_image_gen.flow(np.array(data_train),
#                                           shuffle=True,
#                                           batch_size= batch_size,
#                                           seed=42,  # 65657867
#                                           subset='training')
#
# val_data_gen = augmented_image_gen.flow(np.array(data_train),
#                                           shuffle=True,
#                                           batch_size= batch_size,
#                                           seed=42,  # 65657867
#                                           subset='validation')


# test_data_gen = normal_image_gen.flow_from_directory(batch_size=batch_size,
#                                                     directory=dir_data,
#                                                     color_mode="grayscale",
#                                                     shuffle=True,
#                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                     class_mode="categorical",
#                                                     seed=42,
#                                                     subset='validation')
# print(train_data_gen[0])
# print(val_data_gen[0].shape)