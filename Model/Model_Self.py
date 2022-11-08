import os

import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, MaxPooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
from keras.optimizers import Adam, SGD
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

dir_train_data = "../dataset/data/training_data"
dir_val_data = "../dataset/data/testing_data"
batch_size = 32
epochs = 50
IMG_HEIGHT = 28
IMG_WIDTH = 28


def preprocessing_fun(img):
    #     print(img.shape)
    #     print(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = img.reshape((28, 28, 1))
    thresh = cv2.adaptiveThreshold(img, 255, 1, 1, 11, 2)
    print(thresh.shape)


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
                                                         seed=65657867,
                                                         subset='training')
val_data_gen = normal_image_gen.flow_from_directory(batch_size=batch_size,
                                                    directory=dir_val_data,
                                                    color_mode="grayscale",
                                                    shuffle=True,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    class_mode="categorical",
                                                    seed=65657867,
                                                    subset='validation')


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(36, activation='softmax'))
    return model


model = define_model()

checkpoint_path = "training_self_50/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

EarlyStop_callback = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
checkpoint = ModelCheckpoint(checkpoint_path,
                             monitor='val_loss', mode='min', save_best_only=True)
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001)
my_callback = [EarlyStop_callback, checkpoint]

model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['categorical_accuracy'])

history = model.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.samples // batch_size,
    callbacks=my_callback)

np.save('history_self_50.npy', history.history)

model.save('model_self_50.h5')
