import os

import cv2
import numpy as np
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import tensorflow as tf
from keras.optimizers import Adam
from keras_preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

dir_train_data = "../../dataset/data/training_data"
dir_val_data = "../../dataset/data/testing_data"
train_data = []
val_data = []
batch_size = 32
epochs = 100
IMG_HEIGHT = 32
IMG_WIDTH = 32
# data_size = 400

augmented_image_gen = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=2,
    width_shift_range=.1,
    height_shift_range=.1,
    zoom_range=0.1,
    # shear_range=2,
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
                                                         seed=42, #65657867
                                                         subset='training')
val_data_gen = normal_image_gen.flow_from_directory(batch_size=batch_size,
                                                    directory=dir_train_data,
                                                    color_mode="grayscale",
                                                    shuffle=True,
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    class_mode="categorical",
                                                    seed=42,
                                                    subset='validation')

vgg_model = Sequential()

pretrained_model = tf.keras.applications.VGG16(include_top=False,
                                               input_shape=(IMG_HEIGHT, IMG_WIDTH, 1),
                                               pooling='max',
                                               classes=36,
                                               weights=None,
                                               classifier_activation="softmax",
                                               )

vgg_model.add(pretrained_model)
vgg_model.add(Flatten())
vgg_model.add(Dense(units=4096, activation="relu"))
vgg_model.add(Dense(units=4096, activation="relu"))
vgg_model.add(Dense(units=36, activation="softmax"))

checkpoint_path = "pretrained_vgg_100_aug/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

vgg_model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                     optimizer=Adam(learning_rate=0.01),
                     metrics=['categorical_accuracy'])

history = vgg_model.fit(train_data_gen,
                 epochs=epochs,
                 steps_per_epoch=train_data_gen.samples // batch_size,
                 validation_steps=val_data_gen.samples // batch_size,
                 validation_data=val_data_gen,
                 verbose=1,
                 callbacks=[cp_callback])

os.listdir(checkpoint_dir)

# loss, acc = vgg_model.evaluate(val_images, val_labels, verbose=2)

# print(loss * 100, acc * 100)

vgg_model.save('model_vgg_100_aug.h5')



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


# model = create_model()
# model.summary()