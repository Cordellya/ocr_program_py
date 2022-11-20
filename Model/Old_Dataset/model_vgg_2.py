import os

import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dense, MaxPooling2D, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer

# from Data_Ready import data_ready

dir_train_data = "../../dataset/data/training_data"
dir_val_data = "../../dataset/data/testing_data"
train_data = []
val_data = []
img_size = 224

# to read the folder and its file one by one and append it into array
for i in os.listdir(dir_train_data):
    sub_dir = os.path.join(dir_train_data, i)
    count = 0
    for j in os.listdir(sub_dir):
        count += 1
        if count > 400:
            break
        img = cv2.imread(os.path.join(sub_dir, j), 0)  # to loads an image from the specified file and returns it
        img = cv2.resize(img, (img_size, img_size))  # changing dimension of the image to the desire size
        _, thresh_train = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        train_data.append([thresh_train, i])  # append into array and add label into each image

# to read the folder and its file one by one and append it into array
for i in os.listdir(dir_val_data):
    sub_dir = os.path.join(dir_val_data, i)
    count = 0
    for j in os.listdir(sub_dir):
        count += 1
        if count > 400:
            break
        img = cv2.imread(os.path.join(sub_dir, j), 0)  # to loads an image from the specified file and returns it
        img = cv2.resize(img, (img_size, img_size))  # changing dimension of the image to the desire size
        _, thresh_val = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        val_data.append([thresh_val, i])  # append into array and add label into each image

# to seperate features and labels of each images
train_images = []
train_labels = []
for feature, label in train_data:
    train_images.append(feature)
    train_labels.append(label)

val_images = []
val_labels = []
for feature, label in val_data:
    val_images.append(feature)
    val_labels.append(label)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
LB = LabelBinarizer()
train_labels = LB.fit_transform(train_labels)
val_labels = LB.fit_transform(val_labels)
# # Normalization
# # train_x = train_x / 255.0
train_images = np.array(train_images) / 255.0
val_images = np.array(val_images) / 255.0
# # Reshape
train_images = train_images.reshape(-1, 224, 224, 1)
val_images = val_images.reshape(-1, 224, 224, 1)

train_labels = np.array(train_labels)
val_labels = np.array(val_labels)

def create_model():
    model = Sequential()

    model.add(Conv2D(input_shape=(224, 224, 1), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=36, activation="softmax"))

    # model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model


# train_images, train_labels, val_images, val_labels = data_ready()

model = create_model()
model.summary()

checkpoint_path = "training_200/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model.fit(train_images, train_labels,
          epochs=200, batch_size=32,
          validation_data=(val_images, val_labels),
          verbose=1,
          callbacks=[cp_callback])

os.listdir(checkpoint_dir)

loss, acc = model.evaluate(val_images, val_labels, verbose=2)

print(loss, acc)

model.save('model_200.h5')
