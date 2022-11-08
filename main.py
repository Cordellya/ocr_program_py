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

dir_data = "dataset/data/training_data"
train_data = []
img_size = 32

# to read the folder and its file one by one and append it into array
for i in os.listdir(dir_data):
    sub_dir = os.path.join(dir_data, i)
    for j in os.listdir(sub_dir):
        img = cv2.imread(os.path.join(sub_dir, j), 0)  # to loads an image from the specified file and returns it
        img = cv2.resize(img, (img_size, img_size))  # changing dimension of the image to the desire size
        train_data.append([img, i])  # append into array and add label into each image

# df_train = pd.DataFrame(np.array(train_data))
#
# print(df_train)

# train_y = df_train['label']
# train_x = df_train.drop(labels=['label'], axis=1)

# to seperate features and labels of each images
train_x = []
train_y = []
for feature, label in train_data:
    train_x.append(feature)
    train_y.append(label)

# Split the train and the validaiton for the fitting
random_seed = 40
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2, random_state=random_seed)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
LB = LabelBinarizer()
train_y = LB.fit_transform(train_y)
val_y = LB.fit_transform(val_y)

# Normalization
# train_x = train_x / 255.0
train_x = np.array(train_x) / 255.0
val_x = np.array(val_x) / 255.0

print(train_x)

# Reshape
train_x = train_x.reshape(-1, 32, 32, 1)
val_x = val_x.reshape(-1, 32, 32, 1)

train_y = np.array(train_y)
val_y = np.array(val_y)

print(train_x.shape, val_x.shape)
print(train_y.shape, val_y.shape)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding="same", activation='relu', input_shape=(32, 32, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(36, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

history = model.fit(train_x, train_y, epochs=50, batch_size=32, validation_data=(val_x, val_y), verbose=1)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training Accuracy vs Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training Loss vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def get_letters(img):
    letters = []
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(thresh1)
    plt.show()
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    # loop over the contours
    for c in cnts:
        # print(cv2.contourArea(c))
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = gray[y:y + h, x:x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)
        thresh = thresh.astype("float32") / 255.0
        thresh = np.expand_dims(thresh, axis=-1)
        thresh = thresh.reshape(1, 32, 32, 1)
        ypred = model.predict(thresh)
        ypredLB = LB.inverse_transform(ypred)
        ypred = np.argmax(ypred, axis=1)

        [x] = ypred
        letters.append(x)
    return letters, image


def get_word(letter):
    word = "".join(letter)
    return word

#
# letter, image = get_letters("data_uji/dataset13_copy.jpg")
# word = get_word(letter)
# print(word)
# plt.imshow(image)
# plt.show()


test_data = "dataset/data/testing_data"
# model = tf.keras.models.load_model('model_50.h5')
true_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                'L',
                'M', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for i in os.listdir(test_data):
    sub_dir = os.path.join(test_data, i)
    for j in os.listdir(sub_dir):
        img = cv2.imread(os.path.join(sub_dir, j), 0)  # to loads an image from the specified file and returns it
        img = cv2.resize(img, (32, 32))  # changing dimension of the image to the desire size
        # thresh_test, _ = image_processing(img)  # append into array and add label into each image
        # res_test = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        res_test = img.astype("float32") / 255.0
        res_test = np.expand_dims(res_test, axis=-1)
        res_test = res_test.reshape(1, 32, 32, 1)
        ypred_test = model.predict(res_test)
        ypred_test = np.argmax(ypred_test, axis=1)
        print(j, true_classes[ypred_test[0]])
        break

# Example of images
# plt.imshow(train_x[4000][:, :, 0])
# plt.show()
