import os

import cv2
import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

IMG_HEIGHT = IMG_WIDTH = 32

normal_image_gen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True
)
test_data_gen = normal_image_gen.flow_from_directory(batch_size=5193,
                                                     directory="dataset/data/testing_data",
                                                     color_mode="grayscale",
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode="categorical")

model = tf.keras.models.load_model('saved_model/model_resnet_50_3.h5')

test_images, test_labels = next(test_data_gen)
filenames = test_data_gen.filenames
test_pred = model.predict(test_images)

true_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
                'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

true_labels = tf.argmax(test_labels,1)
predicted_labels = tf.argmax(test_pred,1)

print(true_labels)
print(predicted_labels)

arr_ytest = []
for label_test in true_labels:
    arr_ytest.append(true_classes[label_test])

arr_ypred = []
for pred_label in predicted_labels:
    arr_ypred.append(true_classes[pred_label])

print(arr_ytest, '\n', arr_ypred)
# print(cm)
# df_cm = pd.DataFrame(cm, index=[i for i in true_labels], columns=[i for i in predicted_labels])
# plt.figure(figsize=(10,10))
# sns.heatmap(df_cm, annot=True)
# plt.show()


arr_ypred2 = []
arr_ytest2 = []
for i in os.listdir("dataset/data/testing_data"):
    sub_dir = os.path.join("dataset/data/testing_data", i)
    count = 0
    for j in os.listdir(sub_dir):
        count += 1
        if count > 10:
            break
        img = cv2.imread(os.path.join(sub_dir, j), 0)  # to loads an image from the specified file and returns it
        img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))  # changing dimension of the image to the desire size
        # thresh_test, _ = image_processing(img)  # append into array and add label into each image
        # res_test = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        res_test = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        res_test = img.astype("float32") / 255.0
        res_test = np.expand_dims(res_test, axis=-1)
        res_test = res_test.reshape(1, IMG_HEIGHT, IMG_WIDTH, 1)
        ypred_test = model.predict(res_test)
        # print(ypred_test)
        ypred_test = np.argmax(ypred_test, axis=1)
        # print(ypred_test[0])
        arr_ypred2.append(true_classes[ypred_test[0]])
        arr_ytest2.append(i)

print(arr_ytest2, '\n', arr_ypred2)

cm = confusion_matrix(arr_ytest, arr_ypred)

df_cm = pd.DataFrame(cm, index=[i for i in true_classes], columns=[i for i in true_classes])
plt.figure(figsize=(10,10))
sns.heatmap(df_cm, annot=True)
plt.show()

cm2 = confusion_matrix(arr_ytest2, arr_ypred2)
df_cm2 = pd.DataFrame(cm2, index=[i for i in true_classes], columns=[i for i in true_classes])
plt.figure(figsize=(10,10))
sns.heatmap(df_cm2, annot=True)
plt.show()
# print(cm)
# plt.figure(figsize = (10,10))
#
# sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='BuPu_r');
# plt.show()

print(classification_report(arr_ytest, arr_ypred, zero_division=0))
print(accuracy_score(arr_ytest, arr_ypred))

print(classification_report(arr_ytest2, arr_ypred2, zero_division=0))
print(accuracy_score(arr_ytest2, arr_ypred2))

history = np.load('saved_model/history_resnet_50_3.npy', allow_pickle='TRUE').item()

plt.plot(history['categorical_accuracy'], color='black')
plt.plot(history['val_categorical_accuracy'], color='grey')
plt.title('Training Accuracy vs Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history['loss'], color='black')
plt.plot(history['val_loss'], color='grey')
plt.title('Training Loss vs Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()