import itertools
import os

import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# train_images, train_labels, val_images, val_labels = data_ready()

dir_test = "dataset/data/training_data"

# true_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
#                 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

true_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
                 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

# test_data = []
# for i in os.listdir(dir_test):
#     sub_dir = os.path.join(dir_test, i)
#     count = 0
#     for j in os.listdir(sub_dir):
#         count += 1
#         if count > 8:
#             break
#         img = cv2.imread(os.path.join(sub_dir, j), 0)  # to loads an image from the specified file and returns it
#         img = cv2.resize(img, (32, 32))  # changing dimension of the image to the desire size
#         _, thresh_train = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#         test_data.append([thresh_train, i])  # append into array and add label into each image

model = tf.keras.models.load_model('saved_model/model_resnet_50_35.h5')

arr_ypred = []
arr_ytest = []
for i in os.listdir(dir_test):
    sub_dir = os.path.join(dir_test, i)
    count = 0
    if i == 'I':
        continue
    for j in os.listdir(sub_dir):
        count += 1
        if count > 5:
            break
        img = cv2.imread(os.path.join(sub_dir, j), 0)  # to loads an image from the specified file and returns it
        img = cv2.resize(img, (32, 32))  # changing dimension of the image to the desire size
        # thresh_test, _ = image_processing(img)  # append into array and add label into each image
        # res_test = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
        res_test = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        res_test = img.astype("float32") / 255.0
        res_test = np.expand_dims(res_test, axis=-1)
        res_test = res_test.reshape(1, 32, 32, 1)
        ypred_test = model.predict(res_test)
        # print(ypred_test)
        ypred_test = np.argmax(ypred_test, axis=1)
        print(ypred_test[0])
        arr_ypred.append(true_classes[ypred_test[0]])
        arr_ytest.append(i)
        # print(i, true_classes[ypred_test[0] - 1])
        # print(i, ypred_test[0])

print(arr_ytest, "\n", arr_ypred)

cm = confusion_matrix(arr_ypred, arr_ytest)

df_cm = pd.DataFrame(cm, index=[i for i in true_classes], columns=[i for i in true_classes])
plt.figure(figsize=(10,10))
sns.heatmap(df_cm, annot=True)
plt.show()

# print(cm)
# plt.figure(figsize = (10,10))
#
# sns.heatmap(cm, annot=True, fmt=".2f", linewidths=.5, square=True, cmap='BuPu_r');
# plt.show()

print(classification_report(arr_ytest, arr_ypred, zero_division=0))
print(accuracy_score(arr_ytest, arr_ypred))

history = np.load('saved_model/history_resnet_50_35.npy', allow_pickle='TRUE').item()

plt.plot(history['accuracy'], color='black')
plt.plot(history['val_accuracy'], color='grey')
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


# Y_pred = model.predict(val_images[0])
# # Convert predictions classes to one hot vectors
# Y_pred_classes = np.argmax(Y_pred, axis=1)
# # Convert validation observations to one hot vectors
# Y_true = np.argmax(val_labels[0], axis=1)
# # compute the confusion matrix
# confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)

# plot_confusion_matrix(confusion_mtx, classes=range(10))

# loss, acc = model.evaluate(val_images, val_labels, verbose=2)
# print(loss, acc)
#


# test1= "dataset/data/testing_data/0/28310.png"
# img = cv2.imread(test1, 0)  # to loads an image from the specified file and returns it
# img = cv2.resize(img, (32, 32))  # changing dimension of the image to the desire size
# # thresh_test, _ = image_processing(img)  # append into array and add label into each image
# # res_test = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
# res_test = img.astype("float32") / 255.0
# res_test = np.expand_dims(res_test, axis=-1)
# res_test = res_test.reshape(1, 32, 32, 1)
# ypred_test = model.predict(res_test)
# print(ypred_test)
# ypred_test = np.argmax(ypred_test, axis=1)
# print(true_classes[ypred_test[0]])

# plot_confusion_matrix(model, X_test, y_test, cmap='GnBu')
# plt.show()
# print('Precision: %.3f' % precision_score(y_test, y_pred))
# print('Recall: %.3f' % recall_score(y_test, y_pred))
# print('F1: %.3f' % f1_score(y_test, y_pred))
# print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
