import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
import numpy as np

from Cropping_New import final_crop

def prediction(im):
    IMG_SIZE = 32

    true_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    model = tf.keras.models.load_model('Model/saved_model/resnet_new/model_resnet2_100_aug_new.h5')

    # image = cv2.imread(im)

    char_images = final_crop(im)

    letters=[]

    for char_image in char_images:
        char_image = cv2.resize(char_image, (IMG_SIZE, IMG_SIZE))
        # # plt.imshow(res)
        # # plt.show()
        char_image = np.array(char_image, dtype=np.float32) / 255.0
        char_image = np.expand_dims(char_image, axis=-1)
        char_image = char_image.reshape(1, IMG_SIZE, IMG_SIZE, 1)
        # np.append(crop, res)
        ypred = model.predict(char_image)
        ypred = np.argmax(ypred, axis=1)
        # ypred = train_labels[ypred]
        # ypred = LB.inverse_transform(ypred)
        print(ypred[0], true_classes[ypred[0]])
        [x] = true_classes[ypred[0]]
        letters.append(x)
    
    exp_date = "".join(letters[0:7])
    prod_code = "".join(letters[7:])

    
    return [exp_date, prod_code]
    # plt.imshow(characters[0])
    # plt.show()
    # print(char_images)


# prediction("dataset/crop_new/crop (1).jpg")