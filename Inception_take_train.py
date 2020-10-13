# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 21:13:26 2020

@author: param

"""

import tensorflow as tf
import keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2
from matplotlib import pyplot as plt

INIT_LR = 1e-4
EPOCHS = 20
BS = 32

DIRECTORY = "C:/Users/param/Face-Mask-Detection/dataset"
CATEGORIES = ["with_mask", "without_mask"]

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
data = np.array(data, dtype="float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

aug = ImageDataGenerator(rescale=1. / 255,
                         rotation_range=40,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode='nearest')

# test_dataset = train.flow_from_directory('C:/Users/param/Face-Mask-Detection/dataset/test',
# target_size=(224, 224),
# batch_size=32,
# class_mode='binary')
# train_dataset.class_indices

# local_weights_file = 'C:/Users/param/Face-Mask-Detection/face_detector/inceptionv3-model-10ep.h5'
pre_trained_model = NASNetMobile(weights="imagenet", include_top=False,
                                input_tensor=Input(shape=(224, 224, 3)))

# pre_trained_model.load_weights(local_weights_file)
#

#  
pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
# print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output
# construct the head of the model that will be placed on top of the
# the base model
# headModel = pre_trained_model.output
# headModel = AveragePooling2D(pool_size=(5, 5))(last_output)
# headModel = BatchNormalization(axis=1)(last_output)
# headModel = Conv2D(32, (3, 3), padding="same", activation="relu")(headModel)
headModel = AveragePooling2D(pool_size=(5, 5))(last_output)
headModel = Flatten(name="flatten")(headModel)
# headModel = Dropout(0.5)(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
#headModel = BatchNormalization(axis=1)(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.2)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=pre_trained_model.input, outputs=headModel)
model.summary()
for layer in pre_trained_model.layers:
    layer.trainable = False
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process


# compile our model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# train the head of the network
print("[INFO] training head...")
History = model.fit(
    aug.flow(trainX,trainY,batch_size=32),
    steps_per_epoch=len(trainX) // BS,  # 2000 images = batch_size * steps
    epochs=20,
    validation_data=(testX,testY),
    validation_steps=len(testX) // BS,  # 1000 images = batch_size * steps
    verbose=2)
# make predictions on the testing set
# print("[INFO] evaluating network...")
# predIdxs = model.predict(test_dataset)

# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
# predIdxs = np.argmax(predIdxs, axis=1)

# show a nicely formatted classification report
# print(predIdxs)
#	target_names=lb.classes_)

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("C:/Users/param/Face-Mask-Detection/incpv3_take21", save_format="h5")

# plot the training loss and accuracy
# N = EPOCHS
# plt.style.use("ggplot")
# plt.figure()
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig(args["plot"])
