# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 00:11:57 2020

@author: param
"""
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import os
import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()
maskNet = load_model('C:/Users/param/Face-Mask-Detection/incpv3_take20')



def detect_and_predict_mask(frame, maskNet):
    # faces = []
    # locs = []
    # preds = []

        (h, w) = frame.shape[:2]
        faces = []
        locs = []
        preds = []
        # while True:
        # Capture frame-by-frame
        # __, frame = cap.read()
        # faces = []
        # locs = []
        # preds = []

        # Use MTCNN to detect faces
        result = detector.detect_faces(frame)
        # if result != []:
        for person in result:

            bounding_box = person['box']
            keypoints = person['keypoints']
                # cv2.rectangle(frame,
                #               (bounding_box[0], bounding_box[1]),
                #               (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                #               (0,155,255),
                #               2)
            (startX, startY, endX, endY) = np.array(bounding_box, dtype=np.int)
            # (startX, startY) = (max(0, startX), max(0, startY))
            (endX1, endY1) = (startX+endX,startY+endY)
            face = frame[startY:endY1,startX:endX1]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            faces.append(face)
            locs.append((startX, startY, endX, endY))
        if len(faces) > 0:
            preds = maskNet.predict(faces)
        return locs, preds




    # for faster inference we'll make batch predictions on *all*
    # faces at the same time rather than one-by-one predictions
    # in the above `for` loop


# return a 2-tuple of the face locations and their corresponding
# locations

vs = VideoStream(src=0).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX + startX, endY + startY), color, 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


