
import os
import tensorflow as tf
import numpy as np
import classes
import glob
import argparse
import sys
import alexnet
import cv2

dropoutPro = 1
classNum = 1000
skip = []
# Get test images from directory
testPath = "TestModel/"
testImg = []

# Function to read hidden files
def listdir_files(path):
    return glob.glob(os.path.join(path, '*'))

for f in listdir_files(testPath):
    testImg.append(cv2.imread(f))
 
imgMean = np.array([104, 117, 124], np.float)
x = tf.placeholder("float", [1, 227, 227, 3])
 
model = alexnet.alexNet(x, dropoutPro, classNum, skip)
score = model.fc3
softmax = tf.nn.softmax(score)
 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # Loading the model
    model.loadModel(sess) 
    result = []
    for i, img in enumerate(testImg):
        # image preprocessing
        test = cv2.resize(img.astype(float), (227, 227)) 
        test -= imgMean
        test = test.reshape((1, 227, 227, 3))
        maxx = np.argmax(sess.run(softmax, feed_dict = {x: test}))
        res = classes.names[maxx]
        print(res)
        result.append(res)
        font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
        cv2.putText(img, res, (int(img.shape[0]/3), int(img.shape[1]/3)), font, 1, (0, 0, 255), 2) #putting on the labels
        cv2.imshow("demo"+str(i), img) 
        cv2.waitKey(5000)

