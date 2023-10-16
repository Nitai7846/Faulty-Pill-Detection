#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 19:46:55 2023

@author: nitaishah
"""

import numpy as np
import pandas as pd
import pickle
import os
import cv2
import random
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

dir = 'Directory where both Augmented Folders are Saved'

categories = ['Path of Augmented-Defective Images', 'Path of Augmented-Normal Images']

data = []

for category in categories:
    path = os.path.join(dir,category)
    label = categories.index(category)
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        img_arr = cv2.imread(imgpath,0)
        try:
            img_arr = cv2.resize(img_arr,(128,128))
            image = np.array(img_arr).flatten()
            data.append([image,label])
        except Exception as e:
            pass
        
print(len(data))

random.shuffle(data)

features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)



X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)


model_svm = SVC(C=1, kernel='poly', gamma='auto')

model_svm.fit(X_train, y_train)

prediction = model_svm.predict(X_test)

len(prediction)
len(y_test)


cm = confusion_matrix(y_test, prediction)

class_labels = [0, 1]

# Plot the confusion matrix
plt.figure(figsize=(6, 4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(len(class_labels))
plt.xticks(tick_marks, class_labels)
plt.yticks(tick_marks, class_labels)

for i in range(len(class_labels)):
    for j in range(len(class_labels)):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


accuracy_score(y_test, prediction)
recall_score(y_test, prediction)
precision_score(y_test, prediction)
f1_score(y_test, prediction)


