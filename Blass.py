"""
from __future__
To ensure that future statements run under releases prior to 2.1 at least yield runtime exceptions
(the import of __future__ will fail, because there was no module of that name prior to 2.1).
"""

"""
Imports
"""


"""
Data transform
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import cv2
import sklearn
import random
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
train_dir = "chest_xray/train/"
test_dir = "chest_xray/test/"

LOAD_FROM_IMAGES = False


def get_data(folder):
    X = []
    y = []
    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['NORMAL']:
                label = 0
            elif folderName in ['PNEUMONIA']:
                label = 1
            else:
                label = 2
            for image_filename in tqdm(os.listdir(folder + folderName)):
                img_file = cv2.imread(
                    folder + folderName + '/' + image_filename)
                if img_file is not None:
                    img_file = skimage.transform.resize(
                        img_file, (150, 150, 3), mode='constant', anti_aliasing=True)
                    img_file = rgb2gray(img_file)
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


if LOAD_FROM_IMAGES:
    X_train, y_train = get_data(train_dir)
    X_test, y_test = get_data(test_dir)

    np.save('xtrain.npy', X_train)
    np.save('ytrain.npy', y_train)
    np.save('xtest.npy', X_test)
    np.save('ytest.npy', y_test)
else:
    X_train = np.load('xtrain.npy')
    y_train = np.load('ytrain.npy')
    X_test = np.load('xtest.npy')
    y_test = np.load('ytest.npy')

"""
Model
"""
X_trainReshaped = X_train.reshape(len(X_train),150,150,1)
X_testReshaped = X_test.reshape(len(X_test),150,150,1)

model = models.Sequential()
model.add(layers.Conv2D(32, (1, 1), activation='relu', input_shape=(150, 150, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (7, 7), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

model.summary()

"""
Visualization images
"""
def plotHistogram(a):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(a.ravel(), bins=255)
    plt.subplot(1, 2, 2)
    plt.imshow(a, cmap='gray', vmin=0, vmax=1)
    plt.show()

plotHistogram(X_train[1])

"""
Verification of the amount of data
Label 0: without PNEUMONIA
Label 1: Whit PNEUMONIA
"""
plt.figure(figsize=(8, 4))
map_characters = {0: 'without PNEUMONIA', 1: ' Whit PNEUMONIA'}
dict_characters = map_characters
df = pd.DataFrame()
df["labels"] = y_train
lab = df['labels']
dist = lab.value_counts()
sns.countplot(lab)
print(dict_characters)

"""
Model compile
"""
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)
model.fit(X_trainReshaped, 
          y_train, 
          epochs=35,
          validation_data = (X_testReshaped,y_test),
          callbacks=[tensorboard_callback])
