from google.colab import drive
drive.mount('/content/drive')
!unzip -uq "/content/drive/My Drive/BigData/training_set.zip" -d "/content/drive/My Drive/training"

import os
import tqdm
import matplotlib.pyplot as plt
from keras import preprocessing, layers, models, optimizers
import numpy as np


path_cats = []
train_path_cats = '../input/training_set/training_set/cats'
for path in os.listdir(train_path_cats):
    if '.jpg' in path:
        path_cats.append(os.path.join(train_path_cats, path))
path_dogs = []
train_path_dogs = '../input/training_set/training_set/dogs'
for path in os.listdir(train_path_dogs):
    if '.jpg' in path:
        path_dogs.append(os.path.join(train_path_dogs, path))
len(path_dogs), len(path_cats)



training_set = np.zeros((6000, 150, 150, 3), dtype='float32')
for i in range(6000):
    if i < 3000:
        path = path_dogs[i]
        img = preprocessing.image.load_img(path, target_size=(150, 150))
        training_set[i] = preprocessing.image.img_to_array(img)
    else:
        path = path_cats[i - 3000]
        img = preprocessing.image.load_img(path, target_size=(150, 150))
        training_set[i] = preprocessing.image.img_to_array(img)

        
        model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                       input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])
