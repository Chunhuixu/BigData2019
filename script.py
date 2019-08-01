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

        
```
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
#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8) 
model.compile(loss='binary_crossentropy',
             optimizer=optimizers.Adam(lr=1e-4),
             metrics=['acc'])
model.summary()
```
```
history = model.fit(
    training_set,
    train_labels,
    epochs= 30,
    validation_data=(validation_set,validation_labels),
)
```



acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()






 # Use data augmentation
train_datagen = preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)
train_generator = train_datagen.flow(
    training_set,
    train_labels,
    batch_size=32)

# do not augment validation data
test_datagen = preprocessing.image.ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow(
    validation_set,
    validation_labels,
    batch_size=32)

# train
history = model.fit_generator(
    train_generator,
    steps_per_epoch=10,
    epochs= 30
    validation_data=validation_generator,
    validation_steps=10)

