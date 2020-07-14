import os
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import numpy as np
from keras.utils.np_utils import to_categorical


IMG_DIR = 'train/images'
HEIGHT = 224
WIDTH = 224
CHANNEL = 3
BATCH_SIZE = 64

df_train = pd.read_csv('train/train.csv')
df_test = pd.read_csv('test_ApKoW4T.csv')

train_list = list(df_train.image)
test_list = list(df_test.image)

train_imgs = ['train/images/{}'.format(i) for i in os.listdir(IMG_DIR) if i in train_list]
test_imgs = ['train/images/{}'.format(i) for i in os.listdir(IMG_DIR) if i in test_list]
random.shuffle(train_imgs)

y = []
for name in train_imgs:
    y.append((df_train[df_train['image'] == name[13:]]['category']).iloc[0])

def read_and_process_image(image_list):
    '''
    Takes in a list contaning path to image file,
    processes the image and returns X training data
    containing resized images
    '''

    X = []

    for image in image_list:
        X.append(cv2.resize(cv2.imread(image, cv2.IMREAD_COLOR), (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC))

    return X

X = read_and_process_image(train_imgs)
X = np.array(X)
y = np.array(y)

from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(y), 
            y)

weights = {}
for label, weight in enumerate(class_weights):
    weights[label] = weight


y = to_categorical(y-1, num_classes=5)

#plt.figure(figsize=(20, 10))
#columns = 5
#for i in range(columns):
#        plt.subplot(5 / columns + 1, columns, i + 1)
#        plt.imshow(X[i])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=41, stratify=y)

from keras.applications import Xception
conv_base = Xception(input_shape=(WIDTH, HEIGHT, 3), include_top=False, weights='imagenet')

from keras import models
from keras import layers
from keras.metrics import categorical_accuracy

x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Flatten()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(rate=0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(rate=0.25)(x)
outputs = layers.Dense(5, activation='softmax')(x)

model = models.Model(input = conv_base.input, output = outputs)

for index, layer in enumerate(model.layers):
    print(index, layer.name, layer.trainable)


for layer in model.layers[:132]:
    layer.trainable = False


from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

n_train = X_train.shape[0]
n_val = X_val.shape[0]

model.compile(optimizer=optimizers.Adadelta(lr=0.1, rho=0.95), loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range=20,
                                        width_shift_range=0.2, height_shift_range=0.2,
                                        shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)


history = model.fit_generator(train_generator, steps_per_epoch=n_train//BATCH_SIZE,
                                epochs=5, validation_data=val_generator, validation_steps=n_val//BATCH_SIZE, class_weight=weights)

BATCH_SIZE = 16

for layer in model.layers[:16]:
    layer.trainable = False

model.compile(optimizer=optimizers.Adadelta(lr=0.01, rho=0.95), loss='categorical_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range=20,
                                        width_shift_range=0.2, height_shift_range=0.2,
                                        shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

history = model.fit_generator(train_generator, steps_per_epoch=n_train//BATCH_SIZE,
                                epochs=10, validation_data=val_generator, validation_steps=n_val//BATCH_SIZE, class_weight=weights)

BATCH_SIZE = 8

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=optimizers.SGD(lr=0.00001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])


train_datagen = ImageDataGenerator(rescale = 1./255, rotation_range=20,
                                        width_shift_range=0.2, height_shift_range=0.2,
                                        shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

history = model.fit_generator(train_generator, steps_per_epoch=n_train//BATCH_SIZE,
                                epochs=10, validation_data=val_generator, validation_steps=n_val//BATCH_SIZE, class_weight=weights)


#model.save_weights('model_97_att.h5')
#model.load_weights('model_97_att.h5')
X_test = read_and_process_image(test_imgs)
X_test = np.array(X_test)
X_test = X_test / 255.0

prediction = model.predict(X_test)
prediction = np.argmax(prediction, axis=1)
prediction = prediction + 1

df_sub = pd.DataFrame(data={'image':df_test['image'], 'category': prediction})
df_sub.to_csv('Submission_6.csv', index=False)
