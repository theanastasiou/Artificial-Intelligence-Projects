import os
from keras import datasets, layers, models
import matplotlib.pyplot as plt
from matplotlib import pyplot
from tensorflow.keras.optimizers import SGD
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob
from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

import os
batch_size = 16
epochs = 10
IMG_HEIGHT = 300
IMG_WIDTH = 300

train_image_generator = ImageDataGenerator(rescale=1./255, 
                            rotation_range=15,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            shear_range=0.01,
                            zoom_range=[0.9, 1.25],
                            brightness_range=[0.5, 1.5]) # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=10,
                                                           directory='/home/aarodoeht/Desktop/cnnex/NewTrain',
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory( batch_size=5,
                                                              directory='/home/aarodoeht/Desktop/cnnex/NewValid',
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])
sgd = SGD(lr=1e-6)
model.compile( loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                optimizer=sgd,
              metrics=['accuracy'])
   
model.summary()

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch= train_data_gen.n//train_data_gen.batch_size ,
    epochs=3,
    validation_data=val_data_gen,
    validation_steps=val_data_gen.n // val_data_gen.batch_size
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


test_generator = test_datagen.flow_from_directory(
        '/home/aarodoeht/Desktop/cnnex/NewTest',
        target_size=(200, 200),
        batch_size=1,
        class_mode=None,
        shuffle=False)

score = model.evaluate_generator(val_data_gen,steps=val_data_gen.n // val_data_gen.batch_size)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

STEP_SIZE_TEST = test_generator.n//test_generator.batch_size
test_generator.reset()
pred =model.predict_generator(test_generator, 
                steps =STEP_SIZE_TEST,
                verbose=1)

import numpy as np
import pandas as pd
import glob, shutil
import os, random

predicted_class_indices = np.argmax(pred,axis=1)
labels= (train_generator.class_indices)
labels=dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames = test_generator.filenames 
results = pd.DataFrame({"Filename":filenames, "Predictions":predictions})
results.to_csv("results.csv",index=False)
