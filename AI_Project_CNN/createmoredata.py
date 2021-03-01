import os
from keras import datasets, layers, models
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import glob

datagen = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.01,
            zoom_range=[0.9, 1.25],
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='reflect',
            data_format='channels_last',
            brightness_range=[0.3, 1.5])

for filename in glob.glob('/home/aarodoeht/Desktop/cnnex/NewTrain/Covid/*'): #assuming gif
    img = load_img(filename)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (300, 300, 3)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 300, 300, 3)
    x.shape
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,save_to_dir='/home/aarodoeht/Desktop/cnnex/NewTrain/Covid/', save_format='jpeg'):
        i += 1
        if i > 4:
            break  # otherwise the generator would loop indefinitely


for filename in glob.glob('/home/aarodoeht/Desktop/cnnex/NewTrain/Other/*'): #assuming gif
    img = load_img(filename)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (300, 300, 3)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 300, 300, 3)
    x.shape
    # the .flow() command below generates batches of randomly transformed images
    # and saves the results to the `preview/` directory
    i = 0
    for batch in datagen.flow(x, batch_size=1,save_to_dir='/home/aarodoeht/Desktop/cnnex/NewTrain/Other/', save_format='jpeg'):
        i += 1
        if i > 3:
            break  # otherwise the generator would loop indefinitely