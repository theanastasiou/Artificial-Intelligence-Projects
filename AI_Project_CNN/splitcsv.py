import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import ticker
import seaborn as sns
import glob, shutil

frame = pd.read_csv('metadata.csv') #load csv file

trainV1,test = np.split(frame, [int(.95*len(frame))])
train,valid = np.split(trainV1, [int(.85*len(trainV1))])

trainCovid = train.loc[(train['finding'] == 'COVID-19') & (train['folder'] != 'volumes') ] #frame with patients with covid-19 and not volumes
trainOther = train.loc[(train['finding'] != 'COVID-19' )& (train['folder'] != 'volumes')] #frame with patients with other and not volumes
print(len(train))
print(len(valid))
print(len(test))

validCovid = valid.loc[(valid['finding'] == 'COVID-19') & (valid['folder'] != 'volumes') ] #frame with patients with covid-19 and not volumes
validOther = valid.loc[(valid['finding'] != 'COVID-19' )& (valid['folder'] != 'volumes')] #frame with patients with other and not volumes

testCovid = test.loc[(test['finding'] == 'COVID-19') & (test['folder'] != 'volumes') ] #frame with patients with covid-19 and not volumes
testOther = test.loc[(test['finding'] != 'COVID-19' )& (test['folder'] != 'volumes')] #frame with patients with other and not volumes
print('train Covid: ')
print(len(trainCovid))
print('train Other')
print(len(trainOther))
print('test Covid:')
print(len(testCovid))
print('test Other:')
print(len(testOther))

#print(frameNotCovid)

source = '/home/aarodoeht/Desktop/cnnex'
imagesfolder = source+"/"+frame['folder'][0]+"/"
trainimagesCovid = trainCovid['filename'].tolist() #filenames twn eikonwn me covid
trainimagesOther = trainOther['filename'].tolist() #filenames twn eikonwn me other 
testimagesCovid = testCovid['filename'].tolist() #filenames twn eikonwn me covid
testimagesOther = testOther['filename'].tolist() #filenames twn eikonwn me other 
validimagesCovid = validCovid['filename'].tolist() #filenames twn eikonwn me covid
validimagesOther = validOther['filename'].tolist() #filenames twn eikonwn me othe


#metaferei tis eikones ston fakelo train apo tin list TrainC
for file in trainimagesCovid: 
    shutil.move(imagesfolder+file, source+"/NewTrain/Covid")
for file in trainimagesOther : 
    shutil.move(imagesfolder+file, source+"/NewTrain/Other")

for file in testimagesCovid: 
    shutil.move(imagesfolder+file, source+"/NewTest/Covid")
for file in testimagesOther : 
    shutil.move(imagesfolder+file, source+"/NewTest/Other")

for file in validimagesCovid: 
    shutil.move(imagesfolder+file, source+"/NewValid/Covid")
for file in validimagesOther : 
    shutil.move(imagesfolder+file, source+"/NewValid/Other")

# train_covid,test_covid = np.split(frameCovid, [int(.8*len(frameCovid))])
# train_other,test_other = np.split(frameNotCovid, [int(.8*len(frameNotCovid))])

# TrainC = train_covid['filename'].tolist()
# TrainO = train_other['filename'].tolist()

# TestC = test_covid['filename'].tolist()
# TestO = test_other['filename'].tolist()

#metaferei tis eikones ston fakelo train apo tin list TrainC
# for file in TrainC: 
#     shutil.move(imagesfolder+file, source+"/Train/Covid")
# for file in TrainO : 
#     shutil.move(imagesfolder+file, source+"/Train/Other")

#metaferei tis eikones ston fakelo test
# for file in TestC: 
#     shutil.move(imagesfolder+file, source+"/Test/Covid")
# for file in TestO : 
#     shutil.move(imagesfolder+file, source+"/Test/Other")

#metaferei tis eikones (covid-other) stous fakelus antistoixa
# for file in imagesCovid: 
#     shutil.move(imagesfolder+file, source+"/Covid")
# for file in imagesOther: 
#     shutil.move(imagesfolder+file, source+"/Other")
        