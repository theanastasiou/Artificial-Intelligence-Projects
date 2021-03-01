import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = r'/home/at/Desktop/Texniti_2/Ergasia2/' 
dataset = path + "predictednew1_BIAS.csv"
dataset = pd.read_csv(dataset)  #FULLSET
print(len(dataset))

dataset['Bias'] = np.where((dataset['identity_attack']  == True) & (dataset['target']== True) & (dataset['Predicted'] > 0) 
                     ,True, False)
# mask = dataset[~(dataset['identity_attack'] == False).any(axis=0)]
# print(mask.value_counts())
# dataset['Predict_ok'] = np.where(),True,False)
# dataset['wrong_prediction'] =  np.where((dataset['Predicted'] < 0 )&(dataset['identity_attack'] == True),True,False)
dataset['Bias_1'] = np.where((dataset['identity_attack'] == True )& (dataset['target']== True),True,False)
# print(len(dataset['Bias']==True))

print(dataset.target.value_counts())
print('How many times identity_attack is true')
print(dataset.identity_attack.value_counts())
# print('How many times prediction is not correct')
# print(dataset.wrong_prediction.value_counts())
print("How Many times target = true and identity_attack = true")
print(dataset['Bias_1'].value_counts())
dataset.to_csv("checkbias.csv")

print("Times target & identity_attack are both true and the prediction is Correct")
print(dataset['Bias'].value_counts())