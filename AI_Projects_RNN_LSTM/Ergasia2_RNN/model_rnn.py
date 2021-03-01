import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plot_keras_history import plot_history
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, SpatialDropout1D
from keras.layers import TimeDistributed
from random import random
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras import preprocessing
from keras.preprocessing.sequence import pad_sequences
import re
import nltk
import string
import tensorflow as tf
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from keras.models import model_from_json
from numpy import cumsum
import scipy.stats as stats
from sklearn import metrics
from nltk.stem.snowball import SnowballStemmer

REPLACE_SLASH = re.compile('[/(){\}[\]\|@,;]')
BAD_SYMBOLS = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english')) 
NUMERIC = re.compile('123456789')
HTTP = re.compile('http\S+')
WWW = re.compile('www\S+')
H1 = re.compile('\b\w{1}\b')
AD_SPACES = re.compile(' +')

# padding gia na exun ola ta sequence - protasis - comment text to idio length
MAX_SEQUENCE_LENGTH = 250
def pad_text(texts, tokenizer):
    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)

def preprocessing_train(text):
    text = REPLACE_SLASH.sub(' ', text)  # remove SLASHES kai antikatastasi me ' '. 
    text = BAD_SYMBOLS.sub('', text) # remove BAD_SYMBOLS kai antikatastasi me ' '. 
    text = NUMERIC.sub('',text)
    text = HTTP.sub('',text)
    text = WWW.sub('',text)
    text =H1.sub('',text)
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) 
    # print(text) 
    return text

# plot the identities me ta toxic-non toxic 
def identity_counts(df, feature_columns, title):
  toxic = df[df[TOXICITY_COLUMN] >= .5][feature_columns]
  non_toxic = df[df[TOXICITY_COLUMN] < .5][feature_columns]
  toxic_count = toxic.where(toxic == 0, other = 1).sum()
  non_toxic_count = non_toxic.where(non_toxic == 0, other = 1).sum()
  toxic_vs_non_toxic = pd.concat([ toxic_count,non_toxic_count], axis=1).rename(index=str, columns={ 1: "non-toxic", 0: "toxic",})
  toxic_vs_non_toxic.sort_values(by='toxic').plot(kind='bar', stacked=True, figsize=(15,6), fontsize=8).legend(prop={'size': 10})
  plt.title(title, fontsize=15)
  plt.xlabel("Identity")
  plt.ylabel("comments count")
  plt.show()
  plt.close()

#gia to plot tu auc
def plot_auc_heatmap(bias_metrics_results, models):
  metrics_list = [SUBGROUP_AUC]
  #, BACKGROUND_POSITIVE_SUBGROUP_NEGATIVE_AUC, BACKGROUND_NEGATIVE_SUBGROUP_POSITIVE_AUC
  df = bias_metrics_results.set_index('subgroup')
  columns = []
  vlines = [i * len(models) for i in range(len(metrics_list))]
  for metric in metrics_list:
    for model in models:
      columns.append(metric)
  num_rows = len(df)
  num_columns = len(columns)
  fig = plt.figure(figsize=(num_columns, 0.5 * num_rows))
  ax = sns.heatmap(df[columns], annot=True, fmt='.2', cbar=True, cmap='Reds_r',
                   vmin=0.5, vmax=1.0)
  ax.xaxis.tick_top()
  plt.xticks(rotation=90)
  ax.vlines(vlines, *ax.get_ylim())
  return ax


# Seed for Pandas sampling, to get consistent sampling results
RANDOM_STATE = 123456789
# Print some records to compare our model results with the correct labels
TOXICITY_COLUMN = 'target'
TEXT_COLUMN = 'comment_text'

path = r'/home/at/Desktop/Texniti_2/Ergasia2/' # use your path'
train_dataset = path + "train.csv"
test_dataset = path + "test.csv"
train_dataset = pd.read_csv(train_dataset)  #FULLSET
train_dataset = train_dataset.reset_index(drop=True)

print(len(train_dataset))
#identity columns - xrisi argotera
identity_columns = [
    'insult','severe_toxicity','obscene','identity_attack','threat',
    'male', 'female', 'transgender', 'other_gender', 'heterosexual',
    'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation', 'christian',
    'jewish', 'muslim', 'hindu', 'buddhist', 'atheist', 'other_religion', 'black',
    'white', 'asian', 'latino', 'other_race_or_ethnicity',
    'physical_disability', 'intellectual_or_learning_disability',
    'psychiatric_or_mental_illness', 'other_disability','sexual_explicit']


#metatropi se TRUE-FALS olwn twn stilwn tou arxeiou 
def convert_to_bool(df, col_name):
  df[col_name] = np.where(df[col_name] >= 0.5, True, False)

def convert_dataframe_to_bool(df):
  bool_df = df.copy()
  for col in ['target'] + identity_columns:
      convert_to_bool(bool_df, col)
  return bool_df


#-------------------------------- DATA PREPROCESING --------------------------------------------------- 
#preprocess ta dedomena tis stilis "Comment_text" - lower - remove bad characters - replace numeric values 
train_dataset['comment_text'] = train_dataset['comment_text'].str.lower() # lowercase text
train_dataset['comment_text'] = train_dataset['comment_text'].apply(preprocessing_train)
train_dataset['comment_text'] = train_dataset['comment_text'].str.replace('\d+', '')
# train_dataset['comment_text'].to_csv("comment.csv") #metatrepo se csv arxeio

# The maximum number of words to be used 
MAX_NB_WORDS = 90000
# Max number of words in each complaint.
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 32 #fix 

# Tokenizer Initialiazation 
tokenizer = Tokenizer(num_words=MAX_NB_WORDS,split=" ",char_level=False)
# #Tokenize text documents (becomes a list of lists)
tokenizer.fit_on_texts(train_dataset['comment_text'].values)
# train_dataset.to_csv("preprocessed.csv") #metatrepo se csv arxeio


#spasimo arxeiou se train-test , to train gia train - to test gia ta bias
train_size = int(len(train_dataset) * 0.50)
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = train_dataset[0:train_size], train_dataset[train_size:len(train_dataset)]

#metatropi oloklirwn twn dataset se bool kai drop akira columns
train_dataset = convert_dataframe_to_bool(train_dataset)
train_dataset = train_dataset.drop(columns = ['created_date','publication_id','parent_id','article_id','rating','funny','wow','sad','likes','disagree','identity_annotator_count','toxicity_annotator_count'])
test_dataset = convert_dataframe_to_bool(test_dataset)
test_dataset = test_dataset.drop(columns = ['created_date','publication_id','parent_id','article_id','rating','funny','wow','sad','likes','disagree','identity_annotator_count','toxicity_annotator_count'])

# plotting toxicity of train - test sets 
identity_counts(train_dataset, identity_columns, 
                "Plot Showing Identity counts for Toxic and Non-Toxic comments for train data")
identity_counts(test_dataset, identity_columns,
                "Plot Showing Identity counts for Toxic and Non-Toxic comments for test data")


def print_count_and_percent_toxic(df, identity):
  # Query all training comments where the identity column equals True.
  identity_comments = train_dataset.query(identity + ' == True')
  # Query which of those comments also have "toxicity" equals True
  toxic_identity_comments = identity_comments.query('target == True')
  # Print the results.
  num_comments = len(identity_comments)
  if(num_comments!=0):
    percent_toxic = len(toxic_identity_comments) / num_comments 
    print('%d comments refer to the %s identity, %.2f%% are toxic' % (
      num_comments,
      identity,
      # multiply percent_toxic by 100 for easier reading.
      100 * percent_toxic))
  if(num_comments==0):
      print('%d comments refer to the %s identity' % (
      num_comments,
      identity))

#% toxicity for entire train_dataset
all_toxic_df = train_dataset.query('target == True')
print('%.2f%% of all comments are toxic' %
  (100 * len(all_toxic_df) / len(train_dataset)))

for nm in identity_columns:
  # posa % toxic comments iparxun ana identity
  print_count_and_percent_toxic(train_dataset,nm)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index)) #how many unique words found after tokenizer
X = tokenizer.texts_to_sequences(train_dataset['comment_text'].values)# Padding the sequences - for equal comment length.
# train_dataset.to_csv("seq.csv") #metatrepo se csv arxeio

# print(np.array(X[0]))
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH) #gia na exun to idio length ta sequences mas
print('Shape of data tensor:', (np.array(X)).shape)
Y = pd.get_dummies(train_dataset['target']) #dummies
print('Shape of label tensor:',(np.array(Y)).shape)

#split train - validation
X_train, X_test, Y_train, Y_test = train_test_split(np.array(X),np.array(Y), test_size = 0.10, random_state = 33)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(MAX_NB_WORDS,100,input_length=X.shape[1]),
    tf.keras.layers.SpatialDropout1D(0.3),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100,dropout=0.2)),
    tf.keras.layers.Dense(2)
])

#model compile using RMSProp - learning rate 0.01 - 0.001
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.RMSprop(
    learning_rate=0.001),
              metrics=['accuracy'])
print(X_train.shape,Y_train.shape)

#model fit
history = model.fit(X_train,Y_train, epochs=10, batch_size= 100, validation_data=(X_test, Y_test),  verbose=1).history

#----------------------- evaluate model --------------------------- 
loss, accuracy = model.evaluate(X_train, Y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, Y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
plot_history(history)
plt.show()


#---------------save model for later use --------------- 
model.save("model_Rnn.h5") 


#--------------------------- Use the model to score the test set. --------------------------------- 
test_comments_padded = pad_text(test_dataset["comment_text"], tokenizer) #tokenizer
MODEL_NAME = 'Predicted' #column predicted
test_dataset[MODEL_NAME] = model.predict(test_comments_padded)[:, 1] 
test_dataset.to_csv("predictednew1.csv") #write in csv file 


#------------------------- RANDOM COMMENTS PREDICTION -------------------------------------------------
comments= ['What are the leftist snowflakes in Canada doing to stop fat little Kim? Nothing.',
'I really hope like heck, there isnt a fucking Pokemon involved.',
'What a stupid fucking commenting system.','Is one of them gay?',
'Mean, spiteful, liars, power abusers, willing to pervert justice for political power---all those traits Nixon is accused of having are now resident in the Democrat Party and leftwing journalists. Carl, YOU are Nixon. YOU are despicable.',
'Its a black mark on the previous administrations who allowed massive illegal immigration to go on for far too long. Put the blame where it should be.',
'There are no words for these senseless acts.. God help us all if we do not learn that evil lurks even in the face of these teenagers. Such violence against their peers, should make everyone take pause. Be safe children, be safe. God says you can be the light of the world.']
dftest = pd.DataFrame(dict(comment_text=comments)) #metatropi listas se Dataframe
newLi = dftest['comment_text'].str.lower() #preprocess data gia na ginei akrivos to keimeno opws auto pou egine to training 
newLi  = newLi.apply(preprocessing_train)
newLi= newLi.str.replace('\d+', '')
newLi = pad_text(newLi, tokenizer)
dftest['predict'] = model.predict(newLi)[:, 1] #predict
dftest.to_csv('randompredictions.csv')


# ------------------------------------ BIAS - AUC -HEATMAP ----------------------------------------------

# print('training data columns are: %s' % test_dataset.columns)
# print(test_dataset.target.isna().sum())
# print(test_dataset.comment_text.isna().sum())

def calculate_overall_auc(df, model_name):
    true_labels = df[TOXICITY_COLUMN]
    predicted_labels = df[model_name]
    return metrics.roc_auc_score(true_labels, predicted_labels)

print( " Overall_AUC: ")
print(calculate_overall_auc(test_dataset, MODEL_NAME)) 


#ftiaxnei mia lista me stiles oi opoies exun toul 5 true-toxic records 
#gia na meiothei o thorivos - se stiles pou den exun katholu true kai den aksizei na 
#tis xrisimopoioisume gia ton ipologismo ton metrikwn
identities_with_over_5_records = []
for identity in identity_columns:
    num_records = len(test_dataset.query(identity + '==True'))
    if num_records >= 5:
        identities_with_over_5_records.append(identity)

SUBGROUP_AUC = 'subgroup_auc'
BACKGROUND_POSITIVE_SUBGROUP_NEGATIVE_AUC = 'positive_subgroup_negative_auc'
BACKGROUND_NEGATIVE_SUBGROUP_POSITIVE_AUC = 'negative_subgroup_positive_auc'

#upologizei to auc me vasi ta true k ta predict tu test 
def compute_auc(y_true, y_pred):
  try:
    return metrics.roc_auc_score(y_true, y_pred)
  except ValueError:
    return np.nan

#upologizei to auc tou kathe subgroup
def compute_subgroup_auc(df, subgroup, label, model_name):
  #print(label)
  subgroup_examples = df[df[subgroup]]
  return compute_auc(subgroup_examples[label], subgroup_examples[model_name])


# def compute_background_positive_subgroup_negative_auc(df, subgroup, label, model_name):
# #   """Computes the AUC of the within-subgroup negative examples and the background positive examples."""
#   # print(label)
#   # print(subgroup)
#   subgroup_negative_examples = df[df[subgroup] & ~df[label]]
#   non_subgroup_positive_examples = df[~df[subgroup] & df[label]]
#   examples = subgroup_negative_examples.append(non_subgroup_positive_examples)
#   return compute_auc(examples[label], examples[model_name])


# def compute_background_negative_subgroup_positive_auc(df, subgroup, label, model_name):
# #   """Computes the AUC of the within-subgroup positive examples and the background negative examples."""
#   subgroup_positive_examples = df[df[subgroup] & df[label]]
#   non_subgroup_negative_examples = df[~df[subgroup] & ~df[label]]
#   examples = subgroup_positive_examples.append(non_subgroup_negative_examples)
#   return compute_auc(examples[label], examples[model_name])


def compute_bias_metrics_for_model(dataset,subgroups, model, label_col,include_asegs=False):
#   """Computes per-subgroup metrics for all subgroups and one model."""
  records = []
  for subgroup in subgroups: 
    record = {
        'subgroup': subgroup,
        'subgroup_size': len(dataset[dataset[subgroup]])
    }
    record[SUBGROUP_AUC] = compute_subgroup_auc( dataset, subgroup, label_col, model)
    # record[BACKGROUND_POSITIVE_SUBGROUP_NEGATIVE_AUC] = compute_background_positive_subgroup_negative_auc(
    #     dataset, subgroup, label_col, model)
    # record[BACKGROUND_NEGATIVE_SUBGROUP_POSITIVE_AUC] = compute_background_negative_subgroup_positive_auc(
    #     dataset, subgroup, label_col, model)
    records.append(record)
  return pd.DataFrame(records).sort_values('subgroup_auc', ascending=True)


#Xriximopoiei to test_dataset kai tin stili predicted wste na dei poso epireazetai i apofasi analoga me to kathe identity
bias_metrics_df = compute_bias_metrics_for_model(test_dataset, identities_with_over_5_records, MODEL_NAME, TOXICITY_COLUMN)

ax = plot_auc_heatmap(bias_metrics_df, [MODEL_NAME])
plt.show()