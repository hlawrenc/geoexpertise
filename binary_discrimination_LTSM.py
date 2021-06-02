import tensorflow as tf
########################################################################
## Import required packages
########################################################################
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.model_selection import train_test_split# deep learning libraries for text pre-processing
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences# Modeling 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalAveragePooling1D, MaxPooling1D, Dense, Dropout, LSTM, Bidirectional, GRU
 
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

import csv
csv.field_size_limit(2147483647)

########################################################################
## Functions
########################################################################

def plot_graphs(var1, var2, string, model_type, metrics):
    plt.plot(metrics[[var1, var2]])
    plt.title('%s Training and Validation %s' % (model_type, string))
    plt.xlabel ('Number of epochs')
    plt.ylabel(string)
    plt.legend([var1, var2])
    plt.grid(True)

########################################################################
## Load and explore the spam data
########################################################################

filepath = 'data/'
filename = 'test_May10.txt'

colnames = ['userid', 'label', 'review']
reviews = pd.read_csv(f'{filepath}{filename}',
                 warn_bad_lines=False,
                 skiprows=1,
                 names=colnames,
                 quoting=3,
                 sep='|',
                 error_bad_lines=False,
                 encoding='latin1',
                 engine='python')

reviews = reviews.iloc[: , -2:]

##local_reviews = reviews[reviews.label==1]
##nonlocal_reviews = reviews[reviews.label==0]
##
##local_reviews_text = " ".join(local_reviews.review.to_numpy().tolist())
##nonlocal_reviews_text = " ".join(nonlocal_reviews.review.to_numpy().tolist())
##
### wordclouds for local and non-local reviews
##local_reviews_cloud = WordCloud(width=520, height=260, stopwords=STOPWORDS, max_font_size=50, background_color ="black", colormap='Blues').generate(local_reviews_text)
##plt.figure(figsize=(16,10))
##plt.imshow(local_reviews_cloud, interpolation='bilinear')
##plt.axis('off') 
####plt.show()
##
##nonlocal_reviews_cloud = WordCloud(width=520, height=260, stopwords=STOPWORDS, max_font_size=50, background_color ="black", colormap='Blues').generate(nonlocal_reviews_text)
##plt.figure(figsize=(16,10))
##plt.imshow(nonlocal_reviews_cloud, interpolation='bilinear')
##plt.axis('off') 
####plt.show()
##
### This will allow us to view data rowcount differences
##plt.figure(figsize=(8,6))
##sns.countplot(reviews.label)
##plt.show()

# Oversampling
##print(Counter(nonlocal_reviews))
##oversample = RandomOverSampler(sampling_strategy=0.5)
##local_under, nonlocal_under = oversample.fit_resample(local_reviews.label, nonlocal_reviews.label)
##print(Counter(nonlocal_under))
##undersample = RandomUnderSampler(sampling_strategy='minority')
##local_over, nonlocal_over = oversample.fit_resample(local_under, nonlocal_under)
##print(Counter(nonlocal_over))

# This will allow us to view data rowcount differences
##plt.figure(figsize=(8,6))
##sampled_reviews = local_over.append(nonlocal_over).reset_index(drop=True)
##sns.countplot(reviews.label)
##plt.show()

### downsample the local reviews as they vastly outnumber the non-local ones
##local_reviews_df = local_reviews.sample(n = len(nonlocal_reviews), random_state = 23)
##nonlocal_reviews_df = nonlocal_reviews
##
##reviews_df = local_reviews_df.append(nonlocal_reviews_df).reset_index(drop=True)
##plt.figure(figsize=(8,6))
##sns.countplot(reviews_df.label)
##plt.title('Distribution of local and non-local review (after downsampling)')
##plt.xlabel('Local/Non-local')
##plt.show()
##
### Get length column for each text
##reviews_df['text_length'] = reviews_df['message'].apply(len)
###Calculate average length by label types
##labels = reviews_df.groupby('label').mean()

########################################################################
## Prepare train/test data and pre-process text
########################################################################

##train_msg, test_msg, train_labels, test_labels 
X_train, X_test, y_train, y_test = train_test_split(reviews.review, reviews.label, test_size=0.2, random_state=2)

# Defining pre-processing hyperparameters
max_len = 200 #should be same as glove
trunc_type = "post" 
padding_type = "post" 
oov_tok = "<OOV>" 
vocab_size = 5000

tokenizer = Tokenizer(num_words = vocab_size, char_level=False, oov_token = oov_tok)
tokenizer.fit_on_texts(X_train)

word_index = tokenizer.word_index
# check how many words 
tot_words = len(word_index)
print('There are %s unique tokens in training data. ' % tot_words)
        
########################################################################
## Sequencing, Padding, and Embedding
########################################################################

# Sequencing and padding on training and testing 
training_sequences = tokenizer.texts_to_sequences(X_train)
training_padded = pad_sequences (training_sequences, maxlen = max_len, padding = padding_type, truncating = trunc_type )
testing_sequences = tokenizer.texts_to_sequences(X_test)
testing_padded = pad_sequences(testing_sequences, maxlen = max_len, padding = padding_type, truncating = trunc_type)

# Shape of train tensor
print('Shape of training tensor: ', training_padded.shape)
print('Shape of testing tensor: ', testing_padded.shape)

filepath = 'data/'
filename = f'glove.6B.{max_len}d.txt'
print(filename)

embeddings_index = {}
with open(f'{filepath}{filename}', 'r', encoding='UTF-8') as f:
    words = set()
    word_to_vec_map = {}
    for line in f:
      w_line = line.split()
      curr_word = w_line[0]
      embeddings_index[curr_word] = np.array(w_line[1:], dtype=np.float64)
f.close()

print('Found %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((len(word_index) + 1, max_len))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_dim = 16 
drop_value = 0.2 
num_epochs = 50
patience = 10

embedding_layer = Embedding(len(word_index) + 1,
                            max_len,
                            weights=[embedding_matrix],
                            input_length=max_len,
                            trainable=False)
print(embedding_matrix)

########################################################################
## Dense Spam Detection Model
########################################################################

n_dense = 24

#Dense model architecture
model = Sequential()
model.add(embedding_layer)
model.add(GlobalAveragePooling1D())
model.add(Dense(n_dense, activation='relu'))
model.add(Dropout(drop_value))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam' ,metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=patience)
history = model.fit(training_padded, y_train, epochs=num_epochs, validation_data=(testing_padded, y_test),callbacks =[early_stop], verbose=2)
##history = model.fit(training_padded, y_train, epochs=num_epochs, validation_data=(testing_padded, y_test), verbose=2)
model.evaluate(testing_padded, y_test)

metrics = pd.DataFrame(history.history)
metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy', 'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
print(metrics)

fig = plt.figure(1)

plt.subplot(421)
plot_graphs('Training_Loss', 'Validation_Loss', 'loss', 'Dense', metrics)
plt.subplot(422)
plot_graphs('Training_Accuracy', 'Validation_Accuracy', 'accuracy', 'Dense', metrics)

########################################################################
## Long Short Term Memory (LSTM) Model
########################################################################

#LSTM hyperparameters
n_lstm = 32

#LSTM Spam detection architecture
model1 = Sequential()
model1.add(embedding_layer)
model1.add(Conv1D(64, 5, activation='relu'))
model1.add(MaxPooling1D(pool_size=4))
model1.add(LSTM(n_lstm, dropout=drop_value, return_sequences=True))
model1.add(LSTM(n_lstm, dropout=drop_value, return_sequences=True))
model1.add(Dense(1, activation='sigmoid'))
model1.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=patience)
history = model1.fit(training_padded, y_train, epochs=num_epochs, validation_data=(testing_padded, y_test),callbacks =[early_stop], verbose=2)
##history = model1.fit(training_padded, y_train, epochs=num_epochs, validation_data=(testing_padded, y_test), verbose=2)

# Create a dataframe
metrics = pd.DataFrame(history.history)# Rename column
metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy', 'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
print(metrics)

plt.subplot(423)
plot_graphs('Training_Loss', 'Validation_Loss', 'loss', 'LSTM', metrics)
plt.subplot(424)
plot_graphs('Training_Accuracy', 'Validation_Accuracy', 'accuracy', 'LSTM', metrics)

########################################################################
## Bi-directional Long Short Term Memory (BiLSTM) Model
########################################################################

model2 = Sequential()
model2.add(embedding_layer)
model2.add(Conv1D(64, 5, activation='relu'))
model2.add(MaxPooling1D(pool_size=4))
model2.add(Bidirectional(LSTM(n_lstm, dropout=drop_value, return_sequences=True)))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.summary()

# Training
early_stop = EarlyStopping(monitor='val_loss', patience=patience)
history = model2.fit(training_padded, y_train, epochs=num_epochs, validation_data=(testing_padded, y_test),callbacks =[early_stop], verbose=2)
##history = model2.fit(training_padded, y_train, epochs=num_epochs, validation_data=(testing_padded, y_test), verbose=2)

# Create a dataframe
metrics = pd.DataFrame(history.history)# Rename column
metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy', 'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
print(metrics)

plt.subplot(425)
plot_graphs('Training_Loss', 'Validation_Loss', 'loss', 'Bi-directional', metrics)
plt.subplot(426)
plot_graphs('Training_Accuracy', 'Validation_Accuracy', 'accuracy', 'Bi-directional', metrics)

########################################################################
## GRU Model
########################################################################

#GRU hyperparameters
n_gru = 32

#GRU Spam detection architecture
model3 = Sequential()
model3.add(embedding_layer)
model3.add(Conv1D(64, 5, activation='relu'))
model3.add(MaxPooling1D(pool_size=4))
model3.add(GRU(n_gru, dropout=drop_value, return_sequences=True))
model3.add(GRU(n_gru, dropout=drop_value, return_sequences=True))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=patience)
history = model3.fit(training_padded, y_train, epochs=num_epochs, validation_data=(testing_padded, y_test),callbacks =[early_stop], verbose=2)
##history = model3.fit(training_padded, y_train, epochs=num_epochs, validation_data=(testing_padded, y_test), verbose=2)

# Create a dataframe
metrics = pd.DataFrame(history.history)# Rename column
metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy', 'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
print(metrics)

plt.subplot(427)
plot_graphs('Training_Loss', 'Validation_Loss', 'loss', 'GRU', metrics)
plt.subplot(428)
plot_graphs('Training_Accuracy', 'Validation_Accuracy', 'accuracy', 'GRU', metrics)

fig.subplots_adjust(hspace=1)
plt.show()

########################################################################
## Comparison
########################################################################












