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
##%matplotlib inline# library for train test split
from sklearn.model_selection import train_test_split# deep learning libraries for text pre-processing
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences# Modeling 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM, Bidirectional, GRU

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

colnames = ['userid', 'label', 'message']
messages = pd.read_csv(f'{filepath}{filename}',
                 warn_bad_lines=False,
                 skiprows=1,
                 names=colnames,
                 quoting=3,
                 sep='|',
                 error_bad_lines=False,
                 encoding='latin1',
                 engine='python')

messages = messages.iloc[: , -2:]
##url = 'https://raw.githubusercontent.com/ShresthaSudip/SMS_Spam_Detection_DNN_LSTM_BiLSTM/master/SMSSpamCollection'
##messages = pd.read_csv(url, sep ='\t',names=["label", "message"])

# Get all the ham and spam emails
ham_msg = messages[messages.label ==1]
spam_msg = messages[messages.label==0]# Create numpy list to visualize using wordcloud
ham_msg_text = " ".join(ham_msg.message.to_numpy().tolist())
spam_msg_text = " ".join(spam_msg.message.to_numpy().tolist())

# wordcloud of ham messages
ham_msg_cloud = WordCloud(width =520, height =260, stopwords=STOPWORDS,max_font_size=50, background_color ="black", colormap='Blues').generate(ham_msg_text)
plt.figure(figsize=(16,10))
plt.imshow(ham_msg_cloud, interpolation='bilinear')
plt.axis('off') # turn off axis
plt.show()

# wordcloud of spam messages
spam_msg_cloud = WordCloud(width =520, height =260, stopwords=STOPWORDS,max_font_size=50, background_color ="black", colormap='Blues').generate(spam_msg_text)
plt.figure(figsize=(16,10))
plt.imshow(spam_msg_cloud, interpolation='bilinear')
plt.axis('off') # turn off axis
plt.show()

# we can observe imbalance data here 
plt.figure(figsize=(8,6))
sns.countplot(messages.label)
# Percentage of spam messages
(len(spam_msg)/len(ham_msg))*100 # 15.48%
plt.show()

# one way to fix it is to downsample the ham msg
ham_msg_df = ham_msg.sample(n = len(spam_msg), random_state = 44)
spam_msg_df = spam_msg
print(ham_msg_df.shape, spam_msg_df.shape)

# Create a dataframe with these ham and spam msg
msg_df = ham_msg_df.append(spam_msg_df).reset_index(drop=True)
plt.figure(figsize=(8,6))
sns.countplot(msg_df.label)
plt.title('Distribution of ham and spam email messages (after downsampling)')
plt.xlabel('Message types')
plt.show()

# Get length column for each text
msg_df['text_length'] = msg_df['message'].apply(len)#Calculate average length by label types
labels = msg_df.groupby('label').mean()

##sys.exit()

########################################################################
## Prepare train/test data and pre-process text
########################################################################

# Map ham label as 0 and spam as 1
msg_df['msg_type']= msg_df['label'].map({0: 0, 1: 1})
msg_label = msg_df['msg_type'].values# Split data into train and test
train_msg, test_msg, train_labels, test_labels = train_test_split(msg_df['message'], msg_label, test_size=0.2, random_state=434)

# Defining pre-processing hyperparameters
max_len = 50 
trunc_type = "post" 
padding_type = "post" 
oov_tok = "<OOV>" 
vocab_size = 500

tokenizer = Tokenizer(num_words = vocab_size, char_level=False, oov_token = oov_tok)
tokenizer.fit_on_texts(train_msg)

word_index = tokenizer.word_index
# check how many words 
tot_words = len(word_index)
print('There are %s unique tokens in training data. ' % tot_words)

########################################################################
## Sequencing and Padding
########################################################################

# Sequencing and padding on training and testing 
training_sequences = tokenizer.texts_to_sequences(train_msg)
training_padded = pad_sequences (training_sequences, maxlen = max_len, padding = padding_type, truncating = trunc_type )
testing_sequences = tokenizer.texts_to_sequences(test_msg)
testing_padded = pad_sequences(testing_sequences, maxlen = max_len,
padding = padding_type, truncating = trunc_type)

# Shape of train tensor
print('Shape of training tensor: ', training_padded.shape)
print('Shape of testing tensor: ', testing_padded.shape)

########################################################################
## Dense Spam Detection Model
########################################################################

vocab_size = 500 # As defined earlier
embeding_dim = 16
drop_value = 0.2 # dropout
n_dense = 24

#Dense model architecture
model = Sequential()
model.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(24, activation='relu'))
model.add(Dropout(drop_value))
model.add(Dense(1, activation='sigmoid'))

model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam' ,metrics=['accuracy'])
# fitting a dense spam detector model
num_epochs = 30
early_stop = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(training_padded, train_labels, epochs=num_epochs, validation_data=(testing_padded, test_labels),callbacks =[early_stop], verbose=2)
# Model performance on test data 
model.evaluate(testing_padded, test_labels)

# Read as a dataframe 
metrics = pd.DataFrame(history.history)
# Rename column
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
n_lstm = 20
drop_lstm =0.2

#LSTM Spam detection architecture
model1 = Sequential()
model1.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
model1.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
model1.add(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True))
model1.add(Dense(1, activation='sigmoid'))

model1.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

num_epochs = 30
early_stop = EarlyStopping(monitor='val_loss', patience=2)
history = model1.fit(training_padded, train_labels, epochs=num_epochs, validation_data=(testing_padded, test_labels),callbacks =[early_stop], verbose=2)

# Create a dataframe
metrics = pd.DataFrame(history.history)# Rename column
metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy',
                         'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
print(metrics)

plt.subplot(423)
plot_graphs('Training_Loss', 'Validation_Loss', 'loss', 'LSTM', metrics)
plt.subplot(424)
plot_graphs('Training_Accuracy', 'Validation_Accuracy', 'accuracy', 'LSTM', metrics)

########################################################################
## Bi-directional Long Short Term Memory (BiLSTM) Model
########################################################################

model2 = Sequential()
model2.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
model2.add(Bidirectional(LSTM(n_lstm, dropout=drop_lstm, return_sequences=True)))
model2.add(Dense(1, activation='sigmoid'))

model2.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

# Training
num_epochs = 30
early_stop = EarlyStopping(monitor='val_loss', patience=2)
history = model2.fit(training_padded, train_labels, epochs=num_epochs, 
                    validation_data=(testing_padded, test_labels),callbacks =[early_stop], verbose=2)

# Create a dataframe
metrics = pd.DataFrame(history.history)# Rename column
metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy',
                         'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
print(metrics)

plt.subplot(425)
plot_graphs('Training_Loss', 'Validation_Loss', 'loss', 'Bi-directional', metrics)
plt.subplot(426)
plot_graphs('Training_Accuracy', 'Validation_Accuracy', 'accuracy', 'Bi-directional', metrics)


########################################################################
## GRU Model
########################################################################

#GRU hyperparameters
n_gru = 20
drop_gru =0.2

#GRU Spam detection architecture
model3 = Sequential()
model3.add(Embedding(vocab_size, embeding_dim, input_length=max_len))
model3.add(GRU(n_gru, dropout=drop_gru, return_sequences=True))
model3.add(GRU(n_gru, dropout=drop_gru, return_sequences=True))
model3.add(Dense(1, activation='sigmoid'))

model3.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['accuracy'])

num_epochs = 30
early_stop = EarlyStopping(monitor='val_loss', patience=2)
history = model3.fit(training_padded, train_labels, epochs=num_epochs, validation_data=(testing_padded, test_labels),callbacks =[early_stop], verbose=2)

# Create a dataframe
metrics = pd.DataFrame(history.history)# Rename column
metrics.rename(columns = {'loss': 'Training_Loss', 'accuracy': 'Training_Accuracy',
                         'val_loss': 'Validation_Loss', 'val_accuracy': 'Validation_Accuracy'}, inplace = True)
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












