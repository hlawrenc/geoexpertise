import tqdm
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Dropout, Dense, Conv1D, BatchNormalization, MaxPooling1D, GlobalMaxPool1D, GRU, Flatten
from keras.models import Sequential, Model
import keras_metrics
import tensorflow as tf
from tensorflow.keras.models import load_model

from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 
from keras.models import load_model
from keras.callbacks import History 

import matplotlib.pyplot as plt
import seaborn as sns
import nltk, sys, csv
csv.field_size_limit(2147483647)

SEQUENCE_LENGTH = 100 # the length of all sequences (number of words per sample)
EMBEDDING_SIZE = 100  # Using 100-Dimensional GloVe embedding vectors
TEST_SIZE = 0.25 # ratio of testing set

##***************************************************
## Data input stage
##***************************************************
print('Starting file read...')

filepath = '../data/'
##filename = 'full_balanced.txt'
##filename = 'downtown_core.txt'
filename = 'downtown_core_balanced.txt'

colnames = ['userid', 'isLocal', 'text']
data = pd.read_csv(f'{filepath}{filename}',
                 warn_bad_lines=False,
                 skiprows=1,
                 names=colnames,
                 quoting=3,
                 sep=',',
                 error_bad_lines=False,
                 encoding='latin1',
                 engine='python')

X, y = data['text'], data['isLocal']

##***************************************************
## Functions
##***************************************************

def get_embedding_vectors(tokenizer):
    embedding_index = {}
    glovepath = '../data/glove.6B/'
    glovefilename = f'glove.6B.{EMBEDDING_SIZE}d.txt'
    with open(f'{glovepath}{glovefilename}', encoding='utf8') as f:
        for line in tqdm.tqdm(f, 'Reading GloVe'):
            values = line.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors

    word_index = tokenizer.word_index
    # we do +1 because Tokenizer() starts from 1
    embedding_matrix = np.zeros((len(word_index)+1, EMBEDDING_SIZE))
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            # words not found will be 0s
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix


def localness(X_train, y_train, X_test, y_test, tokenizer, params):

    history = History()
    # get the GloVe embedding vectors
    embedding_matrix = get_embedding_vectors(tokenizer)

    inp=Input(shape=(SEQUENCE_LENGTH, ),dtype='int32')

    embedding_layer = (Embedding(len(tokenizer.word_index)+1,
              EMBEDDING_SIZE,
              weights=[embedding_matrix],
              trainable=False,
              input_length=SEQUENCE_LENGTH))
    embedded_sequences = embedding_layer(inp)
    x = LSTM(params['output_count_lstm'], return_sequences=True,name='lstm_layer')(embedded_sequences)
    x = Conv1D(filters=params['filters'], kernel_size=params['kernel_size'], padding='same', activation='relu', kernel_initializer='he_uniform')(x)
    x = MaxPooling1D(params['pool_size'])(x)
    x = GlobalMaxPool1D()(x)
    x = BatchNormalization()(x)
    x = Dense(params['output_1_count_dense'], activation=params['activation'], kernel_initializer='he_uniform')(x)
    x = Dropout(params['dropout'])(x)
    x = Dense(params['output_2_count_dense'], activation=params['activation'], kernel_initializer='he_uniform')(x)
    x = Dropout(params['dropout'])(x)
    preds = Dense(2, activation=params['last_activation'], kernel_initializer='glorot_uniform')(x)
    model = Model(inputs=inp, outputs=preds)
    model.compile(loss=params['loss'], optimizer=params['optimizer'], metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])

    model_info = model.fit(X_train, y_train, validation_data=(X_test, y_test),
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              verbose=params['verbose'],
              callbacks=[history])
    model.save('model1.h5')
    return model, model_info

##***************************************************
## Start
##***************************************************

X_train_first, X_test_first, y_train_first, y_test_first = train_test_split(X, y, test_size=TEST_SIZE, random_state=7)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train_first)
tokenizer.fit_on_texts(X_test_first)

X_train = tokenizer.texts_to_sequences(X_train_first)
X_test = tokenizer.texts_to_sequences(X_test_first)
y_train = np.array(y_train_first, dtype=object)
y_test = np.array(y_test_first, dtype=object)

X_train = np.array(X_train, dtype=object)
X_test = np.array(X_test, dtype=object)
X_train = pad_sequences(X_train, maxlen=SEQUENCE_LENGTH, padding='post')
X_test = pad_sequences(X_test, maxlen=SEQUENCE_LENGTH, padding='post')
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

p={
    'output_count_lstm': 50,
    'output_1_count_dense': 64,
    'output_2_count_dense': 2,
    'filters' : 64,
    'kernel_size' : 3,
    'batch_size': 32,
    'pool_size': 3,
    'epochs': 50,
    'optimizer': 'adam',
    'activation': 'relu',
    'last_activation': 'sigmoid',
    'dropout': 0.1,
    'loss': 'binary_crossentropy',
    'verbose': 2
}
    
##model, model_info = localness(X_train, y_train, X_test, y_test, tokenizer, p)
### summarize history for accuracy
##plt.plot(model_info.history['accuracy'])
##plt.plot(model_info.history['val_accuracy'])
##plt.title('model accuracy')
##plt.ylabel('accuracy')
##plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
##plt.show()
### summarize history for loss
##plt.plot(model_info.history['loss'])
##plt.plot(model_info.history['val_loss'])
##plt.title('model loss')
##plt.ylabel('loss')
##plt.xlabel('epoch')
##plt.legend(['train', 'test'], loc='upper left')
##plt.show()

model = tf.keras.models.load_model('model1.h5', compile=False)
model.compile(loss=p['loss'], optimizer=p['optimizer'], metrics=['accuracy', keras_metrics.precision(), keras_metrics.recall()])
##result = model.evaluate(X_test, y_test)
##loss = result[0]
##accuracy = result[1]
##precision = result[2]
##recall = result[3]
##print(f'[+] Loss:     {loss*100:.2f}%')
##print(f'[+] Accuracy: {accuracy*100:.2f}%')
##print(f'[+] Precision:{precision*100:.2f}%')
##print(f'[+] Recall:   {recall*100:.2f}%')
##
##model.summary()

preds = model.predict(X_test)

##cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(preds, axis=1))
##f = sns.heatmap(cm, annot=True, fmt='g')
##plt.show()

with open('../data/predictions.csv', 'a', encoding='utf-8') as out_file:
    for i in range(len(preds)):
        output = '%s, %s, %s \n' % (X_test_first.values[i], y_test_first.values[i], preds[i][1])
        out_file.write(output)
