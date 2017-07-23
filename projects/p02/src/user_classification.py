import string
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import os
import json
from datetime import date
from fastnumbers import isfloat, isint
import zipfile
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda
from keras.layers import MaxPooling1D, LSTM, Conv1D, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.utils import to_categorical

# ohe функция
def ohe(x, sz):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

def create_vocab_set():
    alphabet = (list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + [' ', '\n'])
    vocab_size = len(alphabet)
    vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix+1
    return vocab, vocab_size

def text2sequence(text, vocab):
    temp = []
    for review in text:
        temp.append([])
        for i in review:
            char = vocab.get(i,0)
            if char != 0:
                temp[-1].append(char)
    return temp

dir_train = '../data'

mappings = {
    'career': 0,
    'theory_and_practice': 1,
    'deep_learning': 2,
    'lang_python': 3,
    '_meetings': 4,
    'kaggle_crackers': 5,
    'big_data': 6,
    'lang_r': 7,
    'hardware': 8,
    'nlp': 9,
    'welcome': 10,
    'datasets': 11,
    'bayesian': 12
}


# parameters initialization
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

# initialize dictionary size and maximum sentence length
MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 400

NAME = "simple ohe lstm"

data = pd.read_csv(os.path.join(dir_train, 'train_set.csv'), usecols=range(1,11), parse_dates=['timestamp', 'thread_timestamp'])
data = data[
    data.channel.isin(['career', 'big_data', 'deep_learning', 'hardware', 'kaggle_crackers',
           'lang_python',  'lang_r', 'nlp', 'theory_and_practice', 'welcome', 'bayesian', '_meetings', 'datasets']) &
    data.main_msg
]

# data_train = data.
date_before = date(2017, 4, 1)
train = data[data['timestamp'] < date_before]
val = data[data['timestamp'] > date_before]

train_data = train[['channel', 'text']].reset_index()[['channel', 'text']]
train_data['channel'] = train_data.channel.map(mappings)
train_data = train_data.sort_values('channel').reset_index()[['channel', 'text']]

val_data = val[['channel', 'text']].reset_index()[['channel', 'text']]
val_data['channel'] = val_data.channel.map(mappings)
val_data = val_data.sort_values('channel').reset_index()[['channel', 'text']]

train_data = train_data[~train_data.text.apply(lambda x: isfloat(x) or isint(x) or len(x) < 20)]
val_data = val_data[~val_data.text.apply(lambda x: isfloat(x) or isint(x) or len(x) < 20)]

train_text = train_data['text'].astype(str)
train_labels =  np.asarray(train_data['channel'], dtype='int8')

val_text = val_data['text'].astype(str)
val_labels = np.asarray(val_data['channel'], dtype='int8')

vocab, vocab_size = create_vocab_set()

X_train = text2sequence(train_text, vocab)
X_val = text2sequence(val_text, vocab)

X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, value=0)
X_val = pad_sequences(X_val, maxlen=MAX_SEQUENCE_LENGTH, value=0)

train_labels = to_categorical(train_labels, num_classes=13)
val_labels = to_categorical(val_labels, num_classes=13)

# callbacks initialization
# automatic generation of learning curves
callback_1 = TensorBoard(log_dir='./logs/logs_{}'.format(NAME), histogram_freq=0,
                             write_graph=False, write_images=False)
# stop training model if accuracy does not increase more than five epochs
callback_2 = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='auto')
# best model saving
callback_3 = ModelCheckpoint("../models/model_{}.hdf5".format(NAME), monitor='val_acc',
                                 save_best_only=True, verbose=1)

NAME = "char_cnn_ohe"
# инициализация входа
in_sentence = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
# Lambda слой для ohe преобразования
embedded = Lambda(ohe, output_shape=lambda x: (x[0], x[1], vocab_size), arguments={"sz": vocab_size})(in_sentence)
block = embedded
# свертки с MaxPooling
for i in range(3):
    block = Conv1D(activation="relu", filters=100, kernel_size=4, padding="valid")(block)
    if i == 0:
        block = MaxPooling1D(pool_size=5)(block)
# LSTM ячейка
block = LSTM(128, dropout=0.1, recurrent_dropout=0.1)(block)
block = Dense(100, activation='relu')(block)
block = Dense(13, activation='softmax')(block)
# собираем модель
model = Model(inputs=in_sentence, outputs=block)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
model.fit(X_train, train_labels, validation_data=[X_val, val_labels],
         batch_size=1024, epochs=100, callbacks=[callback_1, callback_2, callback_3])