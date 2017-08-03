import string
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import os
import re
from datetime import date
from fastnumbers import isfloat, isint
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D, Dense, Dropout, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.utils import to_categorical


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

rus_alphabet = ['а','б','в','г','д','е','ё','ж','з','и','й','к','л','м','н','о','п','р','с','т','у','ф','х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я']

def create_vocab_set():

    alphabet = (rus_alphabet + list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + [' ', '\n'])
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
    'nlp': 8,
    'welcome': 9,
    'datasets': 10,
    'bayesian': 11
}


# parameters initialization
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

# initialize dictionary size and maximum sentence length
MAX_SEQUENCE_LENGTH = 150

data = pd.read_csv(os.path.join(dir_train, 'train_set.csv'), usecols=range(1,11), parse_dates=['timestamp', 'thread_timestamp'])
data = data[
    data.channel.isin(['career', 'big_data', 'deep_learning', 'kaggle_crackers',
           'lang_python',  'lang_r', 'nlp', 'theory_and_practice', 'welcome', 'bayesian', '_meetings', 'datasets']) &
    data.main_msg
]

# data_train = data.
date_before = date(2017, 4, 1)
train = data[data['timestamp'] <= date_before]
val = data[data['timestamp'] > date_before]

train_data = train[['channel', 'text']].reset_index()[['channel', 'text']]
train_data['channel'] = train_data.channel.map(mappings)
train_data = train_data.sort_values('channel').reset_index()[['channel', 'text']]

val_data = val[['channel', 'text']].reset_index()[['channel', 'text']]
val_data['channel'] = val_data.channel.map(mappings)
val_data = val_data.sort_values('channel').reset_index()[['channel', 'text']]

train_data = train_data[~train_data.text.apply(lambda x: isfloat(x) or isint(x) or len(x) < 20)]
val_data = val_data[~val_data.text.apply(lambda x: isfloat(x) or isint(x) or len(x) < 20)]

train_text = train_data['text'].astype(str).apply(lambda x: x.lower())
train_labels =  np.asarray(train_data['channel'], dtype='int8')

val_text = val_data['text'].astype(str).apply(lambda x: x.lower())
val_labels = np.asarray(val_data['channel'], dtype='int8')

train_text = train_text \
    .apply(lambda x: re.sub('(<\S+>:?)|(\s?:\S+:\s?)|(&gt;)|([\w\.]*@[\w\.]*)', ' ', x)) \
    .apply(lambda x: re.sub('\s+', ' ', x))

val_text = val_text \
    .apply(lambda x: re.sub('(<\S+>:?)|(\s?:\S+:\s?)|(&gt;)|([\w\.]*@[\w\.]*)', ' ', x)) \
    .apply(lambda x: re.sub('\s+', ' ', x))

vocab, vocab_size = create_vocab_set()

X_train = text2sequence(train_text, vocab)
X_val = text2sequence(val_text, vocab)

X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, value=0)
X_val = pad_sequences(X_val, maxlen=MAX_SEQUENCE_LENGTH, value=0)

train_labels = to_categorical(train_labels, num_classes=12)
val_labels = to_categorical(val_labels, num_classes=12)

# callbacks initialization
# automatic generation of learning curves
callback_1 = TensorBoard(log_dir='./logs/logs_{}'.format(NAME), histogram_freq=0,
                             write_graph=False, write_images=False)
# stop training model if accuracy does not increase more than five epochs
callback_2 = EarlyStopping(monitor='val_acc', min_delta=0, patience=7, verbose=0, mode='auto')
# best model saving
callback_3 = ModelCheckpoint("../models/model_{}.hdf5".format(NAME), monitor='val_acc',
                                 save_best_only=True, verbose=0)


NAME = "char_cnn_emb"
EMBEDDING_DIM = 150
# инициализируем модель
model = Sequential()
model.add(Embedding(vocab_size+1, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH, trainable=True))
model.add(SpatialDropout1D(0.2))
model.add(Conv1D(activation="relu", filters=150, kernel_size=3, padding="valid"))
model.add(SpatialDropout1D(0.2))
model.add(BatchNormalization())
model.add(Conv1D(activation="relu", filters=150, kernel_size=3, padding="valid"))
model.add(SpatialDropout1D(0.2))
model.add(BatchNormalization())
model.add(Conv1D(activation="relu", filters=150, kernel_size=3, padding="valid"))
model.add(GlobalMaxPooling1D())
model.add(Dense(150, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(150, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(12, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()
model.fit(X_train, train_labels, validation_data=[X_val, val_labels],
         batch_size=1024, epochs=100, callbacks=[callback_1, callback_2, callback_3])
