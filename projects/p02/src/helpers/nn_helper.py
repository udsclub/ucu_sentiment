import string
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import os
import re
from datetime import date
from fastnumbers import isfloat, isint
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Embedding
from keras.layers import BatchNormalization, SpatialDropout1D, Conv1D, Dense, Dropout, GlobalMaxPooling1D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.utils import to_categorical

DIR_TRAIN = '../../data'
dir_models = '../../models'

MAPPINGS = {
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

RUS_ALPHABET = ['а','б','в','г','д','е','ё','ж','з','и','й','к','л','м','н','о','п','р','с','т','у','ф','х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я']

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


def load_data_for_nn():
    data = pd.read_csv(os.path.join(DIR_TRAIN, 'train_set.csv'), usecols=range(1, 11),
                       parse_dates=['timestamp', 'thread_timestamp'])
    data = data[
        data.channel.isin(['career', 'big_data', 'deep_learning', 'kaggle_crackers',
                           'lang_python', 'lang_r', 'nlp', 'theory_and_practice', 'welcome', 'bayesian', '_meetings',
                           'datasets']) &
        data.main_msg
        ]

    # data_train = data.
    date_before = date(2017, 4, 1)
    train = data[data['timestamp'] <= date_before]
    val = data[data['timestamp'] > date_before]

    train_data = train[['channel', 'text']].reset_index()[['channel', 'text']]
    train_data['channel'] = train_data.channel.map(MAPPINGS)
    train_data = train_data.sort_values('channel').reset_index()[['channel', 'text']]

    val_data = val[['channel', 'text']].reset_index()[['channel', 'text']]
    val_data['channel'] = val_data.channel.map(MAPPINGS)
    val_data = val_data.sort_values('channel').reset_index()[['channel', 'text']]

    train_data.text = train_data.text.astype(str) \
        .apply(lambda x: re.sub('(<\S+>:?)|(\s?:\S+:\s?)|(&gt;)|([\w\.]*@[\w\.]*)', ' ', x)) \
        .apply(lambda x: re.sub('\s+', ' ', x))
    train_data = train_data[~train_data.text.apply(lambda x: isfloat(x) or isint(x) or len(x) < 20)]

    val_data.text = val_data.text.astype(str) \
        .apply(lambda x: re.sub('(<\S+>:?)|(\s?:\S+:\s?)|(&gt;)|([\w\.]*@[\w\.]*)', ' ', x)) \
        .apply(lambda x: re.sub('\s+', ' ', x))
    val_data = val_data[~val_data.text.apply(lambda x: isfloat(x) or isint(x) or len(x) < 20)]

    train_text = train_data['text'].astype(str).apply(lambda x: x.lower())
    train_labels = np.asarray(train_data['channel'], dtype='int8')

    val_text = val_data['text'].astype(str).apply(lambda x: x.lower())
    val_labels = np.asarray(val_data['channel'], dtype='int8')

    vocab, vocab_size = create_vocab_set()

    X_train = text2sequence(train_text, vocab)
    X_val = text2sequence(val_text, vocab)

    X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, value=0)
    X_val = pad_sequences(X_val, maxlen=MAX_SEQUENCE_LENGTH, value=0)

    train_labels = to_categorical(train_labels, num_classes=12)
    val_labels = to_categorical(val_labels, num_classes=12)

    return X_train, train_labels, X_val, val_labels

def transform(tokenizer_object, train, test):
    sequences_train = tokenizer_object.texts_to_sequences(train)  # transform words to its indexes
    sequences_test = tokenizer_object.texts_to_sequences(test)

    word_indexes = tokenizer_object.word_index  # dictionary of word:index

    # transform a list to numpy array with shape (nb_samples, MAX_SEQUENCE_LENGTH)
    # be careful because it takes only last MAX_SEQUENCE_LENGTH words
    train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH)
    test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)

    return train, test, word_indexes

def train_model_lstm_channel_classification():
    data_train, labels_train, data_test, labels_test = load_data()
    print(len(data_train), len(data_test))
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='"#$%&()*+-/:;<=>@[\\]^{|}~\t\n,.')
    tokenizer.fit_on_texts(data_train)

    X_train, X_test, word_index = transform(tokenizer, data_train, data_test)
    y_train, y_test = to_categorical(np.asarray(labels_train), num_classes=12), to_categorical(np.asarray(labels_test), num_classes=12)

    embedding_matrix = prepare_embeddings(word_index)

    # инициализируем слой эмбеддингов
    NAME = "lstm_channel_classification"

    # callbacks initialization
    # automatic generation of learning curves
    callback_1 = TensorBoard(log_dir='./logs/logs_{}'.format(NAME), histogram_freq=0,
                             write_graph=False, write_images=False)
    # stop training model if accuracy does not increase more than five epochs
    callback_2 = EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=0, mode='auto')
    # best model saving
    callback_3 = ModelCheckpoint("../models/model_{}.hdf5".format(NAME), monitor='val_acc',
                                 save_best_only=True, verbose=0)

    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False,
                                mask_zero=True)

    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(200, dropout=0.1, recurrent_dropout=0.1)))
    model.add(Dropout(0.2))
    model.add(Dense(12))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()
    model.fit(X_train, y_train, validation_data=[X_test, y_test],
              batch_size=1024, epochs=100, callbacks=[callback_1, callback_2, callback_3])
    return model

def load_model(name):
    model = load_model("../models/{}".format(name))
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    model.summary()
    return model