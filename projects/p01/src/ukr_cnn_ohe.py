import os
import numpy as np
from keras import backend as K
from sklearn.model_selection import train_test_split


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# data path initialization
BASE_DIR = '../'
TEXT_DATA_DIR = BASE_DIR + 'data/'
TEXT_DATA_FILE = "ukrainian_reviews_corpus.csv"
HEADER = True

# parameters initialization
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))


def load_data():
    # function for loading data
    x = []
    iy = -1
    y = []
    with open(os.path.join(TEXT_DATA_DIR, TEXT_DATA_FILE), "r", encoding="utf-8") as f:
        if HEADER:
            _ = next(f)
        for line in f:
            if len(line) < 2 or line[1] != '|':
                x[iy] = x[iy] + line.rstrip('\n').replace("'", "")
            else:
                temp_y, temp_x = line.rstrip("\n").split("|", 1)
                x.append(temp_x.replace("'", ""))
                y.append(temp_y)
                iy += 1
    return x, y


# In[99]:

data, labels = load_data()
labels = np.asarray(labels, dtype='int8')

# In[100]:

# spliting our original data on train and validation sets
data_train, data_val, labels_train, labels_val = train_test_split(data, np.asarray(labels, dtype='int8'),
                                                                  test_size=VALIDATION_SPLIT, random_state=RANDOM_SEED,
                                                                  stratify=labels)

# initialize dictionary size and maximum sentence length
MAX_SEQUENCE_LENGTH = 400

# In[102]:

import string
from keras.preprocessing.sequence import pad_sequences

# In[103]:

ukr_alphabet = ['а', 'б', 'в', 'г', 'ґ', 'д', 'е', 'є', 'ж', 'з', 'і', 'ї', 'й',
                'к', 'л', 'м', 'н', 'о', 'п', 'р', 'с', 'т', 'у', 'и',
                'ф', 'х', 'ц', 'ч', 'ш', 'щ', 'ь', 'ю', 'я']


def create_vocab_set():
    alphabet = (ukr_alphabet + list(string.digits) +
                list(string.punctuation) + list(string.whitespace))
    vocab_size = len(alphabet)
    vocab = {}
    for ix, t in enumerate(alphabet):
        vocab[t] = ix + 1
    return vocab, vocab_size


def text2sequence(text, vocab):
    temp = []
    for review in text:
        temp.append([])
        for i in review:
            char = vocab.get(i, 0)
            if char != 0:
                temp[-1].append(char)
    return temp


vocab, vocab_size = create_vocab_set()

X_train = text2sequence(data_train, vocab)
X_val = text2sequence(data_val, vocab)

X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, value=0)
X_val = pad_sequences(X_val, maxlen=MAX_SEQUENCE_LENGTH, value=0)

import tensorflow as tf
# ohe function
def ohe(x, sz):
    return tf.to_float(tf.one_hot(x, sz, on_value=1, off_value=0, axis=-1))

from keras.models import Model
from keras.layers import Input, Lambda, MaxPooling1D, Dense, Conv1D, LSTM
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, LearningRateScheduler

from keras import optimizers
NAME = "ohe_cnn_ukrainian"
# input initialization
in_sentence = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')
# Lambda layer for ohe transformation
embedded = Lambda(ohe, output_shape=lambda x: (x[0], x[1], vocab_size), arguments={"sz": vocab_size})(in_sentence)
block = embedded
# convolutions with MaxPooling
for i in range(3):
    block = Conv1D(activation="relu", filters=100, kernel_size=4, padding="valid", trainable = True)(block)
    if i == 0:
        block = MaxPooling1D(pool_size=5)(block)
# LSTM cell
block = LSTM(128, dropout=0.1, recurrent_dropout=0.1, trainable = True)(block)
block = Dense(100, activation='relu', trainable = True)(block)
block = Dense(1, activation='sigmoid', trainable = True)(block)

# callbacks initialization
# automatic generation of learning curves
callback_1 = TensorBoard(log_dir='../logs/logs_{}'.format(NAME), histogram_freq=0,
                             write_graph=False, write_images=False)
# stop training model if accuracy does not increase more than five epochs
callback_2 = EarlyStopping(monitor='val_f1', min_delta=0, patience=5, verbose=0, mode='max')
# best model saving
callback_3 = ModelCheckpoint("models/model_{}.hdf5".format(NAME), monitor='val_f1',
                                 save_best_only=True, verbose=0, mode='max')


# initialize model
model = Model(inputs=in_sentence, outputs=block)


opt = optimizers.Adam(lr=0.001)
model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=[f1])


model.summary()

model.fit(X_train, labels_train, validation_data=[X_val, labels_val],
          batch_size=256, epochs=1000, callbacks=[callback_1, callback_2, callback_3])