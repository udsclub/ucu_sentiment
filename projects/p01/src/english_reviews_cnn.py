import os
import numpy as np
from keras import backend as K
import csv

from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences
import string

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# data path initialization
TEXT_DATA_FILE_NEG = "../data/train_neg.csv"
TEXT_DATA_FILE_POS = "../data/train_pos.csv"
HEADER = False

# parameters initialization
VALIDATION_SPLIT = 0.1
RANDOM_SEED = 42


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
    x = []
    y = []

    fin1 = open(TEXT_DATA_FILE_NEG, "r", encoding="utf-8")
    fin1_reader = csv.reader(fin1)
    fin2 = open(TEXT_DATA_FILE_POS, "r", encoding="utf-8")
    fin2_reader = csv.reader(fin2)

    if HEADER:
        next(fin1_reader)
        next(fin2_reader)
    for row in fin1_reader:
        x.append(row[0])
        y.append(1)
    for row in fin2_reader:
        x.append(row[0])
        y.append(0)
    return x, y


data, labels = load_data()
labels = np.asarray(labels, dtype = 'int8')

# spliting our original data on train and validation sets
data_train, data_val, labels_train, labels_val = train_test_split(data,
                                                                  np.asarray(labels, dtype = 'int8'),
                                                                  test_size = VALIDATION_SPLIT,
                                                                  random_state = RANDOM_SEED,
                                                                  stratify = labels)

MAX_SEQUENCE_LENGTH = 600


def create_vocab_set():
    alphabet = (list(string.ascii_lowercase) + list(string.digits) + list(string.punctuation) + list(string.whitespace))
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
            char = vocab.get(i, 0)
            if char != 0:
                temp[-1].append(char)
    return temp

vocab, vocab_size = create_vocab_set()

X_train = text2sequence(data_train, vocab)
X_val = text2sequence(data_val, vocab)

X_train = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, value=0)
X_val = pad_sequences(X_val, maxlen=MAX_SEQUENCE_LENGTH, value=0)

from keras.models import Sequential
from keras.layers import GlobalMaxPooling1D, Conv1D, Dropout, Embedding, Dense
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

NAME = "english_char_cnn_emb"
EMBEDDING_DIM = 100

# initialize model
from keras.layers import concatenate, Input
from keras.models import Model

words_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64')

x = Embedding(vocab_size + 1,
              EMBEDDING_DIM,
              input_length=MAX_SEQUENCE_LENGTH,
              trainable=True)(words_input)

x1 = Conv1D(activation="relu", filters=100, kernel_size=2, padding="same")(x)
x2 = Conv1D(activation="relu", filters=100, kernel_size=3, padding="same")(x)
x3 = Conv1D(activation="relu", filters=100, kernel_size=4, padding="same")(x)
x4 = Conv1D(activation="relu", filters=100, kernel_size=5, padding="same")(x)
x = concatenate([x1, x2, x3, x4])
x = GlobalMaxPooling1D()(x)
x = Dense(100, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)
model = Model(inputs=words_input, outputs=output)

# callbacks initialization
# automatic generation of learning curves
callback_1 = TensorBoard(log_dir='../logs/logs_{}'.format(NAME), histogram_freq=0,
                             write_graph=False, write_images=False)
# stop training model if accuracy does not increase more than five epochs
callback_2 = EarlyStopping(monitor='val_f1', min_delta=0, patience=5, verbose=0, mode='max')
# best model saving
callback_3 = ModelCheckpoint("models/model_{}.hdf5".format(NAME), monitor='val_f1',
                                 save_best_only=True, verbose=0, mode='max')

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[f1])

model.summary()
model.fit(X_train, labels_train, validation_data=[X_val, labels_val],
          batch_size=1024, epochs=1000, callbacks=[callback_1, callback_2, callback_3])
