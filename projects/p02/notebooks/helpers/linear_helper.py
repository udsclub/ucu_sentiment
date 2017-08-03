import pymorphy2
import nltk
import stop_words
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, cross_val_score, StratifiedKFold, ParameterGrid
from sklearn import preprocessing
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import date
import fastnumbers
import re
from fastnumbers import isfloat, isint
import pickle
import os
import pandas as pd
import numpy as np

DIR_TRAIN = '../../data'
dir_models = '../../models'

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

RUS_ALPHABET = ['а','б','в','г','д','е','ё','ж','з','и','й','к','л','м','н','о','п','р','с','т','у','ф','х','ц','ч','ш','щ','ъ','ы','ь','э','ю','я']

def load_data_for_linear():
    data = pd.read_csv(os.path.join(DIR_TRAIN, 'train_set.csv'), usecols=range(1,11), parse_dates=['timestamp', 'thread_timestamp'])
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

    train_data.text = train_data.text.astype(str)\
    .apply(lambda x: re.sub('(<\S+>:?)|(\s?:\S+:\s?)|(&gt;)|([\w\.]*@[\w\.]*)', ' ', x))\
    .apply(lambda x: re.sub('\s+', ' ', x))
    train_data = train_data[~train_data.text.apply(lambda x: isfloat(x) or isint(x) or len(x) < 20)]

    val_data.text = val_data.text.astype(str)\
    .apply(lambda x: re.sub('(<\S+>:?)|(\s?:\S+:\s?)|(&gt;)|([\w\.]*@[\w\.]*)', ' ', x))\
    .apply(lambda x: re.sub('\s+', ' ', x))
    val_data = val_data[~val_data.text.apply(lambda x: isfloat(x) or isint(x) or len(x) < 20)]

    train_text = train_data['text'].astype(str).apply(lambda x: x.lower())
    train_labels =  np.asarray(train_data['channel'], dtype='int8')

    val_text = val_data['text'].astype(str).apply(lambda x: x.lower())
    val_labels = np.asarray(val_data['channel'], dtype='int8')
    return train_text, val_text, train_labels, val_labels

def train_linear_model(train_text, train_labels):
    # library for lemmatization and stop_words
    morph = pymorphy2.MorphAnalyzer()
    # without tuning accuracy_score = 50.81%
    # accuracy_score = 55.33%
    classifier = Pipeline([
        ('vectorizer', TfidfVectorizer(analyzer = 'char', max_features = 1000000,
                                                           ngram_range = (1, 4))),
        ('clf', OneVsRestClassifier(LogisticRegression(),n_jobs=-1))])

    classifier.fit(train_text, train_labels)
    return classifier

def save_model(model, name):
    with open(os.path.join(dir_models, '{0}.pickle'.format(name)), 'wb') as f:
        pickle.dump(model, f)

def load_model(name):
    with open(name, 'rb') as f:
        model = pickle.load(f)
    return model