import string
import pandas as pd
import numpy as np
import os
import re
from datetime import date
from fastnumbers import isfloat, isint
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer

dir_train = '../../data'
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


def load_data_gbm():
    data = pd.read_csv(os.path.join(dir_train, 'train_set.csv'), usecols=range(1, 11),
                       parse_dates=['timestamp', 'thread_timestamp'])
    data = data[
        data.channel.isin(['career', 'big_data', 'deep_learning', 'kaggle_crackers',
                           'lang_python', 'lang_r', 'nlp', 'theory_and_practice', 'welcome', 'bayesian', '_meetings',
                           'datasets']) &
        data.main_msg
        ]

    date_before = date(2017, 4, 1)
    train = data[data['timestamp'] <= date_before]
    val = data[data['timestamp'] > date_before]

    train_data = train[['channel', 'text']].reset_index()[['channel', 'text']]
    train_data['channel'] = train_data.channel.map(mappings)
    train_data = train_data.sort_values('channel').reset_index()[['channel', 'text']]

    val_data = val[['channel', 'text']].reset_index()[['channel', 'text']]
    val_data['channel'] = val_data.channel.map(mappings)
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

    return train_text, val_text, train_labels, val_labels

def prepare_data(train_text, val_text):
    vectorizer = TfidfVectorizer(analyzer = 'char', max_features = 1000000, ngram_range = (1, 4))
    train_matrix = vectorizer.fit_transform(train_text)
    val_matrix = vectorizer.transform(val_text)
    return train_matrix, val_matrix

def train_light_gbm(train_text, val_text, train_labels, val_labels):
    xgb_train = xgb.DMatrix(train_matrix, label=train_labels)
    xgb_val = xgb.DMatrix(val_matrix, label=val_labels)

    xgb_params = {
        'eta': 0.1,
        'seed': 42,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'objective': 'multi:softmax',
        'max_depth': 7,
        'min_child_weight': 1,
        'num_class': 12,
        'eval_metric': 'merror'
    }

    eval_matrix = [(xgb_val, 'xgb_val')]

    final_xgb = xgb.train(xgb_params, xgb_train, num_boost_round = 1000, evals = eval_matrix, early_stopping_rounds=20,
                        verbose_eval=5)
    return final_xgb

def save_model(model, name):
    model_path = os.path.join(dir_models, '{0}.model'.format(name))
    try:
        model_path = open(fn, 'r')
    except IOError:
        model_path = open(fn, 'w')
    model.save_model(model_path)

def load_model(name):
    booster = xgb.Booster()
    return booster.load_model(os.path.join(dir_models, '{0}.model'.format(name)))