import pandas as pd
import numpy as np
import os
import re
from datetime import date
from fastnumbers import isfloat, isint
from sklearn.feature_extraction.text import TfidfVectorizer
import lightgbm as lgb

dir_train = '../data'
dir_models = '../models'

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

data = pd.read_csv(os.path.join(dir_train, 'train_set.csv'), usecols=range(1, 11),
                   parse_dates=['timestamp', 'thread_timestamp'])
data = data[
    data.channel.isin(['career', 'big_data', 'deep_learning', 'kaggle_crackers',
                       'lang_python',  'lang_r', 'nlp', 'theory_and_practice',
                       'welcome', 'bayesian', '_meetings', 'datasets']) &
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

train_data.text = train_data.text.astype(str)\
    .apply(lambda x: re.sub('(<\S+>:?)|(\s?:\S+:\s?)|(&gt;)|([\w]*@[\w]*)', ' ', x))\
    .apply(lambda x: re.sub('\s+', ' ', x))
train_data = train_data[~train_data.text.apply(lambda x: isfloat(x) or isint(x) or len(x) < 20)]

val_data.text = val_data.text.astype(str)\
    .apply(lambda x: re.sub('(<\S+>:?)|(\s?:\S+:\s?)|(&gt;)|([\w]*@[\w]*)', ' ', x))\
    .apply(lambda x: re.sub('\s+', ' ', x))
val_data = val_data[~val_data.text.apply(lambda x: isfloat(x) or isint(x) or len(x) < 20)]

train_text = train_data['text'].astype(str).apply(lambda x: x.lower())
train_labels = np.asarray(train_data['channel'], dtype='int8')

val_text = val_data['text'].astype(str).apply(lambda x: x.lower())
val_labels = np.asarray(val_data['channel'], dtype='int8')

vectorizer = TfidfVectorizer(analyzer='char', max_features=1000000, ngram_range=(1, 7))
train_matrix = vectorizer.fit_transform(train_text)
val_matrix = vectorizer.transform(val_text)

lgb_train = lgb.Dataset(train_matrix, label=train_labels)
lgb_val = lgb.Dataset(val_matrix, label=val_labels, reference=lgb_train)

lgb_params = {
    'learning_rate': 0.1,
    'seed': 42,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'feature_fraction': 0.7,
    'application': 'multiclass',
    'num_leaves': 155,
    'min_child_weight': 1,
    'num_class': 12,
    'metric': 'multi_error',
    'verbose': 0,
    'num_threads': 8
}

eval_matrix = [lgb_val]
eval_name = ['lgb_val']

final_lgb = lgb.train(lgb_params, lgb_train, valid_sets=eval_matrix, valid_names=eval_name,
                      num_boost_round=1000, early_stopping_rounds=10, verbose_eval=5)

final_lgb.save_model(os.path.join(dir_models, 'lightgbm_model.txt'), num_iteration=final_lgb.best_iteration)
