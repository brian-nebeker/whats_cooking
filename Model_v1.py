# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 13:25:06 2020

@author: bnebe
"""


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import re
import numpy as np
import pandas as pd 
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from tqdm import tqdm
tqdm.pandas()


train = pd.read_json('train.json')
test = pd.read_json('test.json')


# Removing outliers, only train on cuisine with >1 ingredient
train['num_ingredients'] = train['ingredients'].apply(len)
train = train[train['num_ingredients'] > 1]

# Lemmatization, change words to base form
lemmatizer = WordNetLemmatizer()
def preprocess(ingredients):
    ingredients_text = ' '.join(ingredients)
    ingredients_text = ingredients_text.lower()
    ingredients_text = ingredients_text.replace('-', ' ')
    words = []
    for word in ingredients_text.split():
        if re.findall('[0-9]', word): continue
        if len(word) <= 2: continue
        if '’' in word: continue
        word = lemmatizer.lemmatize(word)
        if len(word) > 0: words.append(word)
    return ' '.join(words)

for ingredient, expected in [
    ('Eggs', 'egg'),
    ('all-purpose flour', 'all purpose flour'),
    ('purée', 'purée'),
    ('1% low-fat milk', 'low fat milk'),
    ('half & half', 'half half'),
    ('safetida (powder)', 'safetida (powder)')
]:
    actual = preprocess([ingredient])
    assert actual == expected, f'"{expected}" is excpected but got "{actual}"'


# Split and preprocess
train['x'] = train['ingredients'].progress_apply(preprocess)
test['x'] = test['ingredients'].progress_apply(preprocess)


# Compute Tf-idf scores
# Inverse Document Frequency reflects impoirtance of word in a collection of words (recipes)
vectorizer = make_pipeline(TfidfVectorizer(sublinear_tf=True),
                           FunctionTransformer(lambda x: x.astype('float16'), validate=False)
                           )

x_train = vectorizer.fit_transform(train['x'].values)
x_train.sort_indices()
x_test = vectorizer.transform(test['x'].values)


# Encode Cuisine values
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(train['cuisine'].values)
dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))


# Model
# C-Support Vector 
# Lower C value is better, allows for larger margins between classifications
estimator = SVC(C=80,
                kernel='rbf',
                gamma=1.7,
                coef0=1,
                cache_size=500)

classifier = OneVsRestClassifier(estimator, n_jobs=-1)



# Train
classifier.fit(x_train, y_train)


# Predict
y_pred = label_encoder.inverse_transform(classifier.predict(x_train))
y_true = label_encoder.inverse_transform(y_train)

print(f'accuracy score on train data: {accuracy_score(y_true, y_pred)}')


# Submit
y_pred = label_encoder.inverse_transform(classifier.predict(x_test))
test['cuisine'] = y_pred
test[['id', 'cuisine']].to_csv('submission.csv', index=False)