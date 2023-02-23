import re
import numpy as np
import pandas as pd
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
# from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, make_union
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from tqdm import tqdm
tqdm.pandas()

class CookingClassifier(object):
    def __init__(self,
                 do_train: bool = True,
                 do_predict: bool = True,
                 asset_file_path = './data/'):
        self.do_train = do_train
        self.do_predict = do_predict
        self.asset_file_path = asset_file_path


    def execute(self):
        self._read_data()
        self._preprocess()
        self._vectorize()
        self._label_encoder()
        if self.do_train:
            self._train_model()
            print("Trained Model")
        if self.do_predict:
            self._predict()
            print("Predicted Test Set")
        print("Finished Execute")

    def _read_data(self):
        self.train = pd.read_json(f"{self.asset_file_path}train.json")
        self.test = pd.read_json(f"{self.asset_file_path}test.json")

        self.train.loc[:, 'num_ingredients'] = self.train.loc[:, 'ingredients'].apply(len)
        self.train = self.train[self.train['num_ingredients'] > 1]


    def _process_ingredients(self, ingredients):
        lemmatizer = WordNetLemmatizer()
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

    def _preprocess(self):
        self.train['x'] = self.train['ingredients'].progress_apply(self._process_ingredients)
        self.test['x'] = self.test['ingredients'].progress_apply(self._process_ingredients)

    def _vectorize(self):
        vectorizer = make_pipeline(TfidfVectorizer(sublinear_tf=True),
                                   FunctionTransformer(lambda x: x.astype('float16'), validate=False)
                                   )
        self.x_train = vectorizer.fit_transform(self.train['x'].values)
        self.x_train.sort_indices()
        self.x_test = vectorizer.transform(self.test['x'].values)

    def _label_encoder(self):
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(self.train['cuisine'].values)

    def _train_model(self):
        self.estimator = SVC(C=80,
                        kernel='rbf',
                        gamma=1.7,
                        coef0=1,
                        cache_size=500)

        self.classifier = OneVsRestClassifier(self.estimator, n_jobs=-1)
        self.classifier.fit(self.x_train, self.y_train)

    def _predict(self):
        self.y_pred = self.label_encoder.inverse_transform(self.classifier.predict(self.x_train))
        self.y_true = self.label_encoder.inverse_transform(self.y_train)
