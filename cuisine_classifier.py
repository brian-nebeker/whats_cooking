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

class CookingClassifier(object):
    def __init__(self,
                 tune_parameters: bool = True,
                 parameter_dict: dict = None):
        self.tune_parameters = tune_parameters
        self.parameter_dict = parameter_dict

    def fit(self):
        #fit model
        pass

    def predict(self):
        #predict data
        pass

    def tune_model(self):
        if self.tune_parameters:
            self._tune_model()

    def _read_data(self):
        self.train = pd.read_json('./data/train.json')
        self.test = pd.read_json('./data/test.json')

        self.train['num_ingredients'] = self.train['ingredients'].apply(len)
        self.train = self.train[self.train['num_ingredients'] > 1]

    def _process_text(self, ingredients):
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

    def _pre_process(self):
        self.train['x'] = self.train['ingredients'].progress_apply(_process_text)
        self.test['x'] = self.test['ingredients'].progress_apply(_process_text)

