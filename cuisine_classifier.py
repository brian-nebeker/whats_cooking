import re
import nltk
import joblib
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder
from tqdm import tqdm

tqdm.pandas()
nltk.download('wordnet')

class CookingClassifier:
    def __init__(self,
                 do_train: bool = True,
                 pred_ingredients: list = None,
                 asset_file_path: str = './data/'):
        self.do_train = do_train
        self.pred_ingredients = pred_ingredients
        self.asset_file_path = asset_file_path

    def execute(self):
        if self.do_train:
            self._read_data()
            self._preprocess()
            self._vectorize()
            self._label_encoder()
            self._train_model()
            print("Trained Model")
        if self.pred_ingredients is not None:
            self.predict()
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

    def _vectorize_old(self):
        self.vectorizer = make_pipeline(TfidfVectorizer(sublinear_tf=True),
                                        FunctionTransformer(lambda x: x.astype('float16'), validate=False))
        self.x_train = self.vectorizer.fit_transform(self.train['x'].values)
        self.x_train.sort_indices()
        self.x_test = self.vectorizer.transform(self.test['x'].values)
        joblib.dump(value=self.vectorizer, filename=f"{self.asset_file_path}temp")

    def _vectorize(self):
        self.vectorizer = TfidfVectorizer(sublinear_tf=True)
        self.x_train = self.vectorizer.fit_transform(self.train['x'].values)
        self.x_train.sort_indices()
        self.x_test = self.vectorizer.transform(self.test['x'].values)
        joblib.dump(value=self.vectorizer, filename=f"{self.asset_file_path}vectorizer.pkl")

    def _label_encoder(self):
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(self.train['cuisine'].values)
        joblib.dump(value=self.label_encoder, filename=f"{self.asset_file_path}label_encoder.pickle")

    def _train_model(self):
        self.estimator = SVC(C=80,
                        kernel='rbf',
                        gamma=1.7,
                        coef0=1,
                        cache_size=500)

        self.classifier = OneVsRestClassifier(self.estimator, n_jobs=-1)
        self.classifier.fit(self.x_train, self.y_train)

        joblib.dump(value=self.classifier, filename=f"{self.asset_file_path}classifier.pickle")

        self.y_pred = self.label_encoder.inverse_transform(self.classifier.predict(self.x_train))
        self.y_true = self.label_encoder.inverse_transform(self.y_train)

        print("Classification Report for Test Data")
        print(classification_report(self.y_true, self.y_pred))

    def predict(self):
        vectorizer = joblib.load(f"{self.asset_file_path}vectorizer.pkl")
        label_encoder = joblib.load(f"{self.asset_file_path}label_encoder.pickle")
        classifier = joblib.load(f"{self.asset_file_path}classifier.pickle")

        processed_ingredients = self._process_ingredients(self.pred_ingredients)
        vectorized_ingredients = vectorizer.transform([processed_ingredients])
        prediction = label_encoder.inverse_transform(classifier.predict(vectorized_ingredients))

        print(f"Prediction: {prediction}")
