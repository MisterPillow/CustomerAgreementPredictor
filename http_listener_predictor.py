import re
from pymorphy2 import MorphAnalyzer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array
import pandas as pd
import numpy as np
import pickle
# imports for deploying
from flask import Flask, jsonify, request

filename = 'CAR_predictor'
stopwords_ru = stopwords.words("russian")


class Lemmatizer(BaseEstimator):
    patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
    morph = MorphAnalyzer()

    def fit(self, X, y):
        if isinstance(X, np.ndarray):
            # Check that X and y have correct shape
            X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y
        return self

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = check_array(X, accept_sparse=True)
            return X
        return X.apply(self.lemmatize)

    #     def fit_transform(self, X, y):
    #         return self.fit(X, y).transform(X)

    def lemmatize(self, line):
        data = re.sub(self.patterns, ' ', line)
        tokens = []
        for token in data.split():
            if token and token not in stopwords_ru:
                token = token.strip()
                token = self.morph.normal_forms(token)[0]
                tokens.append(token)
        return " ".join(token for token in tokens)


def lem_features(data, features):
    lemmatizer = Lemmatizer()
    for feature in features:
        try:
            data[feature] = lemmatizer.transform(data[feature])
        except Exception as ex:
            print("Bad feature \"{}\" value".format(feature))
            raise ex


model = pickle.load(open(filename, 'rb'))
app = Flask(__name__)
features = ['CustomerInitMessage', 'SellerAnswer', 'CustomerFollowingMessage']


@app.route('/baro', methods=['POST'])
def baro_post_request():
    try:
        # Debug output
        print("Request received. PrevBaro: ", request.json['PrevBaro'], ",",
              "CustomerInitMessage: ", "\"{0}\",".format(request.json['CustomerInitMessage']),
              "SellerAnswer: ", "\"{0}\",".format(request.json['SellerAnswer']),
              "CustomerFollowingMessage: ", "\"{0}\",".format(request.json['CustomerFollowingMessage']))
        data = {
            'PrevBaro': request.json['PrevBaro'],
            'CustomerInitMessage': request.json['CustomerInitMessage'],
            'SellerAnswer': request.json['SellerAnswer'],
            'CustomerFollowingMessage': request.json['CustomerFollowingMessage']
        }
        x = pd.DataFrame(data, index=[0])
        lem_features(x, features)
        y = model.predict(x)
        return jsonify({'result': y[0]})
    except Exception as ex:
        print(ex)
        return jsonify({'result': 500, 'errorMessage': 'Something went wrong'})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
