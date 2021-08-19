import re
from pymorphy2 import MorphAnalyzer
import nltk

from nltk.corpus import stopwords
from sklearn.base import BaseEstimator
import pandas as pd
import pickle
# imports for deploying
from flask import Flask, jsonify, request

nltk.download('stopwords')
model_baro_filename = 'baro_predictor'
model_temperament_filename = 'temperament_predictor'
stopwords_ru = stopwords.words("russian")


class Lemmatizer(BaseEstimator):
    patterns = "[A-Za-z0-9!#$%&'()*+,./:;<=>?@[\]^_`{|}~—\"\-]+"
    morph = MorphAnalyzer()

    def transform(self, X):
        return X.apply(self.lemmatize)

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
            print("Bad feature \"{}\" value".format(feature),
                  flush=True)  # Required to print message to log when app is started via gunicorn
            raise ex


model_baro = pickle.load(open(model_baro_filename, 'rb'))
model_temperament = pickle.load(open(model_temperament_filename, 'rb'))
app = Flask(__name__)
text_features = ['CustomerInitMessage', 'SellerAnswer', 'CustomerFollowingMessage']


def format_param(param):
    return "\"{0}\"".format(param) if isinstance(param, str) else "{0}".format(param)


@app.route('/baro', methods=['POST'])
def baro_post_request():
    try:
        # Debug output
        print(
            "Request received. PrevBaro: {0}, CustomerInitMessage: {1}, SellerAnswer: {2}, CustomerFollowingMessage: {3}".format(
                request.json['PrevBaro'],
                format_param(request.json['CustomerInitMessage']),
                format_param(request.json['SellerAnswer']),
                format_param(request.json['CustomerFollowingMessage'])
            ), flush=True)  # Required to print message to log when app is started via gunicorn
        data = {
            'PrevBaro': request.json['PrevBaro'],
            'CustomerInitMessage': request.json['CustomerInitMessage'],
            'SellerAnswer': request.json['SellerAnswer'],
            'CustomerFollowingMessage': request.json['CustomerFollowingMessage']
        }
        x = pd.DataFrame(data, index=[0])
        lem_features(data=x, features=text_features)
        y = model_baro.predict(x)
        print("Predicted value: {0}".format(y[0]),
              flush=True)  # Required to print message to log when app is started via gunicorn
        return jsonify({'result': y[0]})
    except Exception as ex:
        print(ex)
        return jsonify({'result': 0.5, 'errorMessage': 'Something went wrong'})


@app.route('/temperament', methods=['POST'])
def temperament_post_request():
    """ Human temperaments: choleric, phlegmatic, sanguine and melancholic"""
    try:
        # Debug output
        print(
            "Temperament POST request received. CustomerInitMessage: {0}, SellerAnswer: {1}, CustomerFollowingMessage: {2}".format(
                format_param(request.json['CustomerInitMessage']),
                format_param(request.json['SellerAnswer']),
                format_param(request.json['CustomerFollowingMessage'])
            ), flush=True)  # Required to print message to log when app is started via gunicorn
        data = {
            'CustomerInitMessage': request.json['CustomerInitMessage'],
            'SellerAnswer': request.json['SellerAnswer'],
            'CustomerFollowingMessage': request.json['CustomerFollowingMessage']
        }
        x = pd.DataFrame(data, index=[0])
        lem_features(data=x, features=text_features)
        y = model_temperament.predict(x)
        print("Predicted temperament: choleric {0}, phlegmatic {1}, sanguine {2}, melancholic {3}"
              .format(y[0][0], y[0][1], y[0][2], y[0][3]),
              flush=True)  # Required to print message to log when app is started via gunicorn
        return jsonify(
            {
                "temperament":
                    {
                        "choleric": y[0][0],
                        "phlegmatic": y[0][1],
                        "sanguine": y[0][2],
                        "melancholic": y[0][3]
                    }
            })
    except Exception as ex:
        print(ex)
        return jsonify(
            {
                'errorMessage': 'Something went wrong',
                "temperament":
                    {
                        "choleric": 0,
                        "phlegmatic": 0,
                        "sanguine": 0,
                        "melancholic": 0
                    }
            })


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
