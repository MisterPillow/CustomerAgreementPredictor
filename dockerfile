FROM python:3.7

RUN python -m pip install flask gunicorn numpy pandas pillow pymorphy2 nltk sklearn vosk wave

WORKDIR /app

ADD speech/vosk-model-small-ru-0.15 speech/vosk-model-small-ru-0.15
ADD http_listener_predictor.py http_listener_predictor.py
ADD baro_predictor baro_predictor
ADD temperament_predictor temperament_predictor

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "http_listener_predictor:app", "--log-level", "debug" ]
