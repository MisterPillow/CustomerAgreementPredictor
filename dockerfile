FROM python:3.7

RUN python -m pip install flask gunicorn numpy pandas pillow pymorphy2 nltk sklearn

WORKDIR /app

ADD http_listener_predictor.py http_listener_predictor.py
ADD CAR_predictor CAR_predictor

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "http_listener_predictor:app", "--log-level", "debug" ]
