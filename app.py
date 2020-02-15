from flask import Flask
import pickle
import numpy as np

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore')

app = Flask(__name__)
model = pickle.load(open('/Users/gauravkantrod/Society_5/model.pkl', 'rb'))

@app.route('/')
def app_status():
    return "Application is up and running."


@app.route('/predict')
def predict():
    features = [np.array([1,3,20,63,90,0])]
    prediction = model.predict(features)
    return {"predicted_yield":prediction[0]}


if __name__ == '__main__':
    app.run(debug = True)