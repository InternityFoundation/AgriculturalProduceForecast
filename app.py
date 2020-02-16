from flask import Flask, request, render_template
import pickle
import numpy as np
from wwo_hist import retrieve_hist_data

from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore')

app = Flask(__name__, template_folder='template')
model = pickle.load(open('models/model.pkl', 'rb'))

START_DATE = '15-Nov-2019'
END_DATE = '15-FEB-2020'
API_KEY = 'b1ec70731e66454d9ec62250201602'
LOCATION_LIST = ['pune']

hist_weather_data = retrieve_hist_data(API_KEY,
                                LOCATION_LIST,
                                START_DATE,
                                END_DATE,
                                location_label = False,
                                export_csv = True,
                                store_df = True)

@app.route('/')
def app_status():
    return "Application is up and running."

@app.route('/form')
def show_form():
    return render_template('form.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    month_lst = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    monthwise_prediction = {}

    month = int(request.form['month'])
    sunlight = request.form['sunlight']
    temperature = int(request.form['temperature'])
    humidity = int(request.form['humidity'])
    electricity = int(request.form['electricity'])
    rainfall = int(request.form['rainfall'])

    input_features = [month, temperature, humidity, electricity, rainfall]

    if sunlight.lower() == 'h':
        input_features.extend([0, 1])
    if sunlight.lower() == 'm':
        input_features.extend([1, 0])
    else:
        input_features.extend([0, 0])

    # month	temperature	humidity electricity rainfall sunlight__2 sunlight__3
    features = [np.array(input_features)]
    prediction = model.predict(features)
    return {"predicted_yield_"+str(month): prediction[0]}


if __name__ == '__main__':
    app.run(debug=True)
