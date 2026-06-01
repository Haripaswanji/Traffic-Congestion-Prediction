from sklearn.preprocessing import StandardScaler, MinMaxScaler
from functools import reduce
from sklearn.preprocessing import OneHotEncoder
from flask import Flask, render_template, request
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from flask import Flask, render_template, request
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def unique(list1):
    ans = reduce(lambda re, x: re+[x] if x not in re else re, list1, [])
    print(ans)


n1features = []
n2features = []
x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()
regr = MLPRegressor(random_state=1, max_iter=500)


app = Flask(__name__, static_url_path='')


@app.route('/')
def root():
    return render_template('home.html')


@app.route('/train')
def train():
    # Load and preprocess the dataset
    data = pd.read_csv('static/Train.csv')
    data = data.sort_values(by=['date_time']).reset_index(drop=True)
    
    # Creating lag features for traffic volume
    last_n_hours = [1, 2, 3, 4, 5, 6]
    for n in last_n_hours:
        data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)
    data = data.dropna().reset_index(drop=True)

    # Convert 'is_holiday' column to binary
    data.loc[data['is_holiday'] != 'None', 'is_holiday'] = 1
    data.loc[data['is_holiday'] == 'None', 'is_holiday'] = 0
    data['is_holiday'] = data['is_holiday'].astype(int)

    # Feature engineering based on 'date_time'
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['hour'] = data['date_time'].dt.hour
    data['month_day'] = data['date_time'].dt.day
    data['weekday'] = data['date_time'].dt.weekday + 1
    data['month'] = data['date_time'].dt.month
    data['year'] = data['date_time'].dt.year
    data.to_csv("traffic_volume_data.csv", index=None)

    # Sampling for visualization and analysis
    data = pd.read_csv("traffic_volume_data.csv")
    num_samples = min(10000, len(data))  # Sample up to the number of available rows
    data = data.sample(num_samples).reset_index(drop=True)

    # Extracting weather-related columns
    n1 = data['weather_type']
    n2 = data['weather_description']
    
    # Convert weather types to numerical features
    unique(n1)
    unique(n2)
    n1features = ['Rain', 'Clouds', 'Clear', 'Snow', 'Mist',
                  'Drizzle', 'Haze', 'Thunderstorm', 'Fog', 'Smoke', 'Squall']
    n2features = ['light rain', 'few clouds', 'Sky is Clear', 'light snow', 'sky is clear', 
                  'mist', 'broken clouds', 'moderate rain', 'drizzle', 'overcast clouds', 
                  'scattered clouds', 'haze', 'proximity thunderstorm', 
                  'light intensity drizzle', 'heavy snow', 'heavy intensity rain', 
                  'fog', 'heavy intensity drizzle', 'shower snow', 'snow', 
                  'thunderstorm with rain', 'thunderstorm with heavy rain', 
                  'thunderstorm with light rain', 'proximity thunderstorm with rain', 
                  'thunderstorm with drizzle', 'smoke', 'thunderstorm', 
                  'proximity shower rain', 'very heavy rain', 
                  'proximity thunderstorm with drizzle', 'light rain and snow', 
                  'light intensity shower rain', 'SQUALLS', 
                  'shower drizzle', 'thunderstorm with light drizzle']
    
    n11 = []
    n22 = []
    for i in range(num_samples):
        if(n1[i]) not in n1features:
            n11.append(0)
        else:
            n11.append((n1features.index(n1[i])) + 1)
        if n2[i] not in n2features:
            n22.append(0)
        else:
            n22.append((n2features.index(n2[i])) + 1)
    
    data['weather_type'] = n11
    data['weather_description'] = n22

    # Define features and target
    features = ['is_holiday', 'temperature', 'weekday', 'hour', 
                'month_day', 'year', 'month', 'weather_type', 'weather_description']
    target = ['traffic_volume']
    
    X = data[features]
    y = data[target].values.flatten()

    # Feature Scaling
    X = x_scaler.fit_transform(X)
    y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Train the model
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
    regr.fit(trainX, trainY)

    # Model evaluation
    y_pred = regr.predict(testX)
    print('Mean Absolute Error:', mean_absolute_error(testY, y_pred))

    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    ip = []
    ip.append(1 if request.form['isholiday'] == 'yes' else 0)
    ip.append(int(request.form['temperature']))
    ip.append(int(request.form['day']))
    ip.append(int(request.form['time'][:2]))
    D = request.form['date']
    ip.append(int(D[8:]))  # day
    ip.append(int(D[:4]))   # year
    ip.append(int(D[5:7]))  # month
    s1 = request.form.get('x0')
    s2 = request.form.get('x1')
    ip.append((n1features.index(s1) + 1) if s1 in n1features else 0)
    ip.append((n2features.index(s2) + 1) if s2 in n2features else 0)
    
    ip = x_scaler.transform([ip])
    out = regr.predict(ip)
    print('Before inverse Scaling:', out)
    y_pred = y_scaler.inverse_transform([out])
    print('Traffic Volume:', y_pred)

    # Traffic volume assessment
    if y_pred <= 1000:
        statement = "No Traffic"
    elif 1000 < y_pred <= 3000:
        statement = "Busy or Normal Traffic"
    elif 3000 < y_pred <= 5500:
        statement = "Heavy Traffic"
    else:
        statement = "Worst case"

    return render_template('output.html', data1=ip, op=y_pred, statement=statement)


if __name__ == '__main__':
    app.run(debug=True)
