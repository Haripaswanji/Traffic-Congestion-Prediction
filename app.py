from sklearn.preprocessing import OneHotEncoder
from flask import Flask, render_template, request
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta  # Ensure timedelta is imported

# Suppress warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
data = pd.read_csv('Train.csv')
data = data.sort_values(by=['date_time'], ascending=True).reset_index(drop=True)

# Feature engineering
last_n_hours = [1, 2, 3, 4, 5, 6]
for n in last_n_hours:
    data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)
data = data.dropna().reset_index(drop=True)

data['is_holiday'] = data['is_holiday'].map(lambda x: 1 if x != 'None' else 0).astype(int)

data['date_time'] = pd.to_datetime(data['date_time'])
data['hour'] = data['date_time'].dt.hour
data['month_day'] = data['date_time'].dt.day
data['weekday'] = data['date_time'].dt.weekday + 1
data['month'] = data['date_time'].dt.month
data['year'] = data['date_time'].dt.year

# Prepare the dataset
data.to_csv("traffic_volume_data.csv", index=None)
data = pd.read_csv("traffic_volume_data.csv")
sample_size = min(10000, len(data))  # Adjust sample size based on the available data
data = data.sample(sample_size).reset_index(drop=True)

# Identify numeric and categorical columns
label_columns = ['weather_type', 'weather_description']
numeric_columns = ['is_holiday', 'temperature', 'weekday', 'hour', 'month_day', 'year', 'month']

# One-hot encoding for categorical columns
ohe_encoder = OneHotEncoder(sparse_output=False, drop='first')  # Avoid dummy variable trap
ohe_features = ohe_encoder.fit_transform(data[label_columns])
ohe_feature_names = ohe_encoder.get_feature_names_out(label_columns)

# Combine numeric and encoded features
features = numeric_columns + list(ohe_feature_names)
target = ['traffic_volume']
X = pd.DataFrame(np.concatenate([data[numeric_columns].values, ohe_features], axis=1), columns=features)
y = data[target].values.flatten()

# Scale features and target
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

# Train the model
regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)
print(regr.predict(X[:10]))  # Print sample predictions
print(y[:10])  # Print actual values

# Flask application setup
app = Flask(__name__, static_url_path='')

@app.route('/')
def root():
    return render_template('index.html')

d = {}

@app.route('/predict', methods=['POST'])
def predict():
    d['is_holiday'] = int(request.form['isholiday'] == 'yes')
    d['temperature'] = int(request.form['temperature'])
    d['weekday'] = 0  # Set based on the input if necessary
    d['hour'] = int(request.form['time'][:2])
    d['month_day'] = int(request.form['date'][8:])
    d['year'] = int(request.form['date'][:4])
    d['month'] = int(request.form['date'][5:7])
    d['x0'] = request.form.get('x0')  # Weather type
    d['x1'] = request.form.get('x1')  # Weather description

    # Prepare one-hot encoded inputs (aligning with training)
    x0 = {f'x0_{i}': 0 for i in ['Clear', 'Clouds', 'Drizzle', 'Fog', 'Haze',
                                 'Mist', 'Rain', 'Smoke', 'Snow', 'Thunderstorm']}
    x1 = {f'x1_{i}': 0 for i in ['Sky is Clear', 'broken clouds', 'drizzle', 'few clouds',
                                  'fog', 'haze', 'heavy intensity drizzle', 'heavy intensity rain',
                                  'heavy snow', 'light intensity drizzle', 'light intensity shower rain',
                                  'light rain', 'light rain and snow', 'light shower snow',
                                  'light snow', 'mist', 'moderate rain', 'overcast clouds',
                                  'proximity shower rain', 'proximity thunderstorm',
                                  'proximity thunderstorm with drizzle', 'proximity thunderstorm with rain',
                                  'scattered clouds', 'shower drizzle', 'sky is clear', 'sleet',
                                  'smoke', 'snow', 'thunderstorm', 'thunderstorm with heavy rain',
                                  'thunderstorm with light drizzle', 'thunderstorm with light rain',
                                  'thunderstorm with rain', 'very heavy rain']}

    # Set one-hot encoded variables based on the inputs
    if d['x0'] in x0:
        x0[d['x0']] = 1
    if d['x1'] in x1:
        x1[d['x1']] = 1

    # Prepare the final input array with the correct number of features
    final = np.array([
        d['is_holiday'], d['temperature'], d['weekday'], d['hour'],
        d['month_day'], d['year'], d['month']
    ] + list(x0.values()) + list(x1.values())).reshape(1, -1)

    # Ensure final has the correct number of features
    if final.shape[1] != X.shape[1]:
        raise ValueError(f"Input feature mismatch: expected {X.shape[1]} features, got {final.shape[1]} features.")

    # Scale input
    final_scaled = x_scaler.transform(final)

    # Make the prediction
    output = regr.predict(final_scaled)
    output = y_scaler.inverse_transform(output.reshape(-1, 1))  # Inverse scale to original range
    print(output)
    return render_template('output.html', data1=d, data2=final.tolist(), prediction=output[0][0])

if __name__ == '__main__':
    app.run(debug=True)
