from sklearn.linear_model import LinearRegression
from mlxtend.regressor import StackingRegressor
from skopt.space import Real, Categorical, Integer
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import zscore
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import seaborn as sns  # Visualization module based on matplotlib
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load the data from the corrected file path
data = pd.read_csv(r"C:\Users\nishc\OneDrive\Desktop\Traffic_Prediction-main\Train.csv")

# Sort data by date_time
data = data.sort_values(by=['date_time'], ascending=True).reset_index(drop=True)

# Create lag features for traffic volume
last_n_hours = [1, 2, 3, 4, 5, 6]
for n in last_n_hours:
    data[f'last_{n}_hour_traffic'] = data['traffic_volume'].shift(n)

# Drop rows with NaN values
data = data.dropna().reset_index(drop=True)

# Encode holidays
data.loc[data['is_holiday'] != 'None', 'is_holiday'] = 1
data.loc[data['is_holiday'] == 'None', 'is_holiday'] = 0
data['is_holiday'] = data['is_holiday'].astype(int)

# Extract datetime features
data['date_time'] = pd.to_datetime(data['date_time'])
data['hour'] = data['date_time'].map(lambda x: int(x.strftime("%H")))
data['month_day'] = data['date_time'].map(lambda x: int(x.strftime("%d")))
data['weekday'] = data['date_time'].map(lambda x: x.weekday() + 1)
data['month'] = data['date_time'].map(lambda x: int(x.strftime("%m")))
data['year'] = data['date_time'].map(lambda x: int(x.strftime("%Y")))

# Save processed data for reuse
data.to_csv("traffic_volume_data.csv", index=None)
print("Processed data saved to traffic_volume_data.csv.")

# Reload processed data
data = pd.read_csv("traffic_volume_data.csv")

# Sample up to 10,000 rows or the entire dataset if fewer rows are available
data = data.sample(min(10000, len(data))).reset_index(drop=True)

# Define columns for further processing
label_columns = ['weather_type', 'weather_description']
numeric_columns = ['is_holiday', 'air_pollution_index', 'humidity',
                   'wind_speed', 'wind_direction', 'visibility_in_miles', 'dew_point',
                   'temperature', 'rain_p_h', 'snow_p_h', 'clouds_all', 'weekday', 'hour',
                   'month_day', 'year', 'month', 'last_1_hour_traffic',
                   'last_2_hour_traffic', 'last_3_hour_traffic']

# Perform OneHotEncoding on categorical columns
from sklearn.preprocessing import OneHotEncoder
ohe_encoder = OneHotEncoder()
x_ohehot = ohe_encoder.fit_transform(data[label_columns])
ohe_features = ohe_encoder.get_feature_names_out()

# Create a dataframe for OneHotEncoded features and concatenate it with the main dataframe
x_ohehot = pd.DataFrame(x_ohehot.toarray(), columns=ohe_features)
data = pd.concat([data[['date_time']], data[['traffic_volume'] + numeric_columns], x_ohehot], axis=1)

# Visualize the distribution of traffic volume
data['traffic_volume'].hist(bins=20)

# Plot traffic trends by metrics
metrics = ['month', 'month_day', 'weekday', 'hour']
fig = plt.figure(figsize=(8, 4 * len(metrics)))
for i, metric in enumerate(metrics):
    ax = fig.add_subplot(len(metrics), 1, i + 1)
    ax.plot(data.groupby(metric)['traffic_volume'].mean(), '-o')
    ax.set_xlabel(metric)
    ax.set_ylabel("Mean Traffic")
    ax.set_title(f"Traffic Trend by {metric}")
plt.tight_layout()
plt.show()

# Define features and target variable
features = numeric_columns + list(ohe_features)
target = ['traffic_volume']
X = data[features]
y = data[target]

# Scale features and target variable
x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)
y_scaler = MinMaxScaler()
y = y_scaler.fit_transform(y).flatten()

# Suppress warnings
warnings.filterwarnings('ignore')

# Train an MLPRegressor
regr = MLPRegressor(random_state=1, max_iter=500).fit(X, y)

# Predict and display results
print("Predictions on first 10 samples:", regr.predict(X[:10]))
print("Actual values for first 10 samples:", y[:10])
