# Traffic_Prediction
Traffic Prediction using Deep Learning 

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Application

### Training the Original Model
```
cd mp
python main.py
```

### Running the Original Web Interface
```
cd mp
python app.py
```

The web application will be available at http://127.0.0.1:5000/

### Bangalore Traffic Prediction Model

This project includes a dedicated model for Bangalore traffic prediction using machine learning.

#### Training the Bangalore Model
```
cd mp
python train_bangalore_model.py
```

#### Making Predictions with the Bangalore Model
```
cd mp
python predict_bangalore_traffic.py
```

#### Running the Bangalore Traffic Web Application
```
cd mp
python bangalore_traffic_app.py
```

The Bangalore traffic prediction web application will be available at http://127.0.0.1:5050/

## Project Structure
- `mp/` - Main project directory
  - `main.py` - Original model training script
  - `app.py` - Original Flask web application
  - `test.py` - Testing script
  - `train_bangalore_model.py` - Script to train the Bangalore traffic model
  - `predict_bangalore_traffic.py` - Script to make predictions with the Bangalore model
  - `bangalore_traffic_app.py` - Web application for Bangalore traffic prediction
  - `models/` - Directory for saved trained models
  - `templates/` - HTML templates for the web interfaces
  - `static/` - Static files (CSS, images, data files)
  - `Banglore_traffic_Dataset.csv` - Dataset for Bangalore traffic prediction 
