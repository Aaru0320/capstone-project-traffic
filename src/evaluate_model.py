import pickle
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from create_sequences import create_sequences
import matplotlib.pyplot as plt

def evaluate_model():
    model = load_model('models/traffic_lstm_model.h5')
    scaler_X = pickle.load(open('models/traffic_lstm_scaler_X.pkl', 'rb'))
    scaler_y = pickle.load(open('models/traffic_lstm_scaler_y.pkl', 'rb'))

    df = pd.read_csv('data/traffic_cleaned.csv')
    features = ['Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%']
    target = 'Traffic_Speed_kmh'

    scaled_X = scaler_X.transform(df[features])
    scaled_y = scaler_y.transform(df[[target]])

    seq_length = 10
    X, y = create_sequences(scaled_X, seq_length)
    y_true = y[:, 1]
    y_pred = model.predict(X)
    
    y_true = scaler_y.inverse_transform(y[:, 1].reshape(-1, 1))
    y_pred = scaler_y.inverse_transform(y_pred)

    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.title('Traffic Speed Prediction')
    plt.savefig('models/evaluation_plot.png')

if __name__ == "__main__":
    evaluate_model()