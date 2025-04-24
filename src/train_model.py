import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from create_sequences import create_sequences

def train_model():
    df = pd.read_csv('data/traffic_cleaned.csv')

    features = ['Vehicle_Count', 'Traffic_Speed_kmh', 'Road_Occupancy_%']
    target = 'Traffic_Speed_kmh'

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaled_X = scaler_X.fit_transform(df[features])
    scaled_y = scaler_y.fit_transform(df[[target]])

    seq_length = 10
    X, y = create_sequences(scaled_X, seq_length)
    y = y[:, 1]  # Predicting traffic speed

    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss=MeanSquaredError())
    model.fit(X, y, epochs=10, batch_size=32)

    model.save('models/traffic_lstm_model.h5')
    pickle.dump(scaler_X, open('models/traffic_lstm_scaler_X.pkl', 'wb'))
    pickle.dump(scaler_y, open('models/traffic_lstm_scaler_y.pkl', 'wb'))

if __name__ == "__main__":
    train_model()
