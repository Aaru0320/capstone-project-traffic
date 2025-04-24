import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError
from src.create_sequences import create_sequences
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from io import StringIO
from datetime import datetime

st.set_page_config(page_title="Smart Traffic Forecasting", layout="wide")
st.title("🚦 Smart Traffic Forecasting Dashboard")

@st.cache_data
def load_clean_data():
    return pd.read_csv("data/traffic_cleaned.csv")

model_path = "models/traffic_lstm_model.h5"

uploaded_file = st.sidebar.file_uploader("Upload Traffic Data (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ Custom data loaded")
else:
    df = load_clean_data()

# Filter numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("❌ Not enough numeric columns for prediction. Minimum 2 required.")
    st.stop()

# Allow user to select target variable
target_metric = st.sidebar.selectbox("Select Target Variable", numeric_cols)
features = [col for col in numeric_cols if col != target_metric]

n_points = st.sidebar.slider("Number of Predictions to Display", min_value=50, max_value=1000, value=200)

try:
    from sklearn.preprocessing import MinMaxScaler
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    scaled_X = scaler_X.fit_transform(df[features])
    scaled_y = scaler_y.fit_transform(df[[target_metric]])

    X, y = create_sequences(scaled_X, 10)
    y_true = scaled_y[10:]

    model = Sequential([
        LSTM(64, input_shape=(X.shape[1], X.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    history = model.fit(X, y_true, epochs=5, batch_size=32, verbose=0)

    y_pred = model.predict(X)

    y_true_inv = scaler_y.inverse_transform(y_true)
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    st.subheader(f"📈 Actual vs Predicted: {target_metric}")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y_true_inv[:n_points], label='Actual')
    ax.plot(y_pred_inv[:n_points], label='Predicted')
    ax.set_xlabel("Time Steps")
    ax.set_ylabel(target_metric)
    ax.set_title("Traffic Prediction")
    ax.legend()
    st.pyplot(fig)

    st.subheader("📊 Model Evaluation Metrics")
    mae = mean_absolute_error(y_true_inv, y_pred_inv)
    mse = mean_squared_error(y_true_inv, y_pred_inv)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_inv, y_pred_inv)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")
    col4.metric("R² Score", f"{r2:.2f}")
    
    # ----------- Actual vs Predicted Scatter Plot -----------
    with st.expander("📈 Actual vs Predicted Scatter Plot"):
        import matplotlib.pyplot as plt

        actual = y_true_inv.flatten()
        predicted = y_pred_inv.flatten()

        plt.figure(figsize=(8, 6))
        plt.scatter(actual, predicted, color='blue', label='Predictions', alpha=0.6)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r-', linewidth=2, label='Perfect Fit')
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title(f"Actual vs Predicted - {target_metric}")
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)


    # ----------- TRAINING LOSS CURVE -----------
    with st.expander("📉 Training Loss Curve"):
        plt.figure(figsize=(8, 3))
        plt.plot(history.history['loss'], label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        st.pyplot(plt)

    # ----------- DATA PREVIEW -----------
    with st.expander("📂 Show Data Preview"):
        st.dataframe(df.head(10))

    # ----------- DOWNLOAD -----------
    with st.expander("📤 Download Predictions"):
        pred_df = pd.DataFrame({
            "Actual": y_true_inv.flatten(),
            "Predicted": y_pred_inv.flatten()
        })
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download as CSV", csv, "predictions.csv", "text/csv")

except Exception as e:
    st.error(f"❌ Error during prediction: {e}")

# ----------- INFO -----------
with st.expander("ℹ️ Model Info"):
    st.markdown(f"""
    - Model: LSTM (trained live)
    - Features used: {features}
    - Target: {target_metric}
    - Loss Function: MeanSquaredError
    - Framework: TensorFlow & Streamlit
    """)

with st.expander("📘 Glossary & Help"):
    st.markdown("""
    **LSTM** (Long Short-Term Memory) is a type of RNN used for time-series prediction.
    
    **MAE (Mean Absolute Error):** Average of absolute errors.
    
    **MSE (Mean Squared Error):** Average of squared errors.

    **RMSE:** Square root of MSE. Penalizes larger errors.

    **R² Score:** Measures how well predictions approximate actual values. Closer to 1 is better.
    
    This dashboard forecasts traffic metrics from any dataset containing at least 2 numeric columns.
    """)
