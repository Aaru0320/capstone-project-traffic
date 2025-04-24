# 📊 Smart Traffic Forecasting Using LSTM

A deep learning-powered project that predicts real-time urban traffic parameters like traffic speed, road occupancy, and energy consumption using historical time-series data and deploys an interactive dashboard for visualization and forecasting.

---

## 🧠 Project Description

This project leverages **Long Short-Term Memory (LSTM)** networks to forecast key smart city traffic indicators. The model is designed to learn patterns from historical traffic data to predict values such as:

- Traffic Speed (km/h)
- Road Occupancy (%)
- Energy Consumption (Litres/hour)

Users can upload any structured traffic dataset, choose a target metric, and visualize actual vs predicted results in a fully functional **Streamlit dashboard**.

The project is inspired by the real-world problem highlighted in this [Times of India article](https://timesofindia.indiatimes.com/india/traffic-congestion-costs-four-major-indian-cities-rs-1-5-lakh-crore-a-year/articleshow/63918040.cms?utm_source=chatgpt.com)(2018), which states that **traffic congestion costs Indian cities like Delhi, Mumbai, Bengaluru, and Kolkata over Rs 1.5 lakh crore every year**. Efficient forecasting can help reduce fuel loss, air pollution, and commute delays.

---

## 🔍 Use Cases

- **Smart City Administrators**: Optimize traffic flow and manage congestion hotspots.
- **Traffic Police**: Deploy manpower dynamically based on congestion prediction.
- **Ride-hailing Apps**: Predict peak demand periods and optimize driver allocation.
- **Citizens & Commuters**: Plan travel to avoid high-traffic periods and save time.

---

## 📂 Project Structure

```smart_traffic_forecasting/
├── app.py                    # Streamlit dashboard
├── data/
│   ├── traffic.csv           # Raw traffic data
│   └── traffic_cleaned.csv   # Cleaned dataset
├── models/                   # Trained model & scalers
├── notebooks/                # Step-by-step development notebooks
│   ├── 1_data_cleaning.ipynb
│   ├── 2_feature_engineering.ipynb
│   ├── 3_model_training.ipynb
│   └── 4_model_evaluation.ipynb
├── src/                      # Source code scripts
│   ├── clean_traffic_data.py
│   ├── create_sequences.py
│   ├── train_model.py
│   └── evaluate_model.py
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation


yaml
Copy
Edit```

---

## ⚙️ Installation Instructions

### 🔧 Prerequisites

- Python 3.8 or higher
- pip

### ✅ Step 1: Clone the repository

```bash
git clone https://github.com/yourusername/smart_traffic_forecasting.git
cd smart_traffic_forecasting
✅ Step 2: Create and activate a virtual environment
bash
Copy
Edit
python -m venv venv
# On macOS/Linux
source venv/bin/activate
# On Windows
venv\Scripts\activate
✅ Step 3: Install the dependencies
bash
Copy
Edit
pip install -r requirements.txt
✅ Step 4: Run the Streamlit app
bash
Copy
Edit
streamlit run app.py
```
---

##📈 Features

Upload any custom traffic CSV file

Select target variable to forecast (Speed, Occupancy, Energy)

Live LSTM model training using the selected data

Line chart: Actual vs Predicted values

SHAP-based feature importance analysis

Display of MAE, RMSE, R² Score

Downloadable CSV of predictions

----

##🧪 Tech Stack

Python 3.8+

Streamlit

TensorFlow / Keras

SHAP

scikit-learn

pandas, matplotlib, numpy
