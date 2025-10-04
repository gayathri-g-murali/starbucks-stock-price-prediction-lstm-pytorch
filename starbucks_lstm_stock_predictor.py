import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LSTM Model Class
class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# Streamlit page config
st.set_page_config(
    page_title="☕ StarPredict: Starbucks Stock Crystal Ball 🔮",
    layout="wide",
    page_icon="☕"
)

# Custom header similar to Starbucks site
st.markdown("""
<div style="background-color:#00704A;padding:25px;border-radius:10px">
<h1 style="color:white;text-align:center;font-family:sans-serif;">
☕ StarPredict: Starbucks Stock Crystal Ball 🔮
</h1>
<p style="color:white;text-align:center;font-size:18px;">
Predict Starbucks stock prices with LSTM AI. Select dates and explore trends! 📈💡
</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Upload CSV
uploaded_file = st.file_uploader("📂 Upload your Starbucks stock CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview", df.head())

    # --- Preprocess Dates ---
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])  # drop rows with invalid dates
    df['Date'] = df['Date'].apply(lambda x: x.tz_localize(None) if x.tzinfo else x)  # remove timezone if any
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)

    # Only consider data from 2023 onwards
    df_recent = df[df.index >= pd.Timestamp('2023-01-01')]

    # Scale 'Close'
    scaler = StandardScaler()
    df_recent['Close'] = scaler.fit_transform(df_recent[['Close']])

    # Create sequences
    seq_length = 30
    data = []
    for i in range(len(df_recent) - seq_length):
        data.append(df_recent['Close'].values[i:i+seq_length])
    data = np.array(data).reshape((-1, seq_length, 1))

    X = torch.from_numpy(data[:, :-1, :]).float().to(device)

    # Load trained model
    model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)
    model.load_state_dict(torch.load("starbucks_lstm.pth", map_location=device))
    model.eval()

    # Predict
    with torch.no_grad():
        predictions = model(X).cpu().numpy()
    predictions = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(df_recent['Close'].values[seq_length:].reshape(-1,1))

    # --- Calendar Date Picker ---
    min_date = df_recent.index.min()
    max_date = pd.Timestamp('2040-12-31')  # allow prediction up to 2040

    start_date = st.date_input("Start Date", min_value=min_date.date(), max_value=max_date.date(), value=min_date.date())
    end_date = st.date_input("End Date", min_value=min_date.date(), max_value=max_date.date(), value=max_date.date())

    # Extend predictions if end_date > df_recent
    days_to_predict = (end_date - df_recent.index[-1].date()).days
    if days_to_predict > 0:
        last_seq = data[-1:]  # last available sequence
        future_preds = []
        for _ in range(days_to_predict):
            last_tensor = torch.from_numpy(last_seq[:, 1:, :]).float().to(device)
            with torch.no_grad():
                pred = model(last_tensor).cpu().numpy()
            future_preds.append(pred.flatten()[0])
            # Append new prediction to last_seq
            new_val = np.array(pred).reshape((1,1,1))
            last_seq = np.concatenate([last_seq[:,1:,:], new_val], axis=1)
        future_dates = pd.date_range(df_recent.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)
        future_preds = np.array(future_preds).reshape(-1,1)
        predictions_full = np.vstack([predictions, future_preds])
        dates_full = np.concatenate([df_recent.index[seq_length:], future_dates])
        actual_full = np.vstack([actual_prices, np.full((days_to_predict,1), np.nan)])
    else:
        mask = (df_recent.index[seq_length:] >= pd.Timestamp(start_date)) & (df_recent.index[seq_length:] <= pd.Timestamp(end_date))
        predictions_full = predictions[mask]
        actual_full = actual_prices[mask]
        dates_full = df_recent.index[seq_length:][mask]

    # --- Plotly Chart ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_full,
        y=actual_full.flatten(), 
        mode='lines+markers',
        name='Actual', 
        line=dict(color='#00704A', width=3),
        hovertemplate='Date: %{x}<br>Actual: %{y:.2f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=dates_full,
        y=predictions_full.flatten(), 
        mode='lines+markers',
        name='Predicted', 
        line=dict(color='#FFD700', width=3, dash='dash'),
        hovertemplate='Date: %{x}<br>Predicted: %{y:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="☕ StarPredict: Starbucks Stock Crystal Ball 🔮",
            x=0.5,
            xanchor='center',
            font=dict(size=26, color='#FFFFFF', family='Arial')
        ),
        xaxis_title="Date",
        yaxis_title="Stock Price (USD)",
        legend_title="Legend",
        template="plotly_dark",
        plot_bgcolor="#004635",
        paper_bgcolor="#004635",
        font=dict(color='white', family='Arial'),
        margin=dict(t=120)
    )

    st.plotly_chart(fig, use_container_width=True)
    st.success("✅ Prediction Completed!")
