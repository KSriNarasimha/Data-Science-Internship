import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the trained Holt-Winters model
model = joblib.load('best_HW_model.pkl')

# Streamlit UI
st.title("📈 Holt-Winters Forecasting")

# Get the last available date from the model data
last_date = model.model.data.dates[-1]

# Create two columns for side-by-side inputs
col1, col2 = st.columns(2)

with col1:
    start_date = st.date_input("Select Start Date:", value=last_date,min_value=pd.to_datetime(last_date))

with col2:
    end_date = st.date_input("Select End Date:", value=pd.to_datetime(last_date) + pd.DateOffset(months=12),min_value=pd.to_datetime(last_date)+ pd.DateOffset(days=1))

# Convert to datetime for calculations
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

# Ensure end date is after start date
if end_date <= start_date:
    st.error("End date must be after the start date. Please select a valid range.")
else:
    # Calculate the number of months between start and end date
    n_periods = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

    # Forecast the next `n_periods` months
    forecast = model.forecast(n_periods)

    # Generate future date range
    forecast_dates = pd.date_range(start=start_date, periods=n_periods, freq='ME')

    # Convert forecast to Pandas DataFrame with formatted short month names
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecasted Close Price': forecast})
    forecast_df['Date'] = forecast_df['Date'].dt.strftime('%b %Y')  # Convert to "Jan 2025" format
    forecast_df.set_index('Date', inplace=True)
    forecast_df = forecast_df.round(2)

st.subheader(f"🔮 Forecast Trend for {n_periods} months")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecast_df.index, forecast_df['Forecasted Close Price'], linestyle="dashed", marker="o", color="green", label="Forecast")
ax.set_xlabel("Date")
ax.set_ylabel("Close Price")
ax.legend()
plt.xticks(rotation=90)
st.pyplot(fig)


st.markdown("<h3 style='color: darkred;'> Forecasted Data</h3>", unsafe_allow_html=True)
st.dataframe(forecast_df)
