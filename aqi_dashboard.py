import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# -------------------------------
# PAGE TITLE
# -------------------------------

st.set_page_config(page_title="AQI Dashboard", layout="wide")

st.title("Air Quality Index (AQI) Analysis & Forecasting Dashboard")

# -------------------------------
# SIDEBAR INPUT
# -------------------------------

st.sidebar.header("Enter AQI Data")

date = st.sidebar.text_input("Date (YYYY-MM-DD)")
city = st.sidebar.text_input("City")

pm25 = st.sidebar.number_input("PM2.5", min_value=0.0)
pm10 = st.sidebar.number_input("PM10", min_value=0.0)
no2 = st.sidebar.number_input("NO2", min_value=0.0)
so2 = st.sidebar.number_input("SO2", min_value=0.0)

aqi = st.sidebar.number_input("AQI", min_value=0.0)

# -------------------------------
# STORE DATA
# -------------------------------

if "data" not in st.session_state:
    st.session_state.data = []

if st.sidebar.button("Add Data"):
    
    st.session_state.data.append([
        date, city, pm25, pm10, no2, so2, aqi
    ])

# -------------------------------
# CREATE DATAFRAME
# -------------------------------

df = pd.DataFrame(
    st.session_state.data,
    columns=["Date","City","PM2.5","PM10","NO2","SO2","AQI"]
)

# -------------------------------
# DASHBOARD OUTPUT
# -------------------------------

if not df.empty:

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    st.subheader("AQI Dataset")

    st.dataframe(df, use_container_width=True)

    # -------------------------------
    # AQI TREND
    # -------------------------------

    st.subheader("AQI Trend Over Time")

    fig1, ax1 = plt.subplots()

    ax1.plot(df["Date"], df["AQI"], marker="o")

    ax1.set_xlabel("Date")
    ax1.set_ylabel("AQI")

    ax1.grid(True)

    st.pyplot(fig1)

    # -------------------------------
    # POLLUTANT DISTRIBUTION
    # -------------------------------

    st.subheader("Pollutant Distribution")

    fig2, ax2 = plt.subplots()

    sns.boxplot(data=df[["PM2.5","PM10","NO2","SO2"]], ax=ax2)

    st.pyplot(fig2)

    # -------------------------------
    # ARIMA FORECAST
    # -------------------------------

    st.subheader("AQI Forecast using ARIMA")

    series = df.set_index("Date")["AQI"]

    if len(series) > 3:

        model = ARIMA(series, order=(1,1,1))

        model_fit = model.fit()

        forecast = model_fit.forecast(steps=7)

        fig3, ax3 = plt.subplots()

        ax3.plot(series, label="Original AQI")

        ax3.plot(forecast, label="Forecast AQI")

        ax3.legend()

        st.pyplot(fig3)

    else:

        st.warning("Add more data for ARIMA forecasting")

    # -------------------------------
    # PROPHET FORECAST
    # -------------------------------

    st.subheader("AQI Forecast using Prophet")

    prophet_df = df[["Date","AQI"]]

    prophet_df.columns = ["ds","y"]

    if len(prophet_df) > 3:

        model = Prophet()

        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=14)

        forecast = model.predict(future)

        fig4 = model.plot(forecast)

        st.pyplot(fig4)

    else:

        st.warning("Add more data for Prophet forecasting")

    # -------------------------------
    # HIGH POLLUTION DAYS
    # -------------------------------

    st.subheader("High Pollution Days (AQI > 200)")

    high_pollution = df[df["AQI"] > 200]

    if not high_pollution.empty:

        st.dataframe(high_pollution)

    else:

        st.success("No severe pollution days found")

else:

    st.info("Enter AQI data from the sidebar to start analysis")