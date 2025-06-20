import streamlit as st
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model



#  Title 
st.title("Stockers - Stock Info & Prediction")

if "watchlist" not in st.session_state:
    st.session_state.watchlist = []

# User Input 
ticker = st.text_input("Enter Stock Ticker (e.g., INFY, HAL, TCS):", "HAL")

if st.button("➕ Add to Watchlist"):
    if ticker not in st.session_state.watchlist:
        st.session_state.watchlist.append(ticker)
        st.success(f"{ticker} added to watchlist.")
    else:
        st.info(f"{ticker} is already in your watchlist.")

#  Load Your LSTM Model 
model = load_model("TTM_prediction.h5")
  # or use load_model('model.h5') if that's your format

# Fetch Stock Info from Screener
def fetch_company_info(ticker):
    url = f'https://www.screener.in/company/{ticker}/'
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')

        # Show Current Price
        try:
            price = soup.find(class_="flex flex-align-center").text.split()
            price_value = price[1]
            percent_change = price[2]
            color = "green" if "-" not in percent_change else "red"
            st.markdown(
                f"<h3>₹ {price_value} <span style='color:{color}; font-size:18px;'> {percent_change}</span></h3>",
                unsafe_allow_html=True
            )
            class3 = "h2 shrink-text"
            soup.find(class_=class3).text
        except:
            st.warning("Could not fetch price.")

        # Show Ratios
        st.subheader("Key Financial Ratios:")
        ratios_list = soup.select("ul#top-ratios li")
        for item in ratios_list:
            name = item.find("span", class_="name")
            value = item.find("span", class_="nowrap value")
            if name and value:
                st.write(f"**{name.text.strip()}**: {value.text.strip()}")

        # Show Peer Comparison using pandas.read_html
        st.header("Peer Comparison")
        try:
            tables = pd.read_html(f"https://www.screener.in/company/{ticker}/consolidated/")
            peer_table = next(t for t in tables if "CMP Rs." in t.columns)  # find the right table
            st.dataframe(peer_table)
        except Exception as e:
            st.warning(f"Could not fetch peer comparison data. Error: {e}")

        
        #show Peer Comparison 
        st.header("Quaterly Result")
        try:
            table = soup.find('table', class_='data-table')  # Peer comparison table
            headers = [th.text.strip() for th in table.find_all('th')]

            rows = []
            for tr in table.find_all('tr')[1:]:  # Skip header
                cols = [td.text.strip() for td in tr.find_all('td')]
                if cols:
                    rows.append(cols)

            df_peers = pd.DataFrame(rows, columns=headers)
            st.dataframe(df_peers)
        except:
            st.warning("Could not fetch Peer Comparison table.")
    else:
        st.error("Failed to fetch data from Screener.in")

# Fetch Historical Data 
def fetch_stock_history(ticker):
    ALPHA_VANTAGE_API_KEY = "API_KEY"
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}.BSE&outputsize=compact&apikey={ALPHA_VANTAGE_API_KEY}'
    r = requests.get(url)
    data = r.json()

    try:
        df = pd.DataFrame(data['Time Series (Daily)']).T
        df = df.astype(float)
        df = df[['4. close']]
        df.rename(columns={'4. close': 'Close'}, inplace=True)
        df = df[::-1]  # oldest to newest
        return df
    except:
        st.error("Error fetching stock history.")
        return None

st.sidebar.header("📈 Your Watchlist")
if st.session_state.watchlist:
    for stock in st.session_state.watchlist:
        st.sidebar.write(f"🔹 {stock}")
else:
    st.sidebar.write("No stocks added yet.")

#  Predict with LSTM 
def predict_prices(df, model):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(df[['Close']])

    last_100_days = scaled_data[-100:]  # reshape for prediction
    X_input = np.reshape(last_100_days, (1, 100, 1))
    
    prediction = model.predict(X_input)
    prediction_price = scaler.inverse_transform(prediction)
    return prediction_price[0][0]

#  Main Logic
if ticker:
    fetch_company_info(ticker)

    df = fetch_stock_history(ticker)
    if df is not None:
        st.line_chart(df['Close'], use_container_width=True)
        predicted_price = predict_prices(df, model)
        st.success(f" Predicted Next Closing Price: ₹ {predicted_price:.2f}")
