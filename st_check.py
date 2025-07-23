import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import date

st.set_page_config(layout="wide")
st.title(":money_with_wings: Monte Carlo Portfolio Optimizer")

# Hide Streamlit menu and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp { padding: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar for input
st.sidebar.header("Simulation Parameters")
ticker_to_name = {
    # US Tech
    'AAPL': 'Apple Inc.',
    'GOOG': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'MSFT': 'Microsoft Corp.',
    'META': 'Meta Platforms',
    'NVDA': 'NVIDIA Corp.',
    'TSLA': 'Tesla Inc.',
    'ORCL': 'Oracle Corp.',
    'IBM': 'IBM',
    'ADBE': 'Adobe Inc.',
    'INTC': 'Intel Corp.',
    'CRM': 'Salesforce Inc.',

    # US Finance / Consumer
    'JPM': 'JPMorgan Chase',
    'WMT': 'Walmart Inc.',
    'NFLX': 'Netflix Inc.',
    'T': 'AT&T Inc.',
    'KO': 'Coca-Cola Co',
    'PEP': 'PepsiCo Inc.',
    'MCD': 'McDonald\'s Corp',
    'DIS': 'Walt Disney Co',

    # Indian Largecaps
    'RELIANCE.NS': 'Reliance Industries',
    'TCS.NS': 'Tata Consultancy Services',
    'INFY.NS': 'Infosys',
    'HDFCBANK.NS': 'HDFC Bank',
    'ICICIBANK.NS': 'ICICI Bank',
    'SBIN.NS': 'State Bank of India',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'ITC.NS': 'ITC Limited',
    'LT.NS': 'Larsen & Toubro',
    'ASIANPAINT.NS': 'Asian Paints',
    'AXISBANK.NS': 'Axis Bank',
    'SUNPHARMA.NS': 'Sun Pharma',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'MARUTI.NS': 'Maruti Suzuki',
    'BAJFINANCE.NS': 'Bajaj Finance',
    'TATAMOTORS.NS': 'Tata Motors',
    'DMART.NS': 'Avenue Supermarts (DMart)',
    'ULTRACEMCO.NS': 'UltraTech Cement',
    'NTPC.NS': 'NTPC Limited',
    'COALINDIA.NS': 'Coal India',
    'HCLTECH.NS': 'HCL Technologies',
    'WIPRO.NS': 'Wipro Ltd.',
    'DABUR.NS': 'Dabur India',
    'HAVELLS.NS': 'Havells India'
}

all_stocks = list(ticker_to_name.keys())
default_stocks = ['AAPL', 'GOOG', 'RELIANCE.NS', 'HDFCBANK.NS', 'TSLA']
name_to_ticker = {v: k for k, v in ticker_to_name.items()}
selected_names = st.sidebar.multiselect(
    "Select Stocks", 
    options=list(name_to_ticker.keys()), 
    default=[ticker_to_name[t] for t in default_stocks]
)
selected_stocks = [name_to_ticker[name] for name in selected_names]

start_date = st.sidebar.date_input("Start Date", value=date(2023, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date(2024, 1, 1))
rf_rate = st.sidebar.slider("Risk-Free Rate (Annual)", 0.0, 0.2, 0.08)
num_portfolios = st.sidebar.slider("Number of Simulations", 500, 10000, 3000, step=500)

# Fetch and process data
@st.cache_data(show_spinner=True)
def get_data(stocks, start, end):
    data = yf.download(stocks, start=start, end=end)['Close'].dropna()
    returns = data.pct_change().dropna()
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    return mean_returns, cov_matrix

if len(selected_stocks) < 2:
    st.warning("Please select at least two stocks.")
    st.stop()

mean_returns, cov_matrix = get_data(selected_stocks, start_date, end_date)

# Run Monte Carlo simulation
results = []
weights_list = []

for _ in range(num_portfolios):
    weights = np.random.random(len(selected_stocks))
    weights /= np.sum(weights)
    ret = np.dot(weights, mean_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe = (ret - rf_rate) / vol
    results.append((ret, vol, sharpe))
    weights_list.append(weights)

results = np.array(results)
max_sharpe_idx = np.argmax(results[:, 2])
max_sharpe = results[max_sharpe_idx]
optimal_weights = weights_list[max_sharpe_idx]

# Display optimal weights
st.subheader("🌟 Max Sharpe Ratio Portfolio")
sharpe_col, ret_col, vol_col = st.columns(3)
sharpe_col.metric("Sharpe Ratio", f"{max_sharpe[2]:.2f}")
ret_col.metric("Expected Return", f"{max_sharpe[0]*100:.2f}%")
vol_col.metric("Volatility (Risk)", f"{max_sharpe[1]*100:.2f}%")

st.markdown("#### 📊 Optimal Weights")
weights_df = pd.DataFrame({
    'Stock': selected_stocks,
    'Weight (%)': np.round(np.array(optimal_weights) * 100, 2)
})
st.dataframe(weights_df, use_container_width=True)

# Plot efficient frontier
fig = go.Figure()
colors = np.where(results[:, 2] < 0.5, 'red',
                  np.where(results[:, 2] < 1.0, 'orange', 'green'))

fig.add_trace(go.Scatter(
    x=results[:, 1],
    y=results[:, 0],
    mode='markers',
    marker=dict(color=colors, size=4, opacity=0.6),
    name='Portfolios'
))

# Highlight max Sharpe
fig.add_trace(go.Scatter(
    x=[max_sharpe[1]],
    y=[max_sharpe[0]],
    mode='markers+text',
    marker=dict(color='gold', size=12, symbol='star'),
    text=["Max Sharpe"],
    textposition="top center",
    name='Max Sharpe'
))

fig.update_layout(
    title="Efficient Frontier (Return vs Risk)",
    xaxis_title="Volatility (Standard Deviation)",
    yaxis_title="Expected Return",
    height=600,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.caption("Note: Historical data is used. Past performance is not indicative of future returns.")
