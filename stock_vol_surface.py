import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

# Add cache to improve performance
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def fetch_option_expirations(ticker_symbol):
    """Fetch and cache option expiration dates"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        return ticker.options
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def fetch_spot_price(ticker_symbol):
    """Fetch and cache current spot price"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        spot_history = ticker.history(period='5d')
        return spot_history['Close'].iloc[-1] if not spot_history.empty else None
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def fetch_option_chain(ticker_symbol, expiration_date):
    """Fetch and cache option chain data for a specific expiration date"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        chain = ticker.option_chain(expiration_date)
        
        # Convert to regular pandas DataFrames and select only needed columns
        calls_df = pd.DataFrame({
            'strike': chain.calls['strike'],
            'bid': chain.calls['bid'],
            'ask': chain.calls['ask'],
            'volume': chain.calls['volume'],
            'openInterest': chain.calls['openInterest']
        })
        
        return calls_df
    except Exception as e:
        return None

def bs_call_price(S, K, T, r, sigma, q=0):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def implied_volatility(price, S, K, T, r, q=0):
    if T <= 0 or price <= 0:
        return np.nan

    def objective_function(sigma):
        return bs_call_price(S, K, T, r, sigma, q) - price

    try:
        implied_vol = brentq(objective_function, 1e-6, 5)
    except (ValueError, RuntimeError):
        implied_vol = np.nan

    return implied_vol

# Add app state management
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()
if 'previous_data' not in st.session_state:
    st.session_state.previous_data = None

st.title('Advanced Implied Volatility Surface Analysis')

# Enhanced sidebar with collapsible sections
with st.sidebar:
    with st.expander("Model Parameters", expanded=True):
        risk_free_rate = st.number_input(
            'Risk-Free Rate (e.g., 0.015 for 1.5%)',
            value=0.015,
            format="%.4f"
        )
        
        dividend_yield = st.number_input(
            'Dividend Yield (e.g., 0.013 for 1.3%)',
            value=0.013,
            format="%.4f"
        )

    with st.expander("Visualization Settings", expanded=True):
        y_axis_option = st.selectbox(
            'Select Y-axis:',
            ('Strike Price ($)', 'Moneyness')
        )
        
        # Set fixed values for color scheme and plot type
        color_scheme = "RdYlBu"  # Fixed to RdYlBu
        plot_type = "3D Surface"  # Fixed to 3D Surface

    with st.expander("Ticker Settings", expanded=True):
        ticker_symbol = st.text_input(
            'Enter Ticker Symbol',
            value='SPY',
            max_chars=10
        ).upper()

        auto_refresh = st.checkbox('Auto-refresh data (5 min)', value=False)
        
        # Add price alert feature
        enable_alerts = st.checkbox('Enable Price Alerts', value=False)
        if enable_alerts:
            alert_threshold = st.number_input(
                'Alert Threshold (%)',
                min_value=1.0,
                max_value=20.0,
                value=5.0
            )

    with st.expander("Strike Price Filters", expanded=True):
        min_strike_pct = st.number_input(
            'Minimum Strike Price (%)',
            min_value=50.0,
            max_value=199.0,
            value=80.0,
            step=1.0,
            format="%.1f"
        )

        max_strike_pct = st.number_input(
            'Maximum Strike Price (%)',
            min_value=51.0,
            max_value=200.0,
            value=120.0,
            step=1.0,
            format="%.1f"
        )

# Input validation
if min_strike_pct >= max_strike_pct:
    st.sidebar.error('Minimum percentage must be less than maximum percentage.')
    st.stop()

# Auto-refresh logic
if auto_refresh and time.time() - st.session_state.last_update > 300:  # 5 minutes
    st.session_state.last_update = time.time()
    st.experimental_rerun()

try:
    with st.spinner('Fetching market data...'):
        # Fetch spot price
        spot_price = fetch_spot_price(ticker_symbol)
        if spot_price is None:
            st.error(f'Unable to fetch spot price for {ticker_symbol}')
            st.stop()
            
        # Fetch option expirations
        expirations = fetch_option_expirations(ticker_symbol)
        if expirations is None:
            st.error(f'Unable to fetch option expirations for {ticker_symbol}')
            st.stop()
            
        # Display current market data
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"${spot_price:.2f}")
        with col2:
            st.metric("Risk-Free Rate", f"{risk_free_rate:.2%}")
        with col3:
            st.metric("Dividend Yield", f"{dividend_yield:.2%}")

        today = pd.Timestamp('today').normalize()

except Exception as e:
    st.error(f'Error fetching data for {ticker_symbol}: {str(e)}')
    st.stop()

if not expirations:
    st.error(f'No available option expiration dates for {ticker_symbol}.')
    st.stop()

# Filter by expiration (up to 1 year)
exp_dates = []
for exp in expirations:
    time_to_exp = (pd.Timestamp(exp) - today).days / 365.0
    if time_to_exp <= 1.0:
        exp_dates.append(pd.Timestamp(exp))

if not exp_dates:
    st.error(f'No option expiration dates within 1 year for {ticker_symbol}.')
    st.stop()

# Process option data with progress bar
option_data = []
progress_bar = st.progress(0)
total_expirations = len(expirations)

for idx, exp_date in enumerate(expirations):
    exp_date = pd.Timestamp(exp_date)
    if exp_date <= today + timedelta(days=7):
        continue
        
    try:
        calls = fetch_option_chain(ticker_symbol, exp_date.strftime('%Y-%m-%d'))
        if calls is None or calls.empty:
            continue
        calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
        
        for _, row in calls.iterrows():
            option_data.append({
                'expirationDate': exp_date,
                'strike': row['strike'],
                'bid': row['bid'],
                'ask': row['ask'],
                'mid': (row['bid'] + row['ask']) / 2,
                'volume': row['volume'],
                'openInterest': row['openInterest']
            })
            
    except Exception as e:
        st.warning(f'Failed to fetch option chain for {exp_date.date()}: {str(e)}')
        continue
        
    progress_bar.progress((idx + 1) / total_expirations)

progress_bar.empty()

if not option_data:
    st.error('No valid option data available after filtering.')
    st.stop()

# Process and visualize the data
options_df = pd.DataFrame(option_data)
options_df['daysToExpiration'] = (options_df['expirationDate'] - today).dt.days
options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365

# Filter for options within 1 year
options_df = options_df[options_df['timeToExpiration'] <= 1.0]

if options_df.empty:
    st.error('No data available after applying time filter.')
    st.stop()

# Apply strike price filters
options_df = options_df[
    (options_df['strike'] >= spot_price * (min_strike_pct / 100)) &
    (options_df['strike'] <= spot_price * (max_strike_pct / 100))
]

if options_df.empty:
    st.error('No data available after applying filters.')
    st.stop()

# Calculate implied volatility
with st.spinner('Calculating implied volatilities...'):
    options_df['impliedVolatility'] = options_df.apply(
        lambda row: implied_volatility(
            price=row['mid'],
            S=spot_price,
            K=row['strike'],
            T=row['timeToExpiration'],
            r=risk_free_rate,
            q=dividend_yield
        ), axis=1
    )

options_df.dropna(subset=['impliedVolatility'], inplace=True)
options_df['impliedVolatility'] *= 100
options_df['moneyness'] = options_df['strike'] / spot_price

# Prepare visualization data
Y = options_df['strike'].values if y_axis_option == 'Strike Price ($)' else options_df['moneyness'].values
X = options_df['timeToExpiration'].values
Z = options_df['impliedVolatility'].values

# Create interpolation grid
ti = np.linspace(X.min(), X.max(), 50)
ki = np.linspace(Y.min(), Y.max(), 50)
T, K = np.meshgrid(ti, ki)
Zi = griddata((X, Y), Z, (T, K), method='linear')
Zi = np.ma.array(Zi, mask=np.isnan(Zi))

# Create 3D surface visualization with RdYlBu color scheme
surface_fig = go.Figure(data=[go.Surface(
    x=T, y=K, z=Zi,
    colorscale='rdylbu',  # Fixed to rdylbu (lowercase for plotly)
    colorbar_title='Implied Volatility (%)'
)])

surface_fig.update_layout(
    title=f'Implied Volatility Surface for {ticker_symbol}',
    scene=dict(
        xaxis_title='Time to Expiration (years)',
        yaxis_title=y_axis_option,
        zaxis_title='Implied Volatility (%)'
    ),
    width=900,
    height=800,
    margin=dict(l=65, r=50, b=65, t=90)
)

st.plotly_chart(surface_fig)

# Add summary statistics
with st.expander("Summary Statistics", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average IV", f"{Z.mean():.2f}%")
    with col2:
        st.metric("Min IV", f"{Z.min():.2f}%")
    with col3:
        st.metric("Max IV", f"{Z.max():.2f}%")

# Price alert check
if enable_alerts and st.session_state.previous_data is not None:
    price_change = abs(spot_price - st.session_state.previous_data) / st.session_state.previous_data * 100
    if price_change > alert_threshold:
        st.warning(f'Price Alert: {ticker_symbol} has moved {price_change:.1f}% from previous value!')

# Update previous price
st.session_state.previous_data = spot_price

# Footer
st.write("---")
st.markdown(
    """
    Created by Alejandro Morales |  [LinkedIn](//www.linkedin.com/in/alejandro-morales-e/)  
    """
)
