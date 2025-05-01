"""
Interactive Options Pricing Dashboard using Streamlit.

This module provides an interactive web interface for exploring 
option pricing models, visualizing Greeks, and analyzing option strategies.
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pricing.black_scholes import black_scholes_price, implied_volatility
from pricing.binomial_tree import binomial_tree_price, trinomial_tree_price
from pricing.greeks import calculate_all_greeks
from app.visuals import (plot_option_price_vs_underlying, plot_option_price_surface,
                        plot_greek_surface, plot_greeks_vs_underlying,
                        plot_option_payoff, plot_strategy_payoff)

# Define custom color scheme
CREAM = "#fbf6f3"
TANGERINE = "#feb06a"
TURQUOISE = "#36d6e7"
BLUE_GRAY = "#5d6c89"

# Define dark mode colors
DARK_BG = "#1e1e1e"
DARK_TEXT = "#f0f0f0"
DARK_SURFACE = "#2c2c2c"

# Set page configuration
st.set_page_config(
    page_title="Options Pricing Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS for Streamlit components
st.markdown(
    f"""
    <style>
    /* Main app background */
    .stApp {{
        background-color: {CREAM};
    }}
    
    /* Container styling */
    .css-1d391kg, .css-12oz5g7 {{
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }}
    
    /* Sidebar */
    [data-testid="stSidebar"] {{
        background-color: {BLUE_GRAY};
        padding-top: 2rem;
    }}
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{
        color: white;
    }}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {{
        color: {BLUE_GRAY} !important;
        font-weight: 600 !important;
    }}
    
    /* Buttons */
    .stButton > button {{
        background-color: {TURQUOISE};
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }}
    
    .stButton > button:hover {{
        background-color: {BLUE_GRAY};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: {CREAM};
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        color: {BLUE_GRAY};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {TURQUOISE};
        color: white !important;
    }}
    
    /* Metric text */
    [data-testid="stMetricValue"] {{
        font-size: 2rem !important;
        font-weight: 600 !important;
        color: {BLUE_GRAY} !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        font-size: 1rem !important;
        font-weight: 400 !important;
        color: {BLUE_GRAY} !important;
    }}
    
    /* Tables */
    .dataframe {{
        font-size: 0.9rem !important;
    }}
    
    thead tr th {{
        background-color: {BLUE_GRAY};
        color: white !important;
    }}
    
    tbody tr:nth-child(even) {{
        background-color: {CREAM};
    }}
    
    /* Sliders */
    [data-testid="stThumbValue"] {{
        color: {BLUE_GRAY} !important;
    }}
    
    .stSlider [data-baseweb="slider"] div div div {{
        background-color: {TANGERINE} !important;
    }}
    
    /* Text inputs */
    [data-baseweb="input"] {{
        border-color: {TURQUOISE};
    }}
    
    /* Labels */
    label, .label {{
        color: {BLUE_GRAY} !important;
        font-weight: 500 !important;
    }}
    
    /* Radio buttons and checkboxes */
    .stRadio div[role="radiogroup"] div label,
    .stCheckbox label {{
        color: {BLUE_GRAY} !important;
    }}
    
    /* Make forms look nicer */
    div[data-testid="stForm"] {{
        border: 1px solid {TURQUOISE};
        padding: 1em;
        border-radius: 10px;
        margin: 1em 0;
    }}
    
    /* Expander */
    .streamlit-expanderHeader {{
        background-color: {CREAM};
        color: {BLUE_GRAY} !important;
        font-weight: 600 !important;
        border-radius: 5px;
    }}
    
    /* Ensure math formulas are visible in both light and dark modes */
    .katex {{ 
        font-size: 1.1em;
    }}
    
        /* Dark mode support */
    @media (prefers-color-scheme: dark) {{
        .stApp {{
            background-color: {DARK_BG};
            color: {DARK_TEXT};
        }}
        
        .css-1d391kg, .css-12oz5g7 {{
            background-color: {DARK_SURFACE};
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }}
        
        h1, h2, h3, h4, h5, h6 {{
            color: {DARK_TEXT} !important;
        }}
        
        [data-testid="stMetricValue"] {{
            color: {DARK_TEXT} !important;
        }}
        
        [data-testid="stMetricLabel"] {{
            color: {DARK_TEXT} !important;
        }}
        
        label, .label {{
            color: {DARK_TEXT} !important;
        }}
        
        .stRadio div[role="radiogroup"] div label,
        .stCheckbox label {{
            color: {DARK_TEXT} !important;
        }}
        
        .streamlit-expanderHeader {{
            background-color: {DARK_SURFACE};
            color: {DARK_TEXT} !important;
        }}
        
        tbody tr:nth-child(even) {{
            background-color: #2a2a2a;
        }}
        
        [data-testid="stThumbValue"] {{
            color: {DARK_TEXT} !important;
        }}
        
        /* Math formulas in dark mode */
        .katex {{
            color: {DARK_TEXT} !important;
        }}
    }}
    
    </style>
    """, 
    unsafe_allow_html=True
)

# Set theme for matplotlib
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[BLUE_GRAY, TURQUOISE, TANGERINE, CREAM])
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = BLUE_GRAY
plt.rcParams['axes.labelcolor'] = BLUE_GRAY
plt.rcParams['xtick.color'] = BLUE_GRAY
plt.rcParams['ytick.color'] = BLUE_GRAY
plt.rcParams['grid.color'] = CREAM
plt.rcParams['grid.alpha'] = 0.4

def main():
    """Main function to run the Streamlit app."""
    
    # Add header and description
    st.title("🧮 Options Pricing Simulator")
    
    with st.container():
        st.markdown("""
        Explore option pricing models, visualize Greeks, and analyze option strategies with this interactive tool.
        
        #### Key Features:
        * Black-Scholes and Binomial Tree pricing models
        * Options Greeks calculation and visualization
        * Payoff diagrams for option strategies
        * Implied volatility calculation
        """)
    
    # Sidebar for navigation
    with st.sidebar:
        st.markdown("### Navigation")
        app_mode = st.radio(
            "",
            ["Option Pricing Models", "Option Greeks", "Option Strategies", "About"],
            label_visibility="collapsed"
        )
        
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Options Pricing Simulator")
        st.sidebar.markdown("A quantitative finance tool for exploring option pricing models and strategies.")
    
    # Different modes
    if app_mode == "Option Pricing Models":
        option_pricing_page()
    elif app_mode == "Option Greeks":
        option_greeks_page()
    elif app_mode == "Option Strategies":
        option_strategies_page()
    else:  # About
        about_page()

def option_pricing_page():
    """Page for exploring option pricing models."""
    st.header("Option Pricing Models")
    
    with st.container():
        st.markdown("""
        Compare different option pricing models:
        * Black-Scholes: Analytical model for European options
        * Binomial Tree: Numerical model for both European and American options
        * Trinomial Tree: Enhanced numerical model with higher accuracy
        """)
    
    # Create columns for input parameters
    with st.container():
        st.markdown("### Model Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            S = st.number_input("Underlying Price (S)", min_value=1.0, value=100.0, step=1.0)
            K = st.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
        
        with col2:
            # Set minimum value to 0.01 to avoid T=0 issues
            T = st.slider("Time to Maturity (T) in years", min_value=0.01, max_value=3.0, value=1.0, step=0.1)
            r = st.slider("Risk-free Rate (r)", min_value=0.0, max_value=0.20, value=0.05, step=0.01)
        
        with col3:
            sigma = st.slider("Volatility (σ)", min_value=0.01, max_value=1.00, value=0.20, step=0.01)
            option_type = st.selectbox("Option Type", ["call", "put"])
    
    # Additional parameters for numerical methods
    with st.expander("Advanced Settings"):
        style_col, steps_col = st.columns(2)
        
        with style_col:
            option_style = st.selectbox("Option Style", ["european", "american"])
        
        with steps_col:
            n_steps = st.slider("Tree Steps (N)", min_value=10, max_value=1000, value=100, step=10)
    
    # Calculate option prices
    bs_price = black_scholes_price(S, K, T, r, sigma, option_type)
    binomial_price = binomial_tree_price(S, K, T, r, sigma, option_type, option_style, n_steps)
    trinomial_price = trinomial_tree_price(S, K, T, r, sigma, option_type, option_style, n_steps//2)
    
    # Display results in a card
    st.markdown("### Option Prices")
    
    price_cols = st.columns(3)
    with price_cols[0]:
        st.metric("Black-Scholes", f"${bs_price:.4f}", "")
    with price_cols[1]:
        st.metric("Binomial Tree", f"${binomial_price:.4f}", f"{binomial_price - bs_price:.4f}")
    with price_cols[2]:
        st.metric("Trinomial Tree", f"${trinomial_price:.4f}", f"{trinomial_price - bs_price:.4f}")
    
    # Price comparison table
    price_data = pd.DataFrame({
        "Model": ["Black-Scholes", "Binomial Tree", "Trinomial Tree"],
        "Price": [bs_price, binomial_price, trinomial_price],
        "Difference from BS": [0, binomial_price - bs_price, trinomial_price - bs_price]
    })
    
    st.table(price_data.style.format({
        "Price": "${:.4f}",
        "Difference from BS": "${:.6f}"
    }))
    
    # Visualizations
    st.markdown("### Visualizations")
    
    viz_tabs = st.tabs(["Price vs Underlying", "Price vs Time", "Price Surface"])
    
    with viz_tabs[0]:
        # Option price versus underlying price
        st.markdown("#### Option Price vs Underlying Price")
        
        # Create price range centered around current price
        S_range = np.linspace(max(1, S * 0.5), S * 1.5, 100)
        
        fig = plot_option_price_vs_underlying(S_range, K, T, r, sigma, option_type)
        st.pyplot(fig)
    
    with viz_tabs[1]:
        # Option price versus time to maturity
        st.markdown("#### Option Price vs Time to Maturity")
        
        # Time range from small positive value to slightly beyond maturity
        # Use 0.001 as minimum to avoid the T=0 case
        T_range = np.linspace(0.001, T * 1.5, 100)
        
        # Use try-except to handle any potential calculation errors
        try:
            prices_over_time = [black_scholes_price(S, K, t, r, sigma, option_type) for t in T_range]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(T_range, prices_over_time, '-', linewidth=2, color=BLUE_GRAY)
            
            # Add vertical line at the current time to maturity
            ax.axvline(x=T, color=BLUE_GRAY, linestyle='--')
            
            ax.set_title(f'{option_type.capitalize()} Option Price vs Time to Maturity', fontsize=14)
            ax.set_xlabel('Time to Maturity (years)', fontsize=12)
            ax.set_ylabel('Option Price ($)', fontsize=12)
            ax.grid(True, color=BLUE_GRAY, alpha=0.3)
            
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error calculating option prices: {e}")
    
    with viz_tabs[2]:
        # 3D surface plot of option price
        st.markdown("#### Option Price Surface")
        st.write("3D surface showing option price as a function of underlying price and time to maturity")
        
        S_range = np.linspace(max(1, S * 0.5), S * 1.5, 20)
        T_range = np.linspace(0.01, T * 1.5, 20)  # Avoid T=0
        
        fig = plot_option_price_surface(S_range, T_range, K, r, sigma, option_type)
        st.pyplot(fig)
    
    # Comparison of models
    st.markdown("### Model Comparison")
    
    # Compare prices as function of steps
    if st.checkbox("Compare convergence of tree models"):
        step_range = np.array([5, 10, 20, 50, 100, 200, 500, 1000])
        binomial_prices = [binomial_tree_price(S, K, T, r, sigma, option_type, option_style, n) 
                          for n in step_range]
        trinomial_prices = [trinomial_tree_price(S, K, T, r, sigma, option_type, option_style, n//2) 
                           for n in step_range]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(step_range, binomial_prices, 'o-', label='Binomial Tree', color=BLUE_GRAY)
        ax.plot(step_range, trinomial_prices, 's-', label='Trinomial Tree', color=BLUE_GRAY)
        ax.axhline(y=bs_price, color=BLUE_GRAY, linestyle='-', label='Black-Scholes')
        
        ax.set_xscale('log')
        ax.set_title('Convergence of Tree Models to Black-Scholes Price', fontsize=14)
        ax.set_xlabel('Number of Steps (N)', fontsize=12)
        ax.set_ylabel('Option Price ($)', fontsize=12)
        ax.grid(True, color=BLUE_GRAY, alpha=0.3)
        ax.legend()
        
        st.pyplot(fig)
        
def option_greeks_page():
    """Page for exploring option Greeks."""
    st.header("Option Greeks")
    
    st.markdown("""
    Explore the sensitivities of option prices to various parameters:
    * **Delta**: Sensitivity to underlying price (∂V/∂S)
    * **Gamma**: Rate of change of Delta (∂²V/∂S²)
    * **Theta**: Sensitivity to time decay (∂V/∂T)
    * **Vega**: Sensitivity to volatility (∂V/∂σ)
    * **Rho**: Sensitivity to interest rate (∂V/∂r)
    """)
    
    # Create columns for input parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        S = st.number_input("Underlying Price (S)", min_value=1.0, value=100.0, step=1.0)
        K = st.number_input("Strike Price (K)", min_value=1.0, value=100.0, step=1.0)
    
    with col2:
        T = st.slider("Time to Maturity (T) in years", min_value=0.01, max_value=3.0, value=1.0, step=0.01)
        r = st.slider("Risk-free Rate (r)", min_value=0.0, max_value=0.20, value=0.05, step=0.01)
    
    with col3:
        sigma = st.slider("Volatility (σ)", min_value=0.01, max_value=1.00, value=0.20, step=0.01)
        option_type = st.selectbox("Option Type", ["call", "put"])
    
    # Calculate option price and Greeks
    price = black_scholes_price(S, K, T, r, sigma, option_type)
    greeks = calculate_all_greeks(S, K, T, r, sigma, option_type)
    
    # Display results in a table
    st.subheader("Option Price and Greeks")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.metric("Price", f"${price:.4f}")
    
    with col2:
        greek_cols = st.columns(5)
        for i, (greek, value) in enumerate(greeks.items()):
            with greek_cols[i]:
                st.metric(greek.capitalize(), f"{value:.4f}")
    
    # Visualizations
    st.subheader("Greek Visualizations")
    
    viz_tabs = st.tabs(["All Greeks", "Individual Greek", "3D Surface"])
    
    with viz_tabs[0]:
        # Plot all Greeks against underlying price
        st.markdown("#### All Greeks vs Underlying Price")
        
        # Create price range centered around current price
        S_range = np.linspace(max(1, S * 0.5), S * 1.5, 100)
        
        fig = plot_greeks_vs_underlying(S_range, K, T, r, sigma, option_type)
        st.pyplot(fig)
    
    with viz_tabs[1]:
        # Plot individual Greek
        st.markdown("#### Individual Greek Analysis")
        
        selected_greek = st.selectbox("Select Greek to visualize", 
                                     ["delta", "gamma", "theta", "vega", "rho"])
        
        # Create ranges for both variables
        param1_name = st.selectbox("X-axis Variable", ["Underlying Price (S)", "Time to Maturity (T)", 
                                                     "Volatility (σ)", "Risk-free Rate (r)"])
        
        if param1_name == "Underlying Price (S)":
            param1_range = np.linspace(max(1, S * 0.5), S * 1.5, 100)
            x_label = "Underlying Price ($)"
            param1_current = S
            param_values = [(x, K, T, r, sigma, option_type) for x in param1_range]
        elif param1_name == "Time to Maturity (T)":
            param1_range = np.linspace(0.01, max(0.01, T * 2), 100)
            x_label = "Time to Maturity (years)"
            param1_current = T
            param_values = [(S, K, x, r, sigma, option_type) for x in param1_range]
        elif param1_name == "Volatility (σ)":
            param1_range = np.linspace(0.01, max(0.01, sigma * 2.5), 100)
            x_label = "Volatility"
            param1_current = sigma
            param_values = [(S, K, T, r, x, option_type) for x in param1_range]
        else:  # "Risk-free Rate (r)"
            param1_range = np.linspace(0.0, max(0.001, r * 3), 100)
            x_label = "Risk-free Rate"
            param1_current = r
            param_values = [(S, K, T, x, sigma, option_type) for x in param1_range]
        
        # Calculate the selected Greek for each parameter value
        greek_values = []
        for params in param_values:
            try:
                all_greeks = calculate_all_greeks(*params)
                greek_values.append(all_greeks[selected_greek])
            except:
                greek_values.append(np.nan)
        
        # Plot the Greek
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(param1_range, greek_values, 'b-', linewidth=2)
        ax.axvline(x=param1_current, color='gray', linestyle='--', alpha=0.7)
        
        ax.set_title(f'{selected_greek.capitalize()} vs {param1_name}', fontsize=14)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(f'{selected_greek.capitalize()} Value', fontsize=12)
        ax.grid(True)
        
        st.pyplot(fig)
    
    with viz_tabs[2]:
        # 3D surface plot of a Greek
        st.markdown("#### Greek Surface Visualization")
        
        selected_greek_3d = st.selectbox("Select Greek for 3D visualization", 
                                        ["delta", "gamma", "theta", "vega", "rho"],
                                        key="greek3d")
        
        # Create ranges
        S_range = np.linspace(max(1, S * 0.5), S * 1.5, 20)
        T_range = np.linspace(0.01, max(0.01, T * 2), 20)
        
        fig = plot_greek_surface(S_range, T_range, K, r, sigma, selected_greek_3d, option_type)
        st.pyplot(fig)

def option_strategies_page():
    """Page for analyzing option strategies and payoffs."""
    st.header("Option Strategies")
    
    st.markdown("""
    Analyze payoffs for different option strategies:
    * Single options: Calls and puts
    * Spreads: Bull/bear, vertical, calendar
    * Combinations: Straddles, strangles, butterflies, iron condors
    """)
    
    # Base parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        S = st.number_input("Current Underlying Price", min_value=1.0, value=100.0, step=1.0)
    
    with col2:
        min_K = st.number_input("Minimum Strike for Analysis", min_value=1.0, value=80.0, step=5.0)
    
    with col3:
        max_K = st.number_input("Maximum Strike for Analysis", min_value=1.0, value=120.0, step=5.0)
    
    # Create price range for payoff analysis
    S_range = np.linspace(min_K * 0.8, max_K * 1.2, 1000)
    
    # Strategy selection
    st.subheader("Select Strategy")
    
    strategy_type = st.selectbox("Strategy Type", 
                               ["Single Options", "Spreads", "Combinations", "Custom"])
    
    if strategy_type == "Single Options":
        # Single option strategy
        option_type = st.selectbox("Option Type", ["call", "put"])
        position_type = st.selectbox("Position", ["long", "short"])
        strike = st.slider("Strike Price", min_value=float(min_K), max_value=float(max_K), 
                          value=float(S), step=1.0)
        premium = st.number_input("Option Premium", min_value=0.01, value=5.0, step=0.1)
        
        quantity = 1 if position_type == "long" else -1
        
        strategy = [{
            "type": option_type,
            "strike": strike,
            "premium": premium,
            "quantity": quantity
        }]
        
    elif strategy_type == "Spreads":
        # Spread strategies
        spread_type = st.selectbox("Spread Type", ["Bull Call Spread", "Bear Call Spread", 
                                                 "Bull Put Spread", "Bear Put Spread",
                                                 "Calendar Call Spread", "Calendar Put Spread"])
        
        low_strike = st.slider("Lower Strike", min_value=float(min_K), max_value=float(max_K), 
                              value=float(S)-10.0, step=1.0)
        high_strike = st.slider("Higher Strike", min_value=float(low_strike), max_value=float(max_K), 
                               value=float(S)+10.0, step=1.0)
        
        low_premium = st.number_input("Lower Strike Premium", min_value=0.01, value=5.0, step=0.1)
        high_premium = st.number_input("Higher Strike Premium", min_value=0.01, value=2.0, step=0.1)
        
        if spread_type == "Bull Call Spread":
            strategy = [
                {"type": "call", "strike": low_strike, "premium": low_premium, "quantity": 1},
                {"type": "call", "strike": high_strike, "premium": high_premium, "quantity": -1}
            ]
        elif spread_type == "Bear Call Spread":
            strategy = [
                {"type": "call", "strike": low_strike, "premium": low_premium, "quantity": -1},
                {"type": "call", "strike": high_strike, "premium": high_premium, "quantity": 1}
            ]
        elif spread_type == "Bull Put Spread":
            strategy = [
                {"type": "put", "strike": low_strike, "premium": low_premium, "quantity": -1},
                {"type": "put", "strike": high_strike, "premium": high_premium, "quantity": 1}
            ]
        elif spread_type == "Bear Put Spread":
            strategy = [
                {"type": "put", "strike": low_strike, "premium": low_premium, "quantity": 1},
                {"type": "put", "strike": high_strike, "premium": high_premium, "quantity": -1}
            ]
        elif spread_type == "Calendar Call Spread":
            strategy = [
                {"type": "call", "strike": low_strike, "premium": low_premium, "quantity": -1},
                {"type": "call", "strike": low_strike, "premium": high_premium, "quantity": 1}
            ]
            st.info("Note: Calendar spreads involve different expiration dates, which this payoff analysis doesn't fully capture.")
        else:  # Calendar Put Spread
            strategy = [
                {"type": "put", "strike": high_strike, "premium": low_premium, "quantity": -1},
                {"type": "put", "strike": high_strike, "premium": high_premium, "quantity": 1}
            ]
            st.info("Note: Calendar spreads involve different expiration dates, which this payoff analysis doesn't fully capture.")
            
    elif strategy_type == "Combinations":
        # Option combinations
        combo_type = st.selectbox("Combination Type", ["Long Straddle", "Short Straddle", 
                                                     "Long Strangle", "Short Strangle",
                                                     "Butterfly", "Iron Condor"])
        
        if combo_type in ["Long Straddle", "Short Straddle"]:
            strike = st.slider("Strike Price", min_value=float(min_K), max_value=float(max_K), 
                              value=float(S), step=1.0)
            call_premium = st.number_input("Call Premium", min_value=0.01, value=5.0, step=0.1)
            put_premium = st.number_input("Put Premium", min_value=0.01, value=4.0, step=0.1)
            
            quantity = 1 if combo_type == "Long Straddle" else -1
            
            strategy = [
                {"type": "call", "strike": strike, "premium": call_premium, "quantity": quantity},
                {"type": "put", "strike": strike, "premium": put_premium, "quantity": quantity}
            ]
            
        elif combo_type in ["Long Strangle", "Short Strangle"]:
            put_strike = st.slider("Put Strike", min_value=float(min_K), max_value=float(max_K), 
                                  value=float(S)-10.0, step=1.0)
            call_strike = st.slider("Call Strike", min_value=float(put_strike), max_value=float(max_K), 
                                   value=float(S)+10.0, step=1.0)
            call_premium = st.number_input("Call Premium", min_value=0.01, value=3.0, step=0.1)
            put_premium = st.number_input("Put Premium", min_value=0.01, value=2.0, step=0.1)
            
            quantity = 1 if combo_type == "Long Strangle" else -1
            
            strategy = [
                {"type": "call", "strike": call_strike, "premium": call_premium, "quantity": quantity},
                {"type": "put", "strike": put_strike, "premium": put_premium, "quantity": quantity}
            ]
            
        elif combo_type == "Butterfly":
            mid_strike = st.slider("Middle Strike", min_value=float(min_K), max_value=float(max_K), 
                                  value=float(S), step=1.0)
            wing_width = st.slider("Wing Width", min_value=1.0, max_value=20.0, 
                                  value=10.0, step=1.0)
            
            low_strike = mid_strike - wing_width
            high_strike = mid_strike + wing_width
            
            low_premium = st.number_input("Lower Strike Premium", min_value=0.01, value=8.0, step=0.1)
            mid_premium = st.number_input("Middle Strike Premium", min_value=0.01, value=5.0, step=0.1)
            high_premium = st.number_input("Higher Strike Premium", min_value=0.01, value=3.0, step=0.1)
            
            strategy = [
                {"type": "call", "strike": low_strike, "premium": low_premium, "quantity": 1},
                {"type": "call", "strike": mid_strike, "premium": mid_premium, "quantity": -2},
                {"type": "call", "strike": high_strike, "premium": high_premium, "quantity": 1}
            ]
            
        else:  # Iron Condor
            put_spread_width = st.slider("Put Spread Width", min_value=1.0, max_value=10.0, 
                                        value=5.0, step=1.0)
            call_spread_width = st.slider("Call Spread Width", min_value=1.0, max_value=10.0, 
                                         value=5.0, step=1.0)
            corridor_width = st.slider("Corridor Width", min_value=5.0, max_value=30.0, 
                                      value=10.0, step=1.0)
            
            short_put_strike = S - corridor_width/2
            long_put_strike = short_put_strike - put_spread_width
            short_call_strike = S + corridor_width/2
            long_call_strike = short_call_strike + call_spread_width
            
            long_put_premium = st.number_input("Long Put Premium", min_value=0.01, value=2.0, step=0.1)
            short_put_premium = st.number_input("Short Put Premium", min_value=0.01, value=4.0, step=0.1)
            short_call_premium = st.number_input("Short Call Premium", min_value=0.01, value=3.5, step=0.1)
            long_call_premium = st.number_input("Long Call Premium", min_value=0.01, value=1.5, step=0.1)
            
            strategy = [
                {"type": "put", "strike": long_put_strike, "premium": long_put_premium, "quantity": 1},
                {"type": "put", "strike": short_put_strike, "premium": short_put_premium, "quantity": -1},
                {"type": "call", "strike": short_call_strike, "premium": short_call_premium, "quantity": -1},
                {"type": "call", "strike": long_call_strike, "premium": long_call_premium, "quantity": 1}
            ]
            
    else:  # Custom strategy
        num_legs = st.number_input("Number of Options in Strategy", min_value=1, max_value=10, value=2, step=1)
        
        strategy = []
        
        for i in range(num_legs):
            st.markdown(f"#### Option {i+1}")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                option_type = st.selectbox(f"Type {i+1}", ["call", "put"], key=f"type_{i}")
            
            with col2:
                strike = st.number_input(f"Strike {i+1}", min_value=float(min_K), max_value=float(max_K), 
                                        value=float(S), step=1.0, key=f"strike_{i}")
            
            with col3:
                premium = st.number_input(f"Premium {i+1}", min_value=0.01, value=5.0, step=0.1, key=f"premium_{i}")
            
            with col4:
                quantity = st.number_input(f"Quantity {i+1}", min_value=-10, max_value=10, value=1, step=1, key=f"qty_{i}")
            
            strategy.append({
                "type": option_type,
                "strike": strike,
                "premium": premium,
                "quantity": quantity
            })
    
    # Visualize the strategy payoff
    st.subheader("Strategy Payoff")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = plot_strategy_payoff(strategy, S_range, ax)
    
    # Add vertical line at current price
    ax.axvline(x=S, color='blue', linestyle='-', alpha=0.3)
    ax.text(S, ax.get_ylim()[1]*0.9, f"Current: ${S}", rotation=90, 
            ha='right', va='top', color='blue', fontsize=10)
    
    # Show the plot
    st.pyplot(fig)
    
    # Calculate and display key metrics
    st.subheader("Strategy Analysis")
    
    # Calculate payoffs at key points
    payoffs = []
    for s in S_range:
        total_payoff = 0
        for leg in strategy:
            option_type = leg["type"]
            strike = leg["strike"]
            premium = leg["premium"]
            quantity = leg["quantity"]
            
            if option_type == "call":
                leg_payoff = quantity * (max(0, s - strike) - premium)
            else:  # put
                leg_payoff = quantity * (max(0, strike - s) - premium)
                
            total_payoff += leg_payoff
        payoffs.append(total_payoff)
    
    payoffs = np.array(payoffs)
    
    # Find max profit, max loss, and break-even points
    max_profit = np.max(payoffs)
    max_loss = np.min(payoffs)
    
    # Find break-even points (approximate)
    break_evens = []
    for i in range(len(S_range)-1):
        if (payoffs[i] <= 0 and payoffs[i+1] > 0) or \
           (payoffs[i] >= 0 and payoffs[i+1] < 0):
            # Linear interpolation for more accurate break-even point
            s1, s2 = S_range[i], S_range[i+1]
            p1, p2 = payoffs[i], payoffs[i+1]
            break_even = s1 - p1 * (s2 - s1) / (p2 - p1)
            break_evens.append(break_even)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Maximum Profit", f"${max_profit:.2f}")
    
    with col2:
        st.metric("Maximum Loss", f"${max_loss:.2f}")
    
    with col3:
        st.metric("Risk/Reward Ratio", f"{abs(max_loss/max_profit):.2f}" if max_profit != 0 else "∞")
    
    # Show break-even points
    if break_evens:
        st.write("Break-even point(s):")
        for i, be in enumerate(break_evens):
            st.write(f"BE {i+1}: ${be:.2f}")
    else:
        st.write("No break-even points found in the analyzed range.")
    
    # Strategy cost/credit
    total_cost = 0
    for leg in strategy:
        total_cost += leg["premium"] * leg["quantity"]
    
    if total_cost > 0:
        st.write(f"Strategy Cost (Debit): ${total_cost:.2f}")
    else:
        st.write(f"Strategy Credit: ${-total_cost:.2f}")

def about_page():
    """About page with project information."""
    st.header("About Options Pricing Simulator")
    
    st.markdown("""
    ### 📚 Project Overview
    
    This Options Pricing Simulator is an educational tool designed to help users understand 
    the mathematical models behind options pricing, Greeks calculation, and option strategies.
    
    ### 🧮 Mathematical Models
    
    #### Black-Scholes Model
    The Black-Scholes formula for pricing European options:
    
    **Call option price:**
    $$C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$$
    
    **Put option price:**
    $$P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)$$
    
    Where:
    $$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2) \cdot T}{\sigma \sqrt{T}}$$
    $$d_2 = d_1 - \sigma \sqrt{T}$$
    
    - $S$: Current price of the underlying asset
    - $K$: Strike price of the option
    - $T$: Time to maturity in years
    - $r$: Risk-free interest rate
    - $\sigma$: Volatility of the underlying asset
    - $N()$: Cumulative distribution function of the standard normal distribution
    
    #### Binomial Tree Model
    The Binomial Tree model discretizes time and the underlying asset's price movements, 
    making it suitable for both European and American options.
    
    - Up factor: $u = e^{\sigma \sqrt{\Delta t}}$
    - Down factor: $d = 1/u$
    - Risk-neutral probability: $p = \frac{e^{r\Delta t} - d}{u - d}$
    
    Where $\Delta t$ is the time step size.
    
    ### 🔢 Option Greeks
    
    The option Greeks measure the sensitivity of the option price to changes in various parameters:
    
    - **Delta ($\Delta$)**: Sensitivity to changes in the underlying price
      - Call Delta: $\Delta_c = N(d_1)$
      - Put Delta: $\Delta_p = N(d_1) - 1$
    
    - **Gamma ($\Gamma$)**: Rate of change of Delta (second derivative of price w.r.t. underlying)
      - $\Gamma = \frac{N'(d_1)}{S \sigma \sqrt{T}}$
    
    - **Theta ($\Theta$)**: Sensitivity to time decay
      - Call Theta: $\Theta_c = -\frac{S \sigma N'(d_1)}{2\sqrt{T}} - rKe^{-rT}N(d_2)$
      - Put Theta: $\Theta_p = -\frac{S \sigma N'(d_1)}{2\sqrt{T}} + rKe^{-rT}N(-d_2)$
    
    - **Vega**: Sensitivity to volatility
      - $\mathcal{V} = S \sqrt{T} N'(d_1)$
    
    - **Rho ($\rho$)**: Sensitivity to interest rate
      - Call Rho: $\rho_c = K T e^{-rT} N(d_2)$
      - Put Rho: $\rho_p = -K T e^{-rT} N(-d_2)$
    
    ### 📊 Features
    
    - Calculate option prices using different models
    - Visualize option prices and Greeks in multiple dimensions
    - Analyze option strategies and payoff diagrams
    - Calculate implied volatility
    
    ### 📝 Credits
    
    This simulator was developed as an educational project to help understand the concepts 
    of options pricing and risk management.
    """)

if __name__ == "__main__":
    main()