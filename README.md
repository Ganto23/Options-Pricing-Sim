# Options Pricing Simulator

A comprehensive, educational, and visually interactive options pricing simulator built in Python.

You can find this app at https://options-pricing-sim.streamlit.app.

## Overview

This project provides tools for pricing financial options using different models, visualizing option prices and Greeks, and analyzing option strategies. It's designed to be both educational and practical for anyone learning about quantitative finance.

## Features

- **Option Pricing Models:**
  - Black-Scholes Model (for European options)
  - Binomial Tree Model (for both European and American options)
  - Trinomial Tree Model (for enhanced accuracy)

- **Options Greeks:**
  - Delta: Sensitivity to underlying price
  - Gamma: Rate of change of Delta
  - Theta: Sensitivity to time decay
  - Vega: Sensitivity to volatility
  - Rho: Sensitivity to interest rate

- **Interactive Visualizations:**
  - Option price surfaces
  - Greek sensitivity surfaces
  - Payoff diagrams for various strategies
  - Interactive Streamlit dashboard

- **Advanced Analytics:**
  - Implied volatility calculation
  - Strategy payoff analysis
  - Break-even point calculations
  - Model convergence analysis

## Project Structure

```
options-pricing-simulator/
│
├── pricing/
│   ├── black_scholes.py         # Black-Scholes pricing model
│   ├── binomial_tree.py         # Binomial tree pricing model
│   └── greeks.py                # Calculation of option Greeks
│
├── app/
│   ├── dashboard.py             # Streamlit UI
│   └── visuals.py               # Plotting functions
│
├── notebooks/
│   └── options_simulation.ipynb # Jupyter notebook for demo and walkthrough
│
├── data/                        # Sample data directory
│
├── tests/                       # Test scripts
│   └── __init__.py
│
├── README.md                    # This file
└── requirements.txt             # Project dependencies
```

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/options-pricing-simulator.git
   cd options-pricing-simulator
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Interactive Dashboard

Run the Streamlit dashboard:

```
streamlit run app/dashboard.py
```

This will launch a web interface where you can:
- Price options using different models
- Visualize option prices and Greeks
- Analyze various option strategies
- Calculate implied volatility

### Using the Library

You can also use the library directly in your Python code:

```python
from pricing.black_scholes import black_scholes_price
from pricing.greeks import calculate_all_greeks

# Price a European call option
price = black_scholes_price(
    S=100,    # Underlying price
    K=100,    # Strike price
    T=1,      # Time to maturity (years)
    r=0.05,   # Risk-free rate
    sigma=0.2, # Volatility
    option_type='call'
)

# Calculate option Greeks
greeks = calculate_all_greeks(100, 100, 1, 0.05, 0.2, 'call')
print(f"Price: ${price:.2f}")
print(f"Delta: {greeks['delta']:.4f}")
```

### Jupyter Notebook

Explore the options simulation notebook for a detailed walkthrough:

```
jupyter notebook notebooks/options_simulation.ipynb
```

## Mathematical Foundation

### Black-Scholes Formula

For a call option:
```
C = S * N(d1) - K * exp(-rT) * N(d2)
```

For a put option:
```
P = K * exp(-rT) * N(-d2) - S * N(-d1)
```

Where:
- d1 = (ln(S/K) + (r + σ²/2) * T) / (σ * √T)
- d2 = d1 - σ * √T
- N() is the CDF of the standard normal distribution
- S is the underlying asset price
- K is the strike price
- T is the time to maturity
- r is the risk-free rate
- σ is the volatility of the underlying

### Binomial Tree Model

The binomial model discretizes time and the underlying's price movements:

- Up factor (u): e^(σ * √Δt)
- Down factor (d): 1/u
- Risk-neutral probability (p): (e^(rΔt) - d) / (u - d)