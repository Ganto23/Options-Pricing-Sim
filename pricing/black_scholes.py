"""
Black-Scholes Option Pricing Model implementation.

This module implements the Black-Scholes formula for pricing European options.
"""

import numpy as np
from scipy.stats import norm

def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option price using Black-Scholes formula.
    
    Parameters:
    -----------
    S : float
        Current price of the underlying asset
    K : float
        Strike price of the option
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annual rate expressed as a decimal)
    sigma : float
        Volatility of the underlying asset (annual volatility expressed as a decimal)
    option_type : str, optional
        Type of the option - either 'call' or 'put'
        
    Returns:
    --------
    float
        The calculated option price
        
    Notes:
    ------
    The Black-Scholes formula for call options is:
    C = S * N(d1) - K * exp(-r * T) * N(d2)
    
    And for put options:
    P = K * exp(-r * T) * N(-d2) - S * N(-d1)
    
    Where:
    d1 = (ln(S/K) + (r + sigma^2/2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    N() is the CDF of the standard normal distribution
    """
    # Validate inputs
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("Stock price, strike, time to maturity and volatility must be positive")
    
    if option_type not in ['call', 'put']:
        raise ValueError("Option type must be either 'call' or 'put'")
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Calculate call or put price
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def implied_volatility(market_price, S, K, T, r, option_type='call', 
                      precision=0.00001, max_iterations=100, initial_vol=0.2):
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Parameters:
    -----------
    market_price : float
        Observed market price of the option
    S : float
        Current price of the underlying asset
    K : float
        Strike price of the option
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annual rate expressed as a decimal)
    option_type : str, optional
        Type of the option - either 'call' or 'put'
    precision : float, optional
        Desired precision for the result
    max_iterations : int, optional
        Maximum number of iterations for the Newton-Raphson algorithm
    initial_vol : float, optional
        Initial volatility guess
        
    Returns:
    --------
    float
        The implied volatility value
        
    Notes:
    ------
    This function uses the Newton-Raphson method to find the volatility
    that makes the Black-Scholes price match the observed market price.
    """
    sigma = initial_vol
    
    for i in range(max_iterations):
        price = black_scholes_price(S, K, T, r, sigma, option_type)
        
        # Calculate option vega
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        
        # Newton-Raphson update
        diff = market_price - price
        
        if abs(diff) < precision:
            return sigma
        
        # Avoid division by zero
        if abs(vega) < 1e-8:
            return sigma
            
        sigma = sigma + diff / vega
        
        # Ensure volatility stays positive
        if sigma <= 0:
            sigma = 0.001
            
    # If we reach max iterations without converging
    return sigma