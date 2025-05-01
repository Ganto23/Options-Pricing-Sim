"""
Option Greeks calculation module.

This module calculates the option Greeks:
- Delta: sensitivity to underlying price
- Gamma: sensitivity of delta to underlying price
- Theta: sensitivity to time decay
- Vega: sensitivity to volatility
- Rho: sensitivity to interest rate
"""

import numpy as np
from scipy.stats import norm
import sys
sys.path.append("..") 
from pricing.black_scholes import black_scholes_price

def calculate_delta(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Delta of an option.
    
    Delta measures the rate of change of the option price with respect 
    to changes in the underlying asset's price.
    
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
        The calculated Delta value
    """
    if T <= 0:
        # At expiration, delta is either 0 or 1 for call, 0 or -1 for put
        if option_type == 'call':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:  # put
        delta = -norm.cdf(-d1)
    
    return delta

def calculate_gamma(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Gamma of an option.
    
    Gamma measures the rate of change in the delta with respect
    to changes in the underlying price.
    
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
        The calculated Gamma value
    """
    if T <= 0:
        return 0.0  # Gamma is 0 at expiration
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Gamma is the same for both calls and puts
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    
    return gamma

def calculate_theta(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Theta of an option.
    
    Theta measures the sensitivity of the option price to the passage of time,
    often referred to as time decay.
    
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
        The calculated Theta value (per year)
    """
    if T <= 0:
        return 0.0  # Theta is 0 at expiration
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # The first part of theta is the same for both calls and puts
    common_term = -(S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T))
    
    if option_type == 'call':
        theta = common_term - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put
        theta = common_term + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    return theta

def calculate_vega(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Vega of an option.
    
    Vega measures sensitivity to volatility. It is the derivative of the option price
    with respect to the volatility of the underlying asset.
    
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
        The calculated Vega value (for a 1% change in volatility)
    """
    if T <= 0:
        return 0.0  # Vega is 0 at expiration
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    
    # Vega is the same for both calls and puts
    vega = S * np.sqrt(T) * norm.pdf(d1) * 0.01  # For a 1% change in volatility
    
    return vega

def calculate_rho(S, K, T, r, sigma, option_type='call'):
    """
    Calculate the Rho of an option.
    
    Rho measures the sensitivity of the option price to changes 
    in the risk-free interest rate.
    
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
        The calculated Rho value (for a 1% change in interest rate)
    """
    if T <= 0:
        return 0.0  # Rho is 0 at expiration
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) * 0.01  # For a 1% change in rate
    else:  # put
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) * 0.01  # For a 1% change in rate
    
    return rho

def calculate_all_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate all option Greeks at once.
    
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
    dict
        A dictionary containing all calculated Greeks
    """
    greeks = {
        'delta': calculate_delta(S, K, T, r, sigma, option_type),
        'gamma': calculate_gamma(S, K, T, r, sigma, option_type),
        'theta': calculate_theta(S, K, T, r, sigma, option_type),
        'vega': calculate_vega(S, K, T, r, sigma, option_type),
        'rho': calculate_rho(S, K, T, r, sigma, option_type)
    }
    
    return greeks

def calculate_greeks_by_finite_diff(S, K, T, r, sigma, option_type='call', h=0.01):
    """
    Calculate option Greeks using finite difference methods.
    
    This is an alternative method that can be used to verify the analytical calculations
    or for options without closed-form solutions.
    
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
    h : float, optional
        Step size for finite difference calculations
        
    Returns:
    --------
    dict
        A dictionary containing all calculated Greeks
    """
    # Base price
    price = black_scholes_price(S, K, T, r, sigma, option_type)
    
    # Delta: dV/dS
    price_up_S = black_scholes_price(S + h, K, T, r, sigma, option_type)
    price_down_S = black_scholes_price(S - h, K, T, r, sigma, option_type)
    delta = (price_up_S - price_down_S) / (2 * h)
    
    # Gamma: d²V/dS²
    gamma = (price_up_S - 2 * price + price_down_S) / (h ** 2)
    
    # Theta: dV/dT (negative, as we're moving forward in time)
    if T <= h:
        # If time is very close to expiration, use forward difference
        price_down_T = black_scholes_price(S, K, T - h, r, sigma, option_type)
        theta = (price - price_down_T) / h
    else:
        price_down_T = black_scholes_price(S, K, T - h, r, sigma, option_type)
        theta = (price_down_T - price) / h  # Note: negated as theta measures decay
    
    # Vega: dV/dσ
    price_up_sig = black_scholes_price(S, K, T, r, sigma + h, option_type)
    price_down_sig = black_scholes_price(S, K, T, r, sigma - h, option_type)
    vega = (price_up_sig - price_down_sig) / (2 * h) * 0.01  # Scale for 1% change
    
    # Rho: dV/dr
    price_up_r = black_scholes_price(S, K, T, r + h, sigma, option_type)
    price_down_r = black_scholes_price(S, K, T, r - h, sigma, option_type)
    rho = (price_up_r - price_down_r) / (2 * h) * 0.01  # Scale for 1% change
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }