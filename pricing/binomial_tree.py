"""
Binomial Tree Option Pricing Model implementation.

This module implements the Binomial Tree model for pricing
both European and American options.
"""

import numpy as np

def binomial_tree_price(S, K, T, r, sigma, option_type='call', option_style='european', 
                        n_steps=100):
    """
    Calculate option price using the Binomial Tree model (Cox-Ross-Rubinstein).
    
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
    option_style : str, optional
        Style of the option - either 'european' or 'american'
    n_steps : int, optional
        Number of time steps in the binomial tree
        
    Returns:
    --------
    float
        The calculated option price
        
    Notes:
    ------
    The binomial tree model discretizes time and the underlying's price movements,
    using an approach that converges to Black-Scholes as n_steps approaches infinity.
    """
    # Validate inputs
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("Stock price, strike, time to maturity, and volatility must be positive")
    
    if option_type not in ['call', 'put']:
        raise ValueError("Option type must be either 'call' or 'put'")
        
    if option_style not in ['european', 'american']:
        raise ValueError("Option style must be either 'european' or 'american'")
    
    # Calculate dt and the up and down factors
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    
    # Risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)
    
    # Initialize asset price tree
    price_tree = np.zeros((n_steps + 1, n_steps + 1))
    for i in range(n_steps + 1):
        for j in range(i + 1):
            price_tree[j, i] = S * (u ** (i - j)) * (d ** j)
    
    # Initialize option value tree at expiration (final column)
    option_tree = np.zeros((n_steps + 1, n_steps + 1))
    
    for i in range(n_steps + 1):
        if option_type == 'call':
            option_tree[i, n_steps] = max(0, price_tree[i, n_steps] - K)
        else:  # put
            option_tree[i, n_steps] = max(0, K - price_tree[i, n_steps])
    
    # Backward recursion to calculate option value at earlier nodes
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            # Expected option value (discounted)
            expected = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            
            if option_style == 'european':
                option_tree[j, i] = expected
            else:  # american
                # For American options, consider early exercise
                if option_type == 'call':
                    intrinsic = max(0, price_tree[j, i] - K)
                else:  # put
                    intrinsic = max(0, K - price_tree[j, i])
                option_tree[j, i] = max(expected, intrinsic)
    
    # Return the option price at the root node
    return option_tree[0, 0]

def trinomial_tree_price(S, K, T, r, sigma, option_type='call', option_style='european',
                         n_steps=50):
    """
    Calculate option price using a Trinomial Tree model.
    
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
    option_style : str, optional
        Style of the option - either 'european' or 'american'
    n_steps : int, optional
        Number of time steps in the trinomial tree
        
    Returns:
    --------
    float
        The calculated option price
    """
    # Validate inputs
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        raise ValueError("Stock price, strike, time to maturity, and volatility must be positive")
    
    if option_type not in ['call', 'put']:
        raise ValueError("Option type must be either 'call' or 'put'")
        
    if option_style not in ['european', 'american']:
        raise ValueError("Option style must be either 'european' or 'american'")
    
    # Calculate dt and factors for the trinomial tree
    dt = T / n_steps
    dx = sigma * np.sqrt(3 * dt)
    
    # Movement factors
    u = np.exp(dx)  # Up factor
    d = np.exp(-dx)  # Down factor
    m = 1.0  # Middle (no movement)
    
    # Risk-neutral probabilities
    pu = ((np.exp(r * dt/2) - np.exp(-sigma * np.sqrt(dt/12))) / 
          (np.exp(dx) - np.exp(-sigma * np.sqrt(dt/12)))) ** 2
    pd = ((np.exp(sigma * np.sqrt(dt/12)) - np.exp(r * dt/2)) /
          (np.exp(sigma * np.sqrt(dt/12)) - np.exp(-dx))) ** 2
    pm = 1 - pu - pd  # Probability of middle state
    
    # Initialize asset price tree (represented as a list of lists)
    prices = []
    for i in range(n_steps + 1):
        prices.append([S * (u ** (i - j)) * (d ** j) for j in range(2*i+1) if j <= 2*i])
    
    # Initialize option values at expiration
    option_values = []
    for i in range(2*n_steps+1):
        if i <= 2*n_steps:
            if option_type == 'call':
                option_values.append(max(0, prices[n_steps][i] - K))
            else:  # put
                option_values.append(max(0, K - prices[n_steps][i]))
    
    # Backward recursion to calculate option values
    for i in range(n_steps-1, -1, -1):
        new_values = []
        for j in range(2*i+1):
            if j <= 2*i:
                # Expected value (discounted)
                expected = np.exp(-r * dt) * (
                    pu * option_values[j] + 
                    pm * option_values[j+1] + 
                    pd * option_values[j+2]
                )
                
                if option_style == 'european':
                    new_values.append(expected)
                else:  # american
                    # Consider early exercise for American options
                    if option_type == 'call':
                        intrinsic = max(0, prices[i][j] - K)
                    else:  # put
                        intrinsic = max(0, K - prices[i][j])
                    new_values.append(max(expected, intrinsic))
        option_values = new_values
        
    # Return the option price (root node)
    return option_values[0]