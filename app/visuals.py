"""
Visualization module for option pricing models.

This module provides functions for visualizing:
- Option price surfaces
- Greek sensitivities
- Payoff diagrams
- Volatility smiles/surfaces
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append("..") 
from pricing.black_scholes import black_scholes_price
from pricing.greeks import calculate_all_greeks

# Define custom color scheme
CREAM = "#fbf6f3"
TANGERINE = "#feb06a"
TURQUOISE = "#36d6e7"
BLUE_GRAY = "#5d6c89"

# Set default style with custom colors
sns.set_theme(style="whitegrid")
custom_palette = [BLUE_GRAY, TURQUOISE, TANGERINE, CREAM]
sns.set_palette(custom_palette)

def plot_option_price_vs_underlying(S_range, K, T, r, sigma, option_type='call'):
    """
    Plot option price as a function of underlying asset price.
    
    Parameters:
    -----------
    S_range : array-like
        Range of underlying asset prices to plot
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
    """
    prices = [black_scholes_price(S, K, T, r, sigma, option_type) for S in S_range]
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.plot(S_range, prices, color=BLUE_GRAY, linewidth=2)
    
    # Add intrinsic value line
    if option_type == 'call':
        intrinsic = [max(0, S - K) for S in S_range]
    else:  # put
        intrinsic = [max(0, K - S) for S in S_range]
    plt.plot(S_range, intrinsic, color=TANGERINE, linestyle='--', linewidth=1.5)
    
    # Add vertical line at strike price
    plt.axvline(x=K, color=CREAM, linestyle='--')
    
    plt.title(f'{option_type.capitalize()} Option Price vs Underlying Price', fontsize=14, color=BLUE_GRAY)
    plt.xlabel('Underlying Price ($)', fontsize=12, color=BLUE_GRAY)
    plt.ylabel('Option Price ($)', fontsize=12, color=BLUE_GRAY)
    plt.grid(True, color=CREAM, alpha=0.4)
    plt.legend(['Option Price', 'Intrinsic Value', 'Strike Price'])
    plt.tight_layout()
    
    return fig

def plot_option_price_surface(S_range, T_range, K, r, sigma, option_type='call'):
    """
    Create a 3D surface plot of option price as a function of underlying price and time.
    
    Parameters:
    -----------
    S_range : array-like
        Range of underlying asset prices
    T_range : array-like
        Range of times to maturity in years
    K : float
        Strike price of the option
    r : float
        Risk-free interest rate (annual rate expressed as a decimal)
    sigma : float
        Volatility of the underlying asset (annual volatility expressed as a decimal)
    option_type : str, optional
        Type of the option - either 'call' or 'put'
    """
    S_mesh, T_mesh = np.meshgrid(S_range, T_range)
    Z = np.zeros_like(S_mesh)
    
    for i in range(len(T_range)):
        for j in range(len(S_range)):
            Z[i, j] = black_scholes_price(S_mesh[i, j], K, T_mesh[i, j], r, sigma, option_type)
    
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Create custom colormap based on our colors
    custom_cmap = plt.cm.colors.LinearSegmentedColormap.from_list('turquoise_tangerine', 
                                                               [BLUE_GRAY, TURQUOISE, TANGERINE])
    
    surface = ax.plot_surface(S_mesh, T_mesh, Z, cmap=custom_cmap, edgecolor='none', alpha=0.8)
    
    ax.set_title(f'{option_type.capitalize()} Option Price Surface', fontsize=14, color=BLUE_GRAY)
    ax.set_xlabel('Underlying Price ($)', fontsize=12, color=BLUE_GRAY)
    ax.set_ylabel('Time to Maturity (years)', fontsize=12, color=BLUE_GRAY)
    ax.set_zlabel('Option Price ($)', fontsize=12, color=BLUE_GRAY)
    
    fig.colorbar(surface, ax=ax, shrink=0.7, aspect=10, label='Option Price ($)')
    
    return fig

def plot_greek_surface(S_range, T_range, K, r, sigma, greek='delta', option_type='call'):
    """
    Create a 3D surface plot of an option Greek as a function of underlying price and time.
    
    Parameters:
    -----------
    S_range : array-like
        Range of underlying asset prices
    T_range : array-like
        Range of times to maturity in years
    K : float
        Strike price of the option
    r : float
        Risk-free interest rate (annual rate expressed as a decimal)
    sigma : float
        Volatility of the underlying asset (annual volatility expressed as a decimal)
    greek : str, optional
        Greek to plot - one of 'delta', 'gamma', 'theta', 'vega', 'rho'
    option_type : str, optional
        Type of the option - either 'call' or 'put'
    """
    greek = greek.lower()
    if greek not in ['delta', 'gamma', 'theta', 'vega', 'rho']:
        raise ValueError("Greek must be one of 'delta', 'gamma', 'theta', 'vega', or 'rho'")
    
    S_mesh, T_mesh = np.meshgrid(S_range, T_range)
    Z = np.zeros_like(S_mesh)
    
    for i in range(len(T_range)):
        for j in range(len(S_range)):
            # Skip calculation for expired options (T=0)
            if T_mesh[i, j] <= 0:
                Z[i, j] = 0
                continue
                
            greeks = calculate_all_greeks(S_mesh[i, j], K, T_mesh[i, j], r, sigma, option_type)
            Z[i, j] = greeks[greek]
    
    fig = plt.figure(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    
    # Create custom colormap based on our colors
    custom_cmap = plt.cm.colors.LinearSegmentedColormap.from_list('turquoise_tangerine', 
                                                               [BLUE_GRAY, TURQUOISE, TANGERINE])
    
    surface = ax.plot_surface(S_mesh, T_mesh, Z, cmap=custom_cmap, edgecolor='none', alpha=0.8)
    
    ax.set_title(f'{option_type.capitalize()} Option {greek.capitalize()} Surface', 
               fontsize=14, color=BLUE_GRAY)
    ax.set_xlabel('Underlying Price ($)', fontsize=12, color=BLUE_GRAY)
    ax.set_ylabel('Time to Maturity (years)', fontsize=12, color=BLUE_GRAY)
    ax.set_zlabel(f'{greek.capitalize()}', fontsize=12, color=BLUE_GRAY)
    
    fig.colorbar(surface, ax=ax, shrink=0.7, aspect=10, label=f'{greek.capitalize()}')
    
    return fig

def plot_greeks_vs_underlying(S_range, K, T, r, sigma, option_type='call'):
    """
    Plot all Greeks against underlying price.
    
    Parameters:
    -----------
    S_range : array-like
        Range of underlying asset prices
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
    """
    greek_values = {
        'delta': [],
        'gamma': [],
        'theta': [],
        'vega': [],
        'rho': []
    }
    
    # Calculate Greeks for each stock price
    for S in S_range:
        greeks = calculate_all_greeks(S, K, T, r, sigma, option_type)
        for greek in greek_values:
            greek_values[greek].append(greeks[greek])
    
    # Create subplots
    fig, axs = plt.subplots(3, 2, figsize=(14, 16))
    fig.patch.set_facecolor('white')
    axs = axs.flatten()
    
    # Plot each Greek
    for i, (greek, values) in enumerate(greek_values.items()):
        if i < 5:  # We have 5 Greeks to plot
            ax = axs[i]
            ax.set_facecolor('white')
            ax.plot(S_range, values, color=BLUE_GRAY, linewidth=2)
            ax.set_title(f"{greek.capitalize()}", fontsize=14, color=BLUE_GRAY)
            ax.set_xlabel('Underlying Price ($)', fontsize=12, color=BLUE_GRAY)
            ax.set_ylabel(f'{greek.capitalize()} Value', fontsize=12, color=BLUE_GRAY)
            ax.axvline(x=K, color=TANGERINE, linestyle='--', alpha=0.7)
            ax.grid(True, color=CREAM, alpha=0.4)
    
    # Remove unused subplot
    fig.delaxes(axs[5])
    
    plt.tight_layout()
    plt.suptitle(f"{option_type.capitalize()} Option Greeks vs Underlying Price (T={T}, K={K}, σ={sigma})", 
                fontsize=16, y=1.02, color=BLUE_GRAY)
    
    return fig

def plot_option_payoff(S_range, K, premium, option_type='call', quantity=1, ax=None):
    """
    Plot the payoff diagram for an option at expiration.
    
    Parameters:
    -----------
    S_range : array-like
        Range of underlying asset prices at expiration
    K : float
        Strike price of the option
    premium : float
        Price paid (or received) for the option
    option_type : str, optional
        Type of the option - 'call' or 'put'
    quantity : int, optional
        Number of option contracts (positive for long, negative for short)
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates a new figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
    
    # Calculate payoff at expiration
    if option_type == 'call':
        payoff = [quantity * (max(0, S - K) - premium) for S in S_range]
        intrinsic = [quantity * max(0, S - K) for S in S_range]
    else:  # put
        payoff = [quantity * (max(0, K - S) - premium) for S in S_range]
        intrinsic = [quantity * max(0, K - S) for S in S_range]
    
    position = "Long" if quantity > 0 else "Short"
    
    # Plot the payoff
    ax.plot(S_range, payoff, color=BLUE_GRAY, linewidth=2)
    ax.plot(S_range, intrinsic, color=TANGERINE, linestyle='--', linewidth=1.5)
    
    # Add horizontal and vertical lines
    ax.axhline(y=0, color=CREAM, linestyle='-', alpha=0.7)
    ax.axvline(x=K, color=CREAM, linestyle='--')
    
    # Break-even points
    if option_type == 'call':
        break_even = K + premium
    else:
        break_even = K - premium
    
    ax.axvline(x=break_even, color=TURQUOISE, linestyle='--', alpha=0.7)
    
    ax.set_title(f'{position} {option_type.capitalize()} Option Payoff at Expiration', 
               fontsize=14, color=BLUE_GRAY)
    ax.set_xlabel('Underlying Price at Expiration ($)', fontsize=12, color=BLUE_GRAY)
    ax.set_ylabel('Profit/Loss ($)', fontsize=12, color=BLUE_GRAY)
    ax.grid(True, color=CREAM, alpha=0.4)
    ax.legend(['Net Payoff', 'Intrinsic Value', 'Break-Even Point'])
    
    return ax

def plot_strategy_payoff(strategy_list, S_range, ax=None):
    """
    Plot the combined payoff diagram for an option strategy.
    
    Parameters:
    -----------
    strategy_list : list of dicts
        List of options in the strategy, each with keys:
        - 'type': 'call' or 'put'
        - 'strike': strike price
        - 'premium': option price
        - 'quantity': number of contracts (positive for long, negative for short)
    S_range : array-like
        Range of underlying asset prices at expiration
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, creates a new figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
    
    combined_payoff = np.zeros(len(S_range))
    
    # Create a color cycle from our custom palette
    colors = [BLUE_GRAY, TURQUOISE, TANGERINE, CREAM]
    color_cycle = plt.cycler(color=colors)
    ax.set_prop_cycle(color_cycle)
    
    # Plot individual components and calculate combined payoff
    for i, option in enumerate(strategy_list):
        option_type = option.get('type', 'call')
        K = option.get('strike', 100)
        premium = option.get('premium', 0)
        quantity = option.get('quantity', 1)
        
        # Calculate payoff
        if option_type == 'call':
            payoff = np.array([quantity * (max(0, S - K) - premium) for S in S_range])
        else:  # put
            payoff = np.array([quantity * (max(0, K - S) - premium) for S in S_range])
        
        combined_payoff += payoff
        
        # Plot individual component with low alpha
        label = f"{abs(quantity)}x {'Long' if quantity > 0 else 'Short'} {option_type.capitalize()} K={K}"
        ax.plot(S_range, payoff, '--', alpha=0.4, label=label)
    
    # Plot combined payoff with stronger line
    ax.plot(S_range, combined_payoff, color=BLUE_GRAY, linewidth=2.5, label='Combined Strategy')
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color=CREAM, linestyle='-', alpha=0.7)
    
    # Find break-even points (where payoff crosses y=0)
    break_even_points = []
    for i in range(len(S_range)-1):
        if (combined_payoff[i] <= 0 and combined_payoff[i+1] > 0) or \
           (combined_payoff[i] >= 0 and combined_payoff[i+1] < 0):
            break_even_points.append(S_range[i])
    
    # Mark break-even points
    for point in break_even_points:
        ax.axvline(x=point, color=TURQUOISE, linestyle='--', alpha=0.7)
    
    ax.set_title('Option Strategy Payoff at Expiration', fontsize=14, color=BLUE_GRAY)
    ax.set_xlabel('Underlying Price at Expiration ($)', fontsize=12, color=BLUE_GRAY)
    ax.set_ylabel('Profit/Loss ($)', fontsize=12, color=BLUE_GRAY)
    ax.grid(True, color=CREAM, alpha=0.4)
    ax.legend(loc='best')
    
    return ax

def plot_implied_volatility_smile(strikes, market_prices, S, T, r, option_type='call'):
    """
    Plot the implied volatility smile.
    
    Parameters:
    -----------
    strikes : array-like
        List of strike prices
    market_prices : array-like
        Corresponding market prices for options
    S : float
        Current price of the underlying asset
    T : float
        Time to maturity in years
    r : float
        Risk-free interest rate (annual rate expressed as a decimal)
    option_type : str, optional
        Type of the options - 'call' or 'put'
    """
    from pricing.black_scholes import implied_volatility
    
    # Calculate implied volatilities
    implied_vols = []
    for K, price in zip(strikes, market_prices):
        try:
            vol = implied_volatility(price, S, K, T, r, option_type)
            implied_vols.append(vol)
        except Exception as e:
            print(f"Error calculating implied vol for K={K}: {e}")
            implied_vols.append(None)
    
    # Filter out None values
    valid_data = [(k, vol) for k, vol in zip(strikes, implied_vols) if vol is not None]
    if not valid_data:
        raise ValueError("No valid implied volatilities could be calculated")
    
    valid_strikes, valid_vols = zip(*valid_data)
    
    fig = plt.figure(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.set_facecolor('white')
    
    ax.plot(valid_strikes, valid_vols, 'o-', color=BLUE_GRAY, linewidth=2)
    
    # Add line for ATM strike
    ax.axvline(x=S, color=TANGERINE, linestyle='--', alpha=0.7)
    
    ax.set_title(f'Implied Volatility Smile (T={T})', fontsize=14, color=BLUE_GRAY)
    ax.set_xlabel('Strike Price ($)', fontsize=12, color=BLUE_GRAY)
    ax.set_ylabel('Implied Volatility', fontsize=12, color=BLUE_GRAY)
    ax.grid(True, color=CREAM, alpha=0.4)
    
    # Add moneyness labels
    ax.text(min(valid_strikes), max(valid_vols)*0.9, 'ITM', fontsize=12, 
             ha='center', va='center', color=BLUE_GRAY)
    ax.text(max(valid_strikes), max(valid_vols)*0.9, 'OTM', fontsize=12,
             ha='center', va='center', color=BLUE_GRAY)
    
    return fig