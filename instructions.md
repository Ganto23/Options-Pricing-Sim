# 🧠 Prompt: Build Me an Options Pricing Simulator in Python

I want to build a fully functional, educational, and visually interactive **Options Pricing Simulator** using Python. Please create a clean, modular, and well-documented project that accomplishes the following:

---

## 🎯 Project Goals

Develop a tool that:
1. **Implements theoretical pricing models for options**:
   - Black-Scholes Model (for European options)
   - Binomial Tree Model (for American options)

2. **Simulates and visualizes option prices** based on key input parameters:
   - Underlying price (S)
   - Strike price (K)
   - Time to maturity (T)
   - Volatility (σ)
   - Risk-free rate (r)
   - Option type (call or put)

3. **Calculates and visualizes option Greeks**:
   - Delta, Gamma, Vega, Theta, Rho

4. **Plots option price surfaces and payoff diagrams**

5. **Provides an interactive frontend** (e.g., sliders) to change parameters and observe real-time output

---

## 🧰 Required Features

- Modular Python code with separate files for:
  - `black_scholes.py`
  - `binomial_tree.py`
  - `greeks.py`
  - `visuals.py`
  - `app.py` or `notebook.ipynb` for running simulations

- Mathematical accuracy and well-commented formulas for:
  - CDF of the standard normal distribution
  - Binomial recursion logic
  - Finite difference approximations for Greeks

- Visualization tools using:
  - `matplotlib`, `seaborn`, or `plotly`
  - Optional: Build a **Streamlit** or **Dash** interface

- Clear documentation and inline math formulas (in comments or markdown)

- Easy user input: Option to run via notebook or web app

---

## 📁 Deliverables

- Python codebase structured as:
options-pricing-simulator/
│
├── pricing/
│   ├── black_scholes.py         # Black-Scholes pricing model
│   ├── binomial_tree.py         # Binomial tree pricing model
│   └── greeks.py                # Calculation of option Greeks
│
├── app/
│   ├── dashboard.py             # Streamlit or Dash UI (optional)
│   └── visuals.py               # Plotting functions (payoffs, surfaces, etc.)
│
├── notebooks/
│   └── options_simulation.ipynb # Jupyter notebook for demo and walkthrough
│
├── data/                        # (Optional) Volatility surfaces or sample price data
│
├── README.md                    # Project overview and instructions
├── requirements.txt             # Dependencies list
└── .env                         # API keys (if applicable)



- README that explains:
- The math behind each model
- How to run the simulator
- Example inputs/outputs
- Screenshots or GIFs of the UI (if applicable)

---

## ✅ Optional Stretch Goals

- Include Implied Volatility Calculator (using Newton-Raphson method)
- Allow download/export of option prices or surfaces
- Add real-time price feeds using `yfinance` or `alpha_vantage`

---

## 📌 Target Python version: 3.10+

Make the project visually clean, mathematically sound, and suitable for a **quant finance portfolio** or **technical interview**. Prioritize accuracy, modular design, and ease of exploration.

