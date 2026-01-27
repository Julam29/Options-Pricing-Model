import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import math #math.sqrt() math.log math.e
import jax.numpy as jnp
from jax import grad
from jax.scipy.stats import norm as jax_norm
from scipy.optimize import brentq

# Black-Scholes Formula for European Call Option Pricing
"""
C = S × N(d1) - K × e^(-rT) × N(d2)

Where:  
d1 = [ln(S/K) + (r + σ²/2) × T] / (σ × √T)
d2 = d1 - σ × √T

N(x) = Cumulative standard normal distribution (use Excel's NORMSDIST or a calculator)
e = exponential constant (~2.71828)
ln = natural logarithm
"""
#Puts:
"""
P = K × e^(-rT) × N(-d2) - S × N(-d1)"""

def getHistoricalVolatility(data):
    log_returns = np.log(data/data.shift(1)).dropna()
    sigma_daily = log_returns.std()
    sigma_annual = sigma_daily * math.sqrt(252)
    return sigma_annual.item()

def getImpliedVolatility(C_mkt, S, K, r, T, sigma_low = 1e-6, sigma_high = 5.0, option_type='call'):
    f = lambda sigma: float(blackScholes(S, K, r, T, sigma, option_type) - C_mkt)
    return brentq(f, sigma_low, sigma_high)

# S = Asset Price, K = Strike Price, r = Interest Rate, T = Maturity time, sigma = Volatility
def blackScholes(S, K, r, T, sigma, option_type='call'):
    d1 = (jnp.log(S/K) + (r + (sigma**2)/2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - (sigma * jnp.sqrt(T))

    if option_type == 'call':
        return S * jax_norm.cdf(d1) - (K * jnp.exp(-r*T)) * jax_norm.cdf(d2)
    elif option_type == 'put':
        return (K * jnp.exp(-r*T)) * jax_norm.cdf(-d2) - S * jax_norm.cdf(-d1)
    else:
        return ("Invalid option type. Use 'call' or 'put'.")
    
def getGreeks(S, K, r, T, sigma, option_type):
    getDelta = grad(blackScholes, argnums = 0)
    getGamma = grad(getDelta, argnums = 0)
    getVega = grad(blackScholes, argnums = 4)
    getTheta = grad(blackScholes, argnums = 3)
    getRho = grad(blackScholes, argnums = 2)

    print("Delta:", getDelta(S, K, r, T, sigma, option_type))
    print("Gamma:", getGamma(S, K, r, T, sigma, option_type))
    print("Vega:", getVega(S, K, r, T, sigma, option_type))
    print("Theta:", getTheta(S, K, r, T, sigma, option_type))
    print("Rho:", getRho(S, K, r, T, sigma, option_type))

volatility1 = getImpliedVolatility(1.76, 248.04, 237.50, 0.0364, 5/252, 1e-6, 5, 'put')
option1 = blackScholes(248.04, 237.50, 0.0364, 5/252, volatility1, 'put')
print("Apple Implied Volatility:", volatility1)
print("Apple Put Option Price from Implied Volatility:", option1)
getGreeks(248.04, 237.50, 0.0364, 5/252, volatility1, 'put')