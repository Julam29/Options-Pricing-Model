import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import math #math.sqrt() math.log math.e
from scipy.stats import norm
# from scipy.optimize import brentq


def getHistoricalVolatility(data):
    log_returns = np.log(data/data.shift(1)).dropna()
    sigma_daily = log_returns.std()
    sigma_annual = sigma_daily * math.sqrt(252)
    return sigma_annual.item()

def GBM(S0, rate, volatility, time):
    S = []
    S.append(S0)
    for x in range(1, int(time*252)):
        Z = np.random.normal()
        ST = S[x-1] * math.exp((rate - 0.5 * volatility**2) * (1/252) + volatility * math.sqrt(1/252) * Z)
        S.append(float(ST))
    return S

def MonteCarloSim(S, K, r, sigma, T, iterations=1000):
    prices = []
    for i in range(iterations):
        path = GBM(S, r, sigma, T)
        last_price = path[-1]
        prices.append(last_price)
    return prices

def histogram(data, title):
    plt.hist(data, bins=50, alpha=0.75)
    plt.title(title)
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    # plt.grid(True)
    plt.show()

def graphing(simulated_prices):
    plt.figure()
    upperbound = max(simulated_prices)
    interval = math.ceil(upperbound/100)*5

    # Sort the prices so the line graph is meaningful
    sorted_prices = sorted(simulated_prices)

    plt.plot(sorted_prices)
    plt.xticks(np.arange(0, len(sorted_prices), interval))
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.title("Simulated Price Path (Line Graph)")
    plt.show()

data = yf.download("AAPL", period = "1y")["Close"]
volatility = getHistoricalVolatility(data)
print("Historical Volatility: ", volatility)
# simulated_prices = GBM(248.04, 0.0364, volatility, 5/252)
simulated_prices = MonteCarloSim(248.04, 250, 0.0364, volatility, 5/252, iterations=1000)
print(np.mean(simulated_prices))
histogram(simulated_prices, "Monte Carlo Simulation of AAPL over 5 days")
