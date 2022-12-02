import requests
import numpy as np
from numba import jit 
import matplotlib.pyplot as plt

def fetch_quote():

    return 0

def fetch_option_chain():
    """
    Returns Option Chain
    """

    # Fetch Expirations
    response = requests.get(

        'https://api.tradier.com/v1/markets/options/expirations',

        params = {
            'symbol': 'SPY', 
            'includeAllRoots': 'true', 
            'strikes': 'false',
            },

        headers = {
            'Authorization': 'Bearer GQZRHrCRsbRqE43Z0k6rKcDj73oU', 
            'Accept': 'application/json',
            }

    )

    json_response = response.json()
    expiry = json_response["expirations"]["date"]

    # Fetch Option Chain
    response = requests.get(
        
        'https://api.tradier.com/v1/markets/options/chains',

        params = {
            'symbol': 'SPY', 
            'expiration': expiry[10], 
            'greeks': 'true',
            },

        headers = {
            'Authorization': 'Bearer GQZRHrCRsbRqE43Z0k6rKcDj73oU', 
            'Accept': 'application/json',
            }

    )

    json_response = response.json()
    data = json_response["options"]["option"]

    midpoint = []
    strikes = []
    ivol = []
    delta = []

    # Grab Data from JSON Object
    for d in data:

        if d["option_type"] == "call":

            bid = d["bid"]
            ask = d["ask"]
            mid = (bid + ask) / 2.0
            midpoint.append(mid)

            strikes.append(d["strike"])
            ivol.append(d["greeks"]["mid_iv"])
            delta.append(d["greeks"]["delta"])

    return midpoint, strikes, ivol, delta

# @jit(nopython = True)
def get_tradier_data():
    """
    Returns Tradier Implied Volatility
    """

    midpoint, strikes, ivol, delta = fetch_option_chain()

    # Interpolate Data
    # Model Requires An Even Grid

    max_elem = np.max(strikes)
    min_elem = np.min(strikes)
    interval = 0.05
    elem_set = int((max_elem - min_elem) / interval) 
    print(elem_set)

    for i in range(elem_set): 

        if (strikes[i + 1] - strikes[i]) > interval: 

            # Linear Interpolation Implied Volatility
            x0 = strikes[i]
            x1 = strikes[i + 1]

            y0 = ivol[i]
            y1 = ivol[i + 1]

            x = strikes[i] + interval
            y = y0 + (x - x0) * ((y1 - y0) / (x1 - x0))
            ivol.insert(i + 1, y)

            # Linear Interpolation Midpoint Price
            x0 = strikes[i]
            x1 = strikes[i + 1]

            y0 = midpoint[i]
            y1 = midpoint[i + 1]

            x = strikes[i] + interval
            y = y0 + (x - x0) * ((y1 - y0) / (x1 - x0))
            midpoint.insert(i + 1, y)

            # Linear Interpolation Delta
            x0 = strikes[i]
            x1 = strikes[i + 1]

            y0 = delta[i]
            y1 = delta[i + 1]

            x = strikes[i] + interval
            y = y0 + (x - x0) * ((y1 - y0) / (x1 - x0))
            delta.insert(i + 1, y)

            # Add Strikes
            strikes.insert(i + 1, strikes[i] + interval)

    # Filter Deltas
    filtered_strikes = []
    filtered_deltas = []
    filtered_midpoint = []
    filtered_ivol = []

    upper_threshold = [1.00, 0.80, 0.60, 0.55]
    lower_threshold = [0.00, 0.20, 0.40, 0.45]
    
    for i in range(len(upper_threshold)):

        index = 0
        filtered_strikes_temp = []
        filtered_deltas_temp = []
        filtered_midpoint_temp = []
        filtered_ivol_temp = []

        for d in delta: 

            if d <= upper_threshold[i] and d >= lower_threshold[i]:

                filtered_strikes_temp.append(strikes[index])
                filtered_deltas_temp.append(delta[index])
                filtered_midpoint_temp.append(midpoint[index])
                filtered_ivol_temp.append(ivol[index])
            
            index+=1

        filtered_strikes.append(filtered_strikes_temp)
        filtered_deltas.append(filtered_deltas_temp)
        filtered_midpoint.append(filtered_midpoint_temp)
        filtered_ivol.append(filtered_ivol_temp)

    # fig, (ax1) = plt.subplots(1)

    # ax1.plot(filtered_strikes, filtered_ivol)
    # ax1.set_xlabel("Strikes")
    # ax1.set_ylabel("Implied Volatility")
    # ax1.set_title("Volatility Surface")

    # fig, (ax1) = plt.subplots(1)

    # ax1.plot(filtered_strikes, filtered_midpoint)
    # ax1.set_xlabel("Strikes")
    # ax1.set_ylabel("Midpoint Price")
    # ax1.set_title("Fair Market Price")

    # plt.show()

    return filtered_midpoint, filtered_ivol, filtered_strikes





