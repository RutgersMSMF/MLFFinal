import requests
import numpy as np
import matplotlib.pyplot as plt

def get_quote():

    return 0

def get_tradier_data():
    """
    Returns Tradier Implied Volatility
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

    # Interpolate Data
    # Model Requires An Even Grid

    max_elem = np.max(strikes)
    min_elem = np.min(strikes)
    elem_set = int(max_elem - min_elem)

    for i in range(elem_set): 

        if (strikes[i + 1] - strikes[i]) != 1: 

            # Linear Interpolation Implied Volatility
            x0 = strikes[i]
            x1 = strikes[i + 1]

            y0 = ivol[i]
            y1 = ivol[i + 1]

            x = strikes[i] + 1
            y = y0 + (x - x0) * ((y1 - y0) / (x1 - x0))
            ivol.insert(i + 1, y)

            # Linear Interpolation Midpoint Price
            x0 = strikes[i]
            x1 = strikes[i + 1]

            y0 = midpoint[i]
            y1 = midpoint[i + 1]

            x = strikes[i] + 1
            y = y0 + (x - x0) * ((y1 - y0) / (x1 - x0))
            midpoint.insert(i + 1, y)

            # Linear Interpolation Delta

            # Add Strikes
            strikes.insert(i + 1, strikes[i] + 1)

    fig, (ax1) = plt.subplots(1)

    ax1.plot(strikes, ivol)
    ax1.set_xlabel("Strikes")
    ax1.set_ylabel("Implied Volatility")
    ax1.set_title("Volatility Surface")

    fig, (ax1) = plt.subplots(1)

    ax1.plot(strikes, midpoint)
    ax1.set_xlabel("Strikes")
    ax1.set_ylabel("Midpoint Price")
    ax1.set_title("Fair Market Price")

    plt.show()

    return np.array(midpoint), np.array(ivol), strikes