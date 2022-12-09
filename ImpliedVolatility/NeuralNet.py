from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt
import numpy as np

from Data.Tradier import get_tradier_data
# from vollib.black_scholes_merton import bsm_call

def get_neural_network(noise = True, noise_level = 0.000001):
    """
    Feed Forward Neural Network to Fit Volatility Surface
    """

    # Fetch Data
    X, y, strikes = get_tradier_data()

    # Add Noise to Data
    if noise: 

        new = []
        old = []
        butterfly_new = []
        butterfly_old = []

        for k in range(len(y)):
            
            new_t = []
            rv = np.random.normal(0, noise_level, len(y[k]))

            for n in range(len(y[k])):
                new_t.append(y[k][n] + rv[n])

            new.append(new_t)
            old.append(y[k])

            # Price Butterfly Spreads
            bn = []
            bo = []
            for v in range(1, len(new_t) - 1):
                bn.append(new_t[v - 1] - 2 * new_t[v] + new_t[v+1])
                bo.append(old[k][v - 1] - 2 * old[k][v] + old[k][v+1])

            butterfly_new.append(bn)
            butterfly_old.append(bo)

        fig, (ax1) = plt.subplots(1)

        ax1.plot(new[3], label = "New")
        ax1.plot(old[3], label = "Old")
        ax1.set_xlabel("Strikes")
        ax1.set_ylabel("Implied Volatility")
        ax1.set_title("Volatility Surface")

        fig, (ax1) = plt.subplots(1)

        ax1.plot(np.abs(new[3]) - np.array(old[3]), label = "Euclidean Distance")
        ax1.set_xlabel("Strikes")
        ax1.set_ylabel("Error")
        ax1.set_title("Residuals")

        fig, (ax1) = plt.subplots(1)

        ax1.plot(butterfly_new[3], label = "New Butterfly Spread")
        ax1.plot(butterfly_old[3], label = "Old Butterfly Spread")
        ax1.set_xlabel("Strikes")
        ax1.set_ylabel("Probability")
        ax1.set_title("Arbitrage Surface")

        plt.show()

    for i in range(len(X)):

        # Train Test Split 
        if noise:
            X_train, X_test, y_train, y_test = train_test_split(np.array(X[i]), np.array(new[i]), test_size = 0.20, random_state = 0)
        else:
            X_train, X_test, y_train, y_test = train_test_split(np.array(X[i]), np.array(y[i]), test_size = 0.20, random_state = 0)

        # Reshape Data
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        # Check Data Shape
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

        # Create Model
        model = Sequential(name = 'Vol-Surface')

        # Input Layer
        model.add(Input(shape = (1, ), name = 'Input-Layer'))

        # Hidden Layer 
        model.add(Dense(2, activation = 'swish', name = 'Hidden-Layer'))
        
        # Output Layer
        model.add(Dense(1, activation = 'sigmoid', name = 'Output-Layer'))

        # print("Training Network")
        # learning_rate = np.linspace(0.0001, 1, 100)
        # epoch_count = 50
        # MSE_TRAIN = []
        # MSE_TEST = []

        # for l in learning_rate:

        #     sgd = SGD(learning_rate = l)
        #     model.compile(loss = "mean_squared_error", optimizer = sgd, metrics = ["mean_squared_error"])
        #     H = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epoch_count, batch_size = 64)
        #     MSE_TRAIN.append(sum(H.history["mean_squared_error"]))
        #     MSE_TEST.append(sum(H.history["val_mean_squared_error"]))

        # fig, (ax1) = plt.subplots(1)

        # ax1.plot(learning_rate, MSE_TRAIN, label = "Train MSE")
        # ax1.plot(learning_rate, MSE_TEST, label = "Test MSE")
        # ax1.set_title("Optimal Learning Rate")
        # ax1.legend(loc = "best")

        # plt.show()

        # Optimal Learning Rate
        print("Training Network")
        learning_rate = np.linspace(0.0001, 1, 100)
        epoch_count = 50
        l_best = learning_rate[20]
        sgd = SGD(learning_rate = l_best)
        model.compile(loss = "mean_squared_error", optimizer = sgd, metrics = ["mean_absolute_percentage_error"])
        H = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epoch_count, batch_size = 64)

        # evaluate the network
        print("Evaluating Network")
        predictions = model.predict(X_test, batch_size = 64)

        # market_price = []
        # predicted_price = []

        # # Compute Risk Neutral Prices
        # for oc in X_test: 

        #     market_price.append()
        #     predicted_price.append()

        # plot the training loss and accuracy
        interval = np.arange(0, epoch_count)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

        ax1.plot(interval, H.history["loss"], label = "Train Loss")
        ax1.plot(interval, H.history["val_loss"], label = "Test Loss")
        ax1.set_title("Loss")
        ax1.legend(loc = "best")

        ax2.plot(interval, H.history["mean_absolute_percentage_error"], label = "Train MAPE")
        ax2.plot(interval, H.history["mean_absolute_percentage_error"], label = "Test MAPE")
        ax2.set_title("Mean Absolute Percentage Error")
        ax2.legend(loc = "best")

        ax3.plot(predictions, label = "Predictions")
        ax3.plot(y_test, label = "Test Set")
        ax3.set_title("Volatility Predictions")
        ax3.legend(loc = "best")

        ax4.plot(y_test - predictions, label = "Market - Prediction")
        ax4.set_title("Volatility Residuals")
        ax4.legend(loc = "best")

        plt.show()

    return 0




