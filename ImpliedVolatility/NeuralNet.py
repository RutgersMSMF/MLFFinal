from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

from Data.Tradier import get_tradier_data

def get_neural_network():
    """
    Feedforward Neural Network to Fit Volatility Surface
    """

    # Fetch Data
    X, y, strikes = get_tradier_data()

    # Train Test Split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

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
    model.add(Dense(2, activation = 'sigmoid', name = 'Hidden-Layer'))
    
    # Output Layer
    model.add(Dense(1, activation = 'tanh', name = 'Output-Layer'))

    print("Training Network")
    learning_rate = np.linspace(0.0001, 1, 100)
    epoch_count = 50
    MSE_TRAIN = []
    MSE_TEST = []

    for l in learning_rate:

        # sgd = SGD(learning_rate = l)
        adam = Adam(learning_rate = l)
        model.compile(loss = "mean_squared_error", optimizer = adam, metrics = ["mean_squared_error"])
        H = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epoch_count, batch_size = 128)
        MSE_TRAIN.append(sum(H.history["mean_squared_error"]))
        MSE_TEST.append(sum(H.history["val_mean_squared_error"]))

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(learning_rate)

    ax2.plot(MSE_TRAIN)
    ax2.plot(MSE_TEST)

    plt.show()

    # evaluate the network
    print("Evaluating Network")
    predictions = model.predict(X_test, batch_size = 128)

    # plot the training loss and accuracy
    interval = np.arange(0, epoch_count)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

    ax1.plot(interval, H.history["loss"], label = "Train Loss")
    ax1.plot(interval, H.history["val_loss"], label = "Test Loss")
    ax1.set_title("Loss")
    ax1.legend(loc = "best")

    ax2.plot(interval, H.history["mean_squared_error"], label = "Train MSE")
    ax2.plot(interval, H.history["val_mean_squared_error"], label = "Test MSE")
    ax2.set_title("Mean Squared Error")
    ax2.legend(loc = "best")

    ax3.plot(predictions, label = "Predictions")
    ax3.plot(y_test, label = "Test Set")
    ax3.set_title("Volatility Predictions")
    ax3.legend(loc = "best")

    plt.show()

    return 0



