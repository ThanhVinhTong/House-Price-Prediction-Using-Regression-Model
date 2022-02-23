import matplotlib.pyplot as pl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

from numpy.core.shape_base import hstack
from collections import deque
from pandas.core.base import DataError

logfile = "log.txt"

# get a random degree permutation of N.
def get_permutation(N, degree):
    save_permutation = []
    Q = deque([([], 0)])
    while len(Q) > 0:
      state = Q.pop()
      permutation = state[0]
      start = state[1]
      for i in range(start, N):
        next_permutation = permutation.copy()
        next_permutation.append(i)
        if len(next_permutation) == degree:
          if next_permutation not in save_permutation:
            save_permutation.append(next_permutation)
        else:
          Q.append((next_permutation, i))
    return np.array(save_permutation).astype(int)

# nang bac + tao feature moi
def polynominal_degree(feature, degree):
    feature_copy = feature.copy()
    for i in range(2, degree + 1):
        permutation_i = get_permutation(feature.shape[1], i)
        feature_i = np.array([np.prod(feature[:, permutation_i[j]], axis =1) for j in range(permutation_i.shape[0])])
        feature_i = feature_i.T
        feature_copy = np.hstack((feature_i, feature_copy))
    return feature_copy
    
def split_data(split, X, Y, shuffle = True):
    split_percentage = int(split * X.shape[0])
    X_copy = X.copy()
    Y_copy = Y.copy()
    if shuffle:
        random = np.random.permutation(X_copy.shape[0])
        X_copy = X_copy[random]
        Y_copy = Y_copy[random]
    return X_copy[:split_percentage], Y_copy[:split_percentage], X_copy[split_percentage:], Y_copy[split_percentage:]

def feature_normalized(feature):
    return (feature - np.min(feature, axis = 0))/(np.max(feature, axis = 0) - np.min(feature, axis = 0))

def load_data_and_split(file_name, split):
    data = pd.read_csv(file_name)
    data_values = data.values
    price = data_values[:, -1:] # Houses' prices in real life
    given_data = data_values[:, :-1] # Houses' features to based on, in order to predict price
    feature = data_values[:, :-1] # This is also houses' features to based on, but for the purpose of training the AI
    feature = polynominal_degree(feature, 3) # polynomial feature transform
    feature = feature_normalized(feature) # Normalizing the feature so that it become small for easy calculation
    X_train, Y_train, X_test, Y_test =  split_data(split, feature, price)
    return given_data, price, X_train, Y_train, X_test, Y_test

def training_GD(file_name, split, out_lr, weight_val, bias_val, number_iterations, batch_size, eps):
    #lr: learning rate
    #bias_val: dang kieu sai so
    lr = out_lr
    given_data, price, X_train, Y_train, X_test, Y_test = load_data_and_split(file_name, split)
    X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
    X_test2 = X_test.copy()
    X_test2 = np.hstack((X_test2, np.ones((X_test2.shape[0], 1))))
    
    weight = np.full((X_train.shape[1], 1), weight_val)
    weight[:, -1] = bias_val
    list_loss = []
    for i in range(number_iterations):
        random = np.random.permutation(int(X_train.shape[0]/batch_size))
        new_weight = None
        sum_loss = 0
        for j in random:
            Xj = X_train[j * batch_size: (j + 1) * batch_size]
            Yj = Y_train[j * batch_size: (j + 1) * batch_size]
            Zj = Xj.dot(weight)
            new_weight = weight - lr * Xj.T.dot(Zj - Yj)
            sum_loss += np.sum(np.sum((Zj - Yj) ** 2, axis = 1), axis = 0)/batch_size
        if np.sum(np.linalg.norm(new_weight - weight)) < eps:    
            break
        else:
            weight = new_weight
        list_loss.append(sum_loss)

    Y_predict1 = (X_train.dot(weight))
    Y_predict2 = (X_test2.dot(weight))
    predicted_data = np.vstack((Y_predict1, Y_predict2))
    write_to_file(given_data, predicted_data, price)

    plt.plot(list_loss)
    plt.title("Loss Graph")
    plt.ylabel('loss value')
    plt.xlabel('eplison')
    plt.show()
    return weight, X_train, Y_train, X_test, Y_test, given_data, price

def score_test(X, Y, weight, train_test):
    mae_score = np.sum(np.sum(np.abs(X.dot(weight) - Y), axis = 1), axis = 0) / X.shape[0] #mean absolutely error
    mse_score = np.sum(np.sum((X.dot(weight) - Y)**2, axis = 1), axis = 0) / X.shape[0] #mean squared error
    print("{} MAE score is : {}".format(train_test, mae_score))
    print("{} MSE score is : {}".format(train_test, mse_score))

def process_data(file_name, split, lr, weight_val, bias_val, number_iterations, batch_size, eps, training_algorithm):
    weight, X_train, Y_train, X_test, Y_test, given_data, price = training_algorithm(file_name, split, lr, weight_val, bias_val, number_iterations, batch_size, eps)
    # Minimum price of the data
    minimum_price = np.amin(price)

    # Maximum price of the data
    maximum_price = np.amax(price)

    # Mean price of the data
    mean_price = np.mean(price)

    # Median price of the data
    median_price = np.median(price)

    # Standard deviation of prices of the data
    std_price = np.std(price)

    # Show the calculated statistics
    print("\nStatistics for Predicting houses' dataset:\n")
    print("Minimum price: ${}".format(minimum_price)) 
    print("Maximum price: ${}".format(maximum_price))
    print("Mean price: ${}".format(mean_price))
    print("Median price ${}".format(median_price))
    print("Standard deviation of prices: ${}\n".format(std_price))
    
    # Show the score test for this Prediction model by calculating MAE and MSE of training and testing sets
    print("Score for Predicting houses' dataset:\n")
    score_test(X_train, Y_train, weight, "Train")
    X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))
    score_test(X_test, Y_test, weight, "Test")
    

def write_to_file(given_data, predicted_data, price):
    predicted_data = np.round(predicted_data.astype(np.float64), 3)
#    given_data = np.round(given_data, 2)
    price = np.round(price.astype(np.float64), 3)
    
    if os.path.exists(logfile):
        os.remove(logfile)

    with open(logfile, "a") as o:
        o.write('{}\t\t'.format("Predicted Price"))
        o.write('{}\t\t'.format("Real Price"))
        o.write('{}\t\t'.format("Given Data"))
        o.write('\n')
        for i in range(len(given_data)):
            o.write('{}\t\t'.format(predicted_data[i]) )
            o.write('{}\t\t'.format(price[i]) )
            o.write('{}\t\t'.format(given_data[i]) )
            o.write('\n')
        o.close()

process_data("data.csv", 0.75, 0.01, 0.002, 0., 3000, 32, 1e-7, training_GD)