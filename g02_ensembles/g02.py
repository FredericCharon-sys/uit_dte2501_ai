import pandas as pd

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import random as rnd

df = pd.read_csv('data_2021.csv')


def create_dataset(w_size):
    x, y = [], []
    for i in range(len(df.Demand) - w_size):
        x.append(df.Demand[i:i + w_size].tolist())
        #print('new X values:', df.Demand[i:i + w_size].tolist())
        y.append(df.Demand[i + w_size])
        #print('new Y value:', df.Demand[i + w_size])
    return x, y


def create_tree(dataset):
    model = DecisionTreeRegressor(random_state=0)
    model.fit(dataset[0], dataset[1])
    return model


def make_prediction(tree, dataset):
    predictions = [0 for i in range(window_size)]
    for i in range(len(dataset[0])):
        #print('original:', Y[i])
        #print('predicted:', tree1.predict([X[i]]))
        predictions.append(tree1.predict([dataset[0][i]]))
    return predictions


def create_random_tree(dataset):
    model = DecisionTreeRegressor(
        criterion=rnd.choice(['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
        splitter=rnd.choice(['best', 'random']),
        max_features=rnd.choice([1.0, 'sqrt', 'log2'])
        )
    model.fit(dataset[0], dataset[1])
    return model


def bootstrapping(dataset):
    x, y = [], []
    for j in range(len(dataset[0])):
        if rnd.random() < 0.5:
            x.append(dataset[0][j])
            y.append(dataset[1][j])
    return x, y


window_size = 20
dataset = create_dataset(window_size)

tree1 = create_tree(dataset)
plt.plot(df.Demand)
plt.plot(make_prediction(tree1, dataset))

all_trees = []
for i in range(80):
    all_trees.append(create_random_tree(bootstrapping(dataset)))
'''plot_tree(tree1)
plt.show()
'''

plt.show()
