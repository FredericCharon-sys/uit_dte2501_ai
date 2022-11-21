import pandas as pd

import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.neural_network import MLPRegressor
import random as rnd
import warnings
warnings.filterwarnings('ignore')


def create_dataset(w_size, dataframe):
    x, y = [], []
    for i in range(len(dataframe.Demand) - w_size):
        x.append(dataframe.Demand[i:i + w_size].tolist())
        #print('new X values:', df.Demand[i:i + w_size].tolist())
        y.append(dataframe.Demand[i + w_size])
        #print('new Y value:', df.Demand[i + w_size])
    return x, y


def make_prediction(tree_list, dataset):
    predictions = [0.0 for i in range(window_size)]
    for i in range(len(dataset[0])):
        pred = 0
        for tree in tree_list:
            pred += tree.predict([dataset[0][i]])
        predictions.append(pred/len(tree_list))
    return predictions


def create_random_tree(dataset):
    model = DecisionTreeRegressor(
        criterion=rnd.choice(['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
        splitter=rnd.choice(['best', 'random']),
        min_samples_split=rnd.randint(2, 5),
        max_depth=rnd.randint(1, 100),
        min_samples_leaf=rnd.randint(1, 3)
        )
    model.fit(dataset[0], dataset[1])
    return model


def bootstrapping(dataset):
    x, y = [], []
    random_index_list = rnd.sample(range(0, len(dataset[0])-1), 180)
    for index in random_index_list:
        x.append(dataset[0][index])
        y.append(dataset[1][index])
    return x, y


def bagging(model_type, n_models, dataset):
    all_models = []
    if model_type == 'tree':
        for i in range(n_models):
            all_models.append(create_random_tree(bootstrapping(dataset)))
    elif model_type == 'MLP':
        for i in range(n_models):
            all_models.append(create_mlp(bootstrapping(dataset)))
    return all_models


def create_mlp(dataset):
    model = MLPRegressor()
    model.fit(dataset[0], dataset[1])
    return model


def make_prediction_unknown_data(model_list, dataset, n_days):
    pred_dataset = dataset
    predictions = make_prediction(model_list, dataset)
    for day in range(len(predictions), n_days):
        last_data = pred_dataset[0][-1][1:]
        last_data.append(predictions[-1][0])
        pred_dataset[0].append(last_data)

        pred = 0
        for model in model_list:
            pred += model.predict([pred_dataset[0][-1]])
        predictions.append(pred / len(model_list))
    return predictions


df_2021 = pd.read_csv('data_2021.csv')
window_size = 40
Dataset = create_dataset(window_size, df_2021)

RandomForest = bagging('tree', 207, Dataset)
MLP_models = bagging('MLP', 207, Dataset)
plt.plot(df_2021.Demand, label='original')
plt.plot(make_prediction([create_random_tree(Dataset)], Dataset), label='1 tree prediction')
plt.plot(make_prediction(RandomForest, Dataset), label='random forest prediction')
plt.plot(make_prediction(MLP_models, Dataset), label='MLP prediction')



plt.show()


df_2022 = pd.read_csv('data_2022.csv')
Dataset_22 = create_dataset(window_size, df_2022)

plt.plot(df_2022.Demand)
plt.plot(make_prediction_unknown_data(RandomForest, Dataset_22, 365))


plt.show()
