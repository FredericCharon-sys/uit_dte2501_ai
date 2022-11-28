import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.neural_network import MLPRegressor
import random as rnd
import warnings
warnings.filterwarnings('ignore')
matplotlib.use('TkAgg')

def create_dataset(w_size, dataframe):
    """
    gets a Pandas dataframe and a window size as input, using the sliding window approach to create a dataset
    """
    x, y = [], []
    for i in range(len(dataframe.Demand) - w_size):
        x.append(dataframe.Demand[i:i + w_size].tolist())
        y.append(dataframe.Demand[i + w_size])
    return x, y


def make_prediction(model_list, dataset):
    """
    gets an amount of models (e.g. trees) and a dataset and calculates the average prediction for each output
       based on the last n (= window size) inputs
    """
    predictions = [None for _ in range(window_size)]  # there are no predictions for the first n (= window size) outputs
    for i in range(len(dataset[0])):    # iterating through each element of the dataset
        pred = 0    # used to get the average prediction in the end
        for model in model_list:    # iterating through each model in the list
            pred += model.predict([dataset[0][i]])
        predictions.append(pred/len(model_list))    # adding the average prediction of all models for y to the list
    return predictions


def create_random_tree(dataset):
    """
    gets a dataset as input, creates a random DecisionTree and trains it on the dataset
    """
    model = DecisionTreeRegressor(
        # parameters to be randomized:
        criterion=rnd.choice(['squared_error', 'friedman_mse', 'absolute_error', 'poisson']),
        splitter=rnd.choice(['best', 'random']),
        min_samples_split=rnd.randint(2, 5),
        max_depth=rnd.randint(1, 100),
        min_samples_leaf=rnd.randint(1, 3)
        )
    model.fit(dataset[0], dataset[1])   # training model on the dataset
    return model


def bootstrapping(dataset):
    """
    using bootstrapping to create a random subset of the initial dataset
    """
    x, y = [], []
    m_classifiers = 180    # we set the amount of classifiers to 180
    random_index_list = rnd.sample(range(0, len(dataset[0])-1), m_classifiers)  # we pick random indexes for the subset
    for index in random_index_list:     # for each random index we add the data to the new dataset
        x.append(dataset[0][index])
        y.append(dataset[1][index])
    return x, y


def create_and_train(model_type, n_models, dataset):
    """
    gets a model type and creates n models of that type using bootstrapping on the dataset
    """
    all_models = []
    if model_type == 'tree':
        # if the type == tree, we create n trees using the bootstrapping function
        for i in range(n_models):
            all_models.append(create_random_tree(bootstrapping(dataset)))
    elif model_type == 'MLP':
        # if the type == MLP, we create n multi-layer perceptron using the bootstrapping function
        for i in range(n_models):
            all_models.append(create_mlp(bootstrapping(dataset)))
    return all_models


def create_mlp(dataset):
    """
    creating a multi-layer perceptron and training it on the dataset
    """
    model = MLPRegressor()
    model.fit(dataset[0], dataset[1])
    return model


def make_prediction_unknown_data(model_list, dataset, n_days):
    """
    a variation of the prediction function, used to predict data for a longer time than we have inputs for.
    Getting a list of models, a dataset and an amount of days as input, the amount of days determines after how many
        outputs the prediction stops
    """
    pred_dataset = dataset
    predictions = make_prediction(model_list, dataset)  # first we predict the outputs like before for the given inputs

    for day in range(len(predictions), n_days):   # iteration for each output where no input data is given, up to n days

        # we use the last prediction as input data for the next predictions by adding it to the dataset
        last_data = pred_dataset[0][-1][1:]
        last_data.append(predictions[-1][0])
        pred_dataset[0].append(last_data)

        # then we calculate the next prediction for all the models and use the mean value
        pred = 0
        for model in model_list:
            pred += model.predict([pred_dataset[0][-1]])
        predictions.append(pred / len(model_list))
    return predictions


# --- Task 2 ---
df_2021 = pd.read_csv('data_2021.csv')
window_size = 40
Dataset = create_dataset(window_size, df_2021)
SingleTree = create_random_tree(Dataset)
plt.plot(df_2021.Demand, label='original')
plt.plot(make_prediction([SingleTree], Dataset), label='single tree prediction')

# --- Task 3 ---
RandomForest = create_and_train('tree', 207, Dataset)
plt.plot(make_prediction(RandomForest, Dataset), label='random forest prediction', color='green')

# --- Task 4 ---
MLP_models = create_and_train('MLP', 207, Dataset)
plt.plot(make_prediction(MLP_models, Dataset), label='MLP prediction', color='red')


plt.legend()
plt.xlabel('days')
plt.ylabel('demand')
plt.title('Year 2021 - demand and predictions')
plt.show()


# --- Task 5 ---
df_2022 = pd.read_csv('data_2022.csv')
Dataset_22 = create_dataset(window_size, df_2022)

plt.plot(df_2022.Demand)
plt.plot(make_prediction_unknown_data(RandomForest, Dataset_22, 365), label='random forest prediction', color='green')
plt.plot(make_prediction_unknown_data(MLP_models, Dataset_22, 365), label='MLP prediction', color='red')


plt.legend()
plt.xlabel('days')
plt.ylabel('demand')
plt.title('Year 2022 - demand and predictions')
plt.show()


# asking Adreas: Test und Training for bootstrapping? task 5 behaviour as expected? How many classifiers for bagging?
