import pandas as pd
import math

'''
Importing the dataframe with pandas
'''
# Iris dataset link
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df_iris = pd.read_csv(URL, header=None, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])

# Renaming the classes by giving them numbers
iris_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
for i in range(len(iris_names)):
    df_iris.loc[df_iris['class'] == iris_names[i], df_iris.columns == 'class'] = i

# Transforming the pandas dataframe into a matrix/nested_list so it's easier for the functions to work with it
data = []
for idx, row in df_iris.iterrows():
    data.append([row['sepal length'], row['sepal width'], row['petal length'], row['petal width'], row['class']])

'''
In this function the euclidean distance between two data points is calculated
This is done by iterating through each feature of these points
'''
def eucl_distance(dp1, dp2):
    distance = 0

    # we use length-1 since the last element is the class and not one of the features
    for i in range(len(dp1) - 1):
        distance += ((dp1[i] - dp2[i]) ** 2)
    return math.sqrt(distance)


'''
To find the k nearest neighbors we need to get a data point as input. Now we calculate the eucl_distance() between our 
    new point with each of our initial data points. Since we only need the k nearest we remove all elements from that
    list except the ones with the closest distance
'''
def find_nearest_neighbors(new_dp, k):
    nearest_neighbors = []

    # we iterate through all our data points
    for element in data:

        # for each element in our dataset we append a new element to the list and sort it by euclidean distance
        nearest_neighbors.append((element, eucl_distance(element, new_dp)))
        nearest_neighbors.sort(key=lambda y: y[1])

        # if the length of the list is greater than k after adding a new element we remove the last object
        #   which is the one with the highest distance (since we sorted it before)
        if len(nearest_neighbors) > k:
            nearest_neighbors.pop()

    print('\nOur %d nearest neighbors are: ' % k)
    for element in nearest_neighbors:
        print('features: %s, class: %s, distance: %f' % (element[0][0:4], iris_names[element[0][4]], element[1]))

    # returning the list with the k nearest neighbors
    return nearest_neighbors


'''
The predict class function gets a new datapoint and a value for k, these are used to get the nearest neighbors from the
    find_nearest_neighbors function. After that we count which class is the most common one for these neighbors and we
    finally get our result
'''
def predict_class(new_data, k):
    nearest_neighbors = find_nearest_neighbors(new_data, k)
    class_list = []

    # we create a new list only containing the classes of the nearest neighbors
    for datapoint in nearest_neighbors:
        class_list.append(datapoint[0][4])
    # we count which class is the most common one
    return max(set(class_list), key=class_list.count)


# the predict class function is called with a new data point and a value for k
resulting_class = predict_class([7, 3.1, 1.3, 0.7], 5)
print('\nOur resulting class is the class', iris_names[resulting_class])
