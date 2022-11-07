import numpy as np


def get_users_average(rating):
    """
    A function that get a user's ratings and returns the average rating
    """
    sum = 0
    n = 0
    for r in rating:
        if r != 'x':
            sum += r
            n += 1
    return sum / n


def get_variance(rating):
    """
    A function that gets a user's ratings and subtracts the user's average rating from them to get the variance
    """
    variance = []
    for r in rating:
        if r == 'x':
            variance.append('x')
        else:
            variance.append(r-get_users_average(rating))
    return variance


def compute_correlation(variances):
    """
    The function from the lecture to compute the correlations
    Variance for the active user is variances[0] and for the other persons variances[1:]
    """
    correlations = []
    active_user = variances[0]
    for other_user in variances[1:]:
        temp_active = []
        temp_other = []

        for i in range(len(variances[0])):
            if active_user[i] != 'x' and other_user[i] != 'x':
                temp_active.append(active_user[i])
                temp_other.append(other_user[i])

        correlation_matrix = np.corrcoef(np.array(temp_active), np.array(temp_other))
        correlation = correlation_matrix[0][1]
        correlations.append(correlation)
    return correlations


def predict_vote(variances, correlations, kappa, average_vote_for_a, vote_index):
    """
    The function from the lecture to predict a vote
    """
    sum_correlation_variance = 0
    for i in range(len(variances[1:])):
        if variances[i+1][vote_index] != 'x':
            sum_correlation_variance += variances[i+1][vote_index] * correlations[i]
    p_a_j = average_vote_for_a + kappa * sum_correlation_variance
    return p_a_j


# creating the list of all ratings, the first entry belongs to the active user and the other ratings to person 1 to  7
all_ratings = [[3, 2, 3, 'x', 4, 1, 'x', 5],
               [5, 5, 'x', 3, 4, 'x', 'x', 'x'],
               [4, 3, 3, 4, 5, 2, 3, 3],
               [1, 3, 3, 2, 2, 3, 4, 1],
               ['x', 3, 'x', 'x', 'x', 5, 'x', 'x'],
               [3, 4, 'x', 4, 5, 5, 1, 'x'],
               [4, 5, 2, 5, 3, 2, 2, 3],
               [1, 1, 'x', 'x', 2, 1, 'x', 'x']]

average_vote_for_a = get_users_average(all_ratings[0])  # getting the
all_variances = list(map(get_variance, all_ratings))    # saving the variances of all ratings in a new list
correlations = compute_correlation(all_variances)

# We predict the active user's vote for index 3 (Westworld) and for index 6 (Skam)
vote1 = predict_vote(all_variances, correlations, kappa=1, average_vote_for_a=average_vote_for_a, vote_index=3)
vote2 = predict_vote(all_variances, correlations, kappa=1, average_vote_for_a=average_vote_for_a, vote_index=6)

print("The predicted vote for Westworld: ", vote1, "\n     rounded: ", round(vote1))
print("\nThe predicted vote for Skam: ", vote2, "\n     rounded: ", round(vote2))
