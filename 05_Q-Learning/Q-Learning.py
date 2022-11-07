states = [i for i in range(6)]

'''
Initializing Q and R-matrix, Q-matrix is filled with 0 and R-matrix is filled with the rewards of each 
    state action pair, if there's no connection between two rooms the reward is set to -1
'''
Q = [[0.0 for i in range(len(states))] for j in range(len(states))]
R = [[-1 for i in range(len(states))] for k in range(len(states))]

# filling R-matrix with the rewards
R[0][1] = 0  # state 1
R[0][3] = 0
R[1][0] = 0  # state 2
R[1][2] = -10
R[1][4] = 0
R[2][1] = 0  # state 3
R[2][5] = 100
R[3][0] = 0  # state 4
R[3][4] = 0
R[4][1] = 0  # state 5
R[4][3] = 0
R[4][5] = 100
R[5][2] = -10  # state 6
R[5][4] = 0
R[5][5] = 100


def update_q(state, action, df=0.9, lr=1):
    '''
    Method to update the Q value for a given state action pair.
    The discount factor is set to 0.9 and the learning rate to 1
    '''
    result = round(Q[state][action] + df * (R[state][action] + lr * max(Q[action]) - Q[state][action]), 1)
    print('\nQ[{}][{}] has been updated to {}'.format(state, action, result))
    Q[state][action] = result


def print_q():
    print('\nnew Q matrix:')
    for i in Q:
        print(i)


print("\n\n----- Task 1, initialize Q and R -----\nnew R matrix:")
for i in R:
    print(i)
print_q()

# 2)
print("\n\n----- Task 2, moving from room 4 to 5 -----")
update_q(3, 4)
print_q()

# 3)
print("\n\n----- Task 3, moving from room 5 to 6 -----")
update_q(4, 5)
print_q()

# 4)
print("\n\n----- Task 4, moving from room 4 to 5 -----")
update_q(3, 4)
print_q()

# 5)
print("\n\n----- Task 5, moving from room 1 to 4 -----")
update_q(0, 3)
print_q()

# 6)
print("\n\n----- Task 6, path to final state from room 2 -----")
current_state_list = [1]  # storing the current states in a list to print the path in the end
while current_state_list[-1] != 5:
    # making a list of neighbors from the current state, they're found by taking the entries from the R-matrix != -1
    neighbors = [i for i, value in enumerate(R[current_state_list[-1]]) if value != -1]

    # for each neighbor the Q-matrix is updated
    for n in neighbors:
        update_q(current_state_list[-1], n)

    # we go to the neighboring state with the maximum value in the Q-matrix
    current_state_list.append(Q[current_state_list[-1]].index(max(Q[current_state_list[-1]])))

print_q()
print("\nThe path taken from room 2 to room 6:", [z+1 for z in current_state_list])
