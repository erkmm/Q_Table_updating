import numpy as np

# Define the size of the state space and action space
state_size = 3  # Example: 10 states
action_size = 2  # Example: 4 actions

# Initialize Q-table with values from the list
Q_table_list = [(0.5, -0.2), (-0.1, 0.4), (0.2, 0.0)]
Q_table = np.array(Q_table_list)

def update_Q_table(state, action, reward, next_state, Q_table):
    # Q-value update formula
    Q_table[state, action] = Q_table[state, action] + 0.1 * (reward + 0.9 * np.max(Q_table[next_state, :]) - Q_table[state, action])

# Example of using the update function
#%%Traj1
update_Q_table(0, 0, 1, 1, Q_table)

update_Q_table(1, 1, 2, 2, Q_table)

update_Q_table(2, 0, 3, 0, Q_table)
print(Q_table)
print()

#%%
update_Q_table(0, 1, 0, 1, Q_table)

update_Q_table(1, 0, -1, 2, Q_table)

update_Q_table(2, 1, 4, 0, Q_table)
print(Q_table)
print()

#%%
update_Q_table(1, 1, 2, 2, Q_table)

update_Q_table(2, 0, 0, 0, Q_table)

update_Q_table(0, 0, 1, 1, Q_table)
print(Q_table)
print()

#%%
update_Q_table(2, 1, -2, 0, Q_table)

update_Q_table(1, 0, 3, 2, Q_table)

update_Q_table(0, 1, 0, 1, Q_table)
print(Q_table)




