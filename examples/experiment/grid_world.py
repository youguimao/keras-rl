import numpy as np

WORLD_HEIGHT = 4
WORLD_WIDTH = 4

GOAL = [[0, 0], [3, 3]]
pip
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
ACTION_PROB = 0.25

def step(state, action):
    i, j = state
    if action == ACTION_UP:
        next_state = [max(i-1, 0), j]
    elif action == ACTION_DOWN:
        next_state = [min(i+1, WORLD_HEIGHT - 1), j]
    elif action == ACTION_LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        next_state = [i, min(j + 1, WORLD_WIDTH -1)]
    reward = -1

    if state in GOAL:
        next_state = state
        reward = 0
    return next_state, reward


def compute_state_value(discount = 1):
    new_state_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH))
    while True:
        state_value = np.copy(new_state_value)
        old_state_value = np.copy(state_value)

        for i in range(WORLD_HEIGHT):
            for j in range(WORLD_WIDTH):
                value = 0
                for action in ACTIONS:
                    next_state, reward = step([i, j], action)
                    value += ACTION_PROB * (reward + discount * state_value[next_state[0], next_state[1]])
                    new_state_value[i, j] = value

        max_delta = np.max(abs(old_state_value - new_state_value))
        if max_delta < 1e-4:
            break

    print('state value is:')
    for row in new_state_value:
        print(row)
    return new_state_value


def print_optimal_policy(state_value):
    optimal_policy = []
    for i in range(WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(WORLD_WIDTH):
            if [i, j] in GOAL:
                optimal_policy[-1].append('G')
                continue

            action_value = np.zeros((1, 4))
            for action in ACTIONS:
                next_state, reward = step([i, j], action)
                action_value[0, action] = state_value[next_state[0], next_state[1]]

            best_action = np.argmax(action_value)

            if best_action == ACTION_UP:
                optimal_policy[-1].append('U')
            elif best_action == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif best_action == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif best_action == ACTION_RIGHT:
                optimal_policy[-1].append('R')

    for row in optimal_policy:
        print(row)


if __name__ == '__main__':
    state_value = compute_state_value(1)
    print('Optimal policy from Policy iteration:')
    print_optimal_policy(state_value)