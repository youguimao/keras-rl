import numpy as np

WORLD_HEIGHT = 4
WORLD_WIDTH = 4

GOAL = [0, 0]

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

    if state == GOAL:
        next_state = state
        reward = 0

    return next_state, reward


def compute_state_value(discount = 1):
    new_state_value = np.zeros((WORLD_HEIGHT, WORLD_WIDTH))
    while True
        state_value = np.deepcopy(new_state_value)
        old_state_value = np.deepcopy(state_value)

        for i in range(WORLD_HEIGHT):
            for j in range(WORLD_WIDTH):
                value = 0
                for action in ACTIONS:
                    next_state, reward = step([i, j], action)
                    value += ACTION_PROB * (reward + discount * state_value[next_state[0], next_state[1]])
                    next_state_value[i, j] = value

        max_delta = np.max(abs(old_state_value - new_state_value))
        if max_delta < 1e-4:
            break

    return new_state_value