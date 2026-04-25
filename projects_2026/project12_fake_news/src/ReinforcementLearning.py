#divya bharathi

import numpy as np
import random

from TrainAndEvaluateClassifier import TrainAndEvaluateClassifer
from TrainAndEvaluateLSTM import TrainAndEvaluateLSTM

# PARAMETERS
ACTIONS = [-20, 0, 20]   # decrease, stay, increase
MIN_ARTICLES = 0
MAX_ARTICLES = 500

EPISODES = 5
ALPHA = 0.1   # learning rate
GAMMA = 0.9   # discount factor
EPSILON = 0.3 # exploration
# Q-TABLE
Q = {}

def get_q(state, action):
    return Q.get((state, action), 0)

def set_q(state, action, value):
    Q[(state, action)] = value

# ENV STEP
def step(state, action):
    new_state = state + action
    new_state = max(MIN_ARTICLES, min(MAX_ARTICLES, new_state))

    # Run model
    data = TrainAndEvaluateClassifer(new_state, 1)
    accuracy = TrainAndEvaluateLSTM(data)

    reward = accuracy

    return new_state, reward

# ENV STEP for reduced training data set
def step_reduced_set(state, action):
    new_state = state + action
    new_state = max(MIN_ARTICLES, min(MAX_ARTICLES, new_state))

    # Run model
    data = TrainAndEvaluateClassifer(new_state, 0,400,800)
    accuracy = TrainAndEvaluateLSTM(data)

    reward = accuracy

    return new_state, reward

# TRAIN Q-LEARNING
def train_q_learning():
    state = 150  # start with 50 synthetic articles

    for episode in range(EPISODES):
        print(f"\n=== Episode {episode+1} ===")

        for step_num in range(5):  # steps per episode

            # ε-greedy action selection
            if random.random() < EPSILON:
                action = random.choice(ACTIONS)
            else:
                q_vals = [get_q(state, a) for a in ACTIONS]
                action = ACTIONS[np.argmax(q_vals)]

            next_state, reward = step(state, action)

            # Q-learning update
            old_q = get_q(state, action)
            future_q = max([get_q(next_state, a) for a in ACTIONS])

            new_q = old_q + ALPHA * (reward + GAMMA * future_q - old_q)
            set_q(state, action, new_q)

            print(f"State: {state}, Action: {action}, Reward: {reward:.4f}")

            state = next_state

    print("\n=== Training Complete ===")

    # Find best state
    best_state = max(Q, key=lambda x: Q[x])[0]
    print(f"Best synthetic article count: {best_state}")

    return best_state

# TRAIN Q-LEARNING with reduced training data set.
def train_q_learning_reduced_set():
    state = 150  # start with 50 synthetic articles

    for episode in range(EPISODES):
        print(f"\n=== Episode {episode+1} ===")

        for step_num in range(5):  # steps per episode

            # ε-greedy action selection
            if random.random() < EPSILON:
                action = random.choice(ACTIONS)
            else:
                q_vals = [get_q(state, a) for a in ACTIONS]
                action = ACTIONS[np.argmax(q_vals)]

            next_state, reward = step_reduced_set(state, action)

            # Q-learning update
            old_q = get_q(state, action)
            future_q = max([get_q(next_state, a) for a in ACTIONS])

            new_q = old_q + ALPHA * (reward + GAMMA * future_q - old_q)
            set_q(state, action, new_q)

            print(f"State: {state}, Action: {action}, Reward: {reward:.4f}")

            state = next_state

    print("\n=== Training Complete ===")

    # Find best state
    best_state = max(Q, key=lambda x: Q[x])[0]
    print(f"Best synthetic article count: {best_state}")

    return best_state

