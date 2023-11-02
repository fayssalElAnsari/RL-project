import numpy as np
import gym
import matplotlib.pyplot as plt
import time

# Hyperparameters
lr = 0.1
gamma = 0.95
num_episodes = 20000
epsilon_decay = 0.999

# Environment setup
env = gym.make('FrozenLake-v1', is_slippery=False)

# Initialize Q-tables for Double Q-Learning
Q1 = np.zeros([env.observation_space.n, env.action_space.n])
Q2 = np.zeros([env.observation_space.n, env.action_space.n])

# Lists to keep track of metrics
rewards_list = []
steps_list = []
success_list = []

# Start time
start_time = time.time()

# Training process
for i in range(num_episodes):
    state = env.reset()
    total_reward = 0
    steps = 0
    done = False
    epsilon = 1.0 / (i / 8000 + 1)

    while not done:
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q1[state, :] + Q2[state, :])

        # Take action and observe reward and next state
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        steps += 1

        # Update rule for Double Q-Learning
        if np.random.rand() < 0.5:
            # Update Q1
            best_next_action = np.argmax(Q1[next_state, :])
            td_target = reward + gamma * Q2[next_state, best_next_action]
            td_error = td_target - Q1[state, action]
            Q1[state, action] += lr * td_error
        else:
            # Update Q2
            best_next_action = np.argmax(Q2[next_state, :])
            td_target = reward + gamma * Q1[next_state, best_next_action]
            td_error = td_target - Q2[state, action]
            Q2[state, action] += lr * td_error

        state = next_state

    # Append episode metrics to the lists
    rewards_list.append(total_reward)
    steps_list.append(steps)
    success_list.append(1 if total_reward > 0 else 0)

# Calculate training time
end_time = time.time()
training_time = end_time - start_time

# Post-training test of the policy
def test_policy(Q1, Q2, num_tests=100, verbose=False):
    success_count = 0
    for test in range(num_tests):
        state = env.reset()
        done = False
        while not done:
            action = np.argmax(Q1[state, :] + Q2[state, :])
            state, reward, done, _ = env.step(action)
            if done:
                if reward == 1.0:  # Success
                    success_count += 1
                break
        if verbose:
            print(f"Test {test+1}: {'Success' if reward == 1.0 else 'Fail'}")
    return success_count / num_tests

# Adjust the verbosity to True if you want to see the outcome of each test
post_training_success_rate = test_policy(Q1, Q2, verbose=False)

print('----------------------------------------------------------')
print("Overall Average reward:", np.mean(rewards_list))
print("Overall Average number of steps:", np.mean(steps_list))
print("Success rate (%):", np.mean(success_list) * 100)
print('Post-Training Success rate (%):', post_training_success_rate * 100)
print('Training Time (seconds):', training_time)
