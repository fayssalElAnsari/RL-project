###########################################################################
# This file contains the code for the Q-Learning agent for the FrozenLake-v1 environment
###########################################################################

# Importing Libraries
import time
import argparse
import numpy as np
import gym
import matplotlib.pyplot as plt

# Define command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
parser.add_argument('--num_episodes', type=int, default=100000, help='number of episodes')
args = parser.parse_args()

# Set the learning rate, discount factor, and number of episodes
lr = args.lr
gamma = args.gamma
num_episodes = args.num_episodes

negative_reward_enabled = True
is_slippery_enabled = True

# early stopping
average_reward_threshold = 0.75
consecutive_episodes = 100  # Number of episodes to consider for moving average
moving_average_rewards = []

custom_map = [
    'SFFF',
    'FHFF',
    'FFHF',
    'HFGF'
]

# Create the environment
env = gym.make('FrozenLake-v1', desc=custom_map, is_slippery=is_slippery_enabled)

# Initialize the Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

# lists to store performance metrics
total_rewards = []
total_steps = []
success_rate = []
test_reward = []

# Hyperparameters for Delayed Q-learning
delay_step = 10  # Update the Q-table every delay_step steps

start_time = time.time()

episodes_to_train = num_episodes

# Run the delayed Q-learning algorithm
for i in range(num_episodes):
    s = env.reset()
    done = False
    episode_reward = 0
    num_steps = 0
    steps_since_update = 0  # Track the number of steps since the last Q-table update

    # Store the experiences for delayed update
    experiences = []

    while not done:
        a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        s_, r, done, _ = env.step(a)

        if negative_reward_enabled:
            if done and r == 0:  # The agent fell into a hole and not at the goal state
                r = -1  # Negative reward for falling into a hole

        experiences.append((s, a, r, s_))  # Store the transition

        s = s_
        episode_reward += r
        num_steps += 1
        steps_since_update += 1

        # Perform a delayed update of the Q-table
        if steps_since_update >= delay_step or done:
            for exp in experiences:
                state, action, reward, next_state = exp
                Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
            steps_since_update = 0
            experiences = []
    
    # append the total reward for the episode, not the average per step
    total_rewards.append(episode_reward)
    total_steps.append(num_steps)
    success_rate.append(int(episode_reward > 0))

    # Print episode metrics
    print("Episode:", i+1, "Reward:", episode_reward, "Steps:", num_steps)

    # Calculate moving average reward
    if i >= consecutive_episodes:
        moving_avg_reward = np.mean(total_rewards[-consecutive_episodes:])
        moving_average_rewards.append(moving_avg_reward)
        
        # Check if the moving average reward exceeds the threshold
        if moving_avg_reward >= average_reward_threshold:
            print(f"Early stopping: Episode {i+1}, Moving Average Reward: {moving_avg_reward}")
            episodes_to_train = i + 1
            break


end_time = time.time()
training_time = end_time - start_time

# To test the policy after training and record steps when winning
def test_policy(env, Q, num_tests=100):
    success_count = 0
    step_list = []  # To store the number of steps for successful episodes
    
    for _ in range(num_tests):
        s = env.reset()
        done = False
        step_count = 0  # Reset step counter at the start of each test
        
        while not done:
            a = np.argmax(Q[s, :])  # Now, we use the max Q value without randomness
            s, r, done, _ = env.step(a)
            step_count += 1  # Increment step counter regardless of outcome
            
            if r == 1:  # Assuming a reward of 1 indicates success
                success_count += 1
                step_list.append(step_count)  # Only append steps for successful episodes
                
    success_rate = success_count / num_tests
    average_steps_when_winning = np.mean(step_list) if step_list else 0  # Avoid division by zero
    
    return success_rate, average_steps_when_winning

post_training_success_rate, average_steps_when_winning = test_policy(env, Q)

# Print overall metrics
print('----------------------------------------------------------')
print("Overall Average reward:", np.mean(total_rewards))
print("Overall Average number of steps:", np.mean(total_steps))
print("Success rate (%):", np.mean(success_rate)*100)


print('')
print('Delayed Q-Learning with Negative reward: ', negative_reward_enabled, '; and slippery: ', is_slippery_enabled)
print('==========================================================')
print('The number of episodes', episodes_to_train)
print('Post-Training Success rate (%):', post_training_success_rate * 100)
print('Average number of steps when winning:', average_steps_when_winning)
print('Training Time (seconds):', training_time)
print('==========================================================')

# Plotting the averge rewards and episode Lengths gained throughout each episode per episode
fig, axs = plt.subplots(1, 2, figsize=(200, 5))
axs[0].plot(total_rewards, 'tab:green')
axs[0].set_title('Reward per Episode')
axs[1].plot(total_steps, 'tab:purple')
axs[1].set_title('number of steps taken per episode')

plt.show()