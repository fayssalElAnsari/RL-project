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

# Create the environment
env = gym.make('FrozenLake-v1', is_slippery=True)

# Initialize the Q-table
Q = np.zeros([env.observation_space.n, env.action_space.n])

# lists to store performance metrics
total_rewards = []
total_steps = []
success_rate = []
test_reward = []

# early stopping
average_reward_threshold = 0.2
consecutive_episodes = 200  # Number of episodes to consider for moving average
moving_average_rewards = []

start_time = time.time()

# Run the Q-learning algorithm
for i in range(num_episodes):
    s = env.reset()
    done = False
    episode_reward = 0
    num_steps = 0
    
    # The Q-Table learning algorithm
    while not done:
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))
        s_, r, done, _ = env.step(a)
        
        # update Q values
        Q[s,a] = Q[s,a] + lr*(r + gamma*np.max(Q[s_,:]) - Q[s,a])
        s = s_
        episode_reward += r
        num_steps += 1
    
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
            break


end_time = time.time()
training_time = end_time - start_time

# To test the policy after training
def test_policy(env, Q, num_tests=100):
    success_count = 0
    for _ in range(num_tests):
        s = env.reset()
        done = False
        while not done:
            a = np.argmax(Q[s, :])  # Now, we use the max Q value without randomness
            s, r, done, _ = env.step(a)
            if r == 1:  # Assuming a reward of 1 indicates success
                success_count += 1
    return success_count / num_tests

post_training_success_rate = test_policy(env, Q)

# Print overall metrics
print('----------------------------------------------------------')
print("Overall Average reward:", np.mean(total_rewards))
print("Overall Average number of steps:", np.mean(total_steps))
print("Success rate (%):", np.mean(success_rate)*100)

# Print the post-training success rate
print('Post-Training Success rate (%):', post_training_success_rate * 100)

# Print the computational efficiency
print('Training Time (seconds):', training_time)

# Function to visualize the agent playing
def play_game(env, Q):
    s = env.reset()
    env.render()
    done = False
    while not done:
        a = np.argmax(Q[s, :])
        s, r, done, _ = env.step(a)
        env.render()
        if done:
            if r == 1:
                print("The agent reached the goal!")
            else:
                print("The agent fell into a hole!")
            time.sleep(2)
    env.close()

# Call the play_game function to visualize the agent playing
play_game(env, Q)




# Plotting the averge rewards and episode Lengths gained throughout each episode per episode
fig, axs = plt.subplots(1, 2, figsize=(200, 5))
axs[0].plot(total_rewards, 'tab:green')
axs[0].set_title('Reward per Episode')
axs[1].plot(total_steps, 'tab:purple')
axs[1].set_title('number of steps taken per episode')

plt.show()