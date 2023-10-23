import gym

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1")

# Define the new reward value for transitions that meet the condition
new_reward_value = -1.0

# Create a reward map that copies the original reward structure
reward_map = {}
for state in range(env.observation_space.n):
    reward_map[state] = {}
    for action in range(env.action_space.n):
        reward_map[state][action] = [] # Initialize with the original rewards

        for transition in env.P[state][action]:
            if transition[2] < 0.5 and transition[3]:  # Check if it's not the goal point and terminates
                # print(transition[0], transition[1])
                # print(transition[2], transition[3])
                # print(state,action)
                n = 2  # Replace the nth value (0-based index)

                # Create a new tuple with the nth value changed
                new_value = -1
                new_transition = transition[:n] + (new_value,) + transition[n + 1:]
                reward_map[state][action].append(new_transition)
            else:
                reward_map[state][action].append(transition)

                # print(transition)

# Now you can create a custom environment using the reward map
custom_env = gym.make("FrozenLake-v1", is_slippery=True)  # Create a custom environment with no random transitions
print(env.P)
# Modify the reward structure of the custom environment using the reward map
custom_env.unwrapped.P = reward_map

# Use the custom environment with the modified reward structure
print(custom_env.P)