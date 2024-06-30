import numpy as np
import random
import matplotlib.pyplot as plt
import datetime
import time
import os

class MultipleKnapsackRL:
    def __init__(self, values, weights, capacities, episodes=10000, alpha=0.1, gamma=0.9, epsilon=0.01, epsilon_decay=0.99, epochs=30):
        # Initialize attributes
        self.values = values
        self.weights = weights
        self.capacities = capacities
        self.num_items = len(values)
        self.num_knapsacks = len(capacities)
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epochs = epochs
        self.q_table = {}
        # Initialize lists to store metrics
        self.rewards_per_episode = []
        self.q_values_per_episode = []
        self.epsilon_history = []
        self.td_errors_per_episode = []
        self.max_value_per_episode = []
        # Lists to store metrics across epochs
        self.rewards_per_episode_epoch = []  
        self.q_values_per_episode_epoch = [] 
        self.epsilon_history_epoch = [] 
        self.td_errors_per_episode_epoch = [] 
        self.max_value_per_episode_epoch = []

    def state_to_index(self, state):
        return tuple(state)

    def get_possible_actions(self, state):
        possible_actions = []
        for item in range(self.num_items):
            if state[item] == 0:  # Item is not yet in any knapsack
                for knapsack in range(self.num_knapsacks):
                    possible_actions.append(item * self.num_knapsacks + knapsack)
        return possible_actions

    def is_valid_action(self, state, action):
        item = action // self.num_knapsacks
        knapsack = action % self.num_knapsacks
        if state[item] != 0:
            return False
        current_weight = sum(self.weights[i] for i in range(self.num_items) if state[i] == knapsack + 1)
        return current_weight + self.weights[item] <= self.capacities[knapsack]

    def take_action(self, state, action):
        new_state = state[:]
        item = action // self.num_knapsacks
        knapsack = action % self.num_knapsacks
        # Check if the item is already allocated in another knapsack
        for k in range(self.num_knapsacks):
            if state[item] == k + 1:
                new_state[item] = 0  # Remove from the current knapsack before reallocating
        new_state[item] = knapsack + 1  # Allocate the item to the new knapsack
        return new_state

    def get_reward(self, state):
        return sum(self.values[i] if state[i] != 0 else 0 for i in range(self.num_items))

    def train(self):
        for episode in range(self.episodes):
            state = [0] * self.num_items
            total_reward = 0
            total_td_error = 0
            num_td_updates = 0
            train_max_value = 0  # Track max value for the last episode of this epoch
            while 0 in state:
                possible_actions = self.get_possible_actions(state)
                if random.uniform(0, 1) < self.epsilon:
                    action = random.choice(possible_actions)
                else:
                    current_state_index = self.state_to_index(state)
                    if current_state_index not in self.q_table:
                        self.q_table[current_state_index] = np.zeros(self.num_items * self.num_knapsacks)
                    q_values = self.q_table[current_state_index]
                    valid_q_values = [q_values[action] if self.is_valid_action(state, action) else -np.inf for action in possible_actions]
                    action = possible_actions[np.argmax(valid_q_values)]

                if self.is_valid_action(state, action):
                    new_state = self.take_action(state, action)
                    reward = self.get_reward(new_state)
                    total_reward += reward
                    current_state_index = self.state_to_index(state)
                    new_state_index = self.state_to_index(new_state)
                    if new_state_index not in self.q_table:
                        self.q_table[new_state_index] = np.zeros(self.num_items * self.num_knapsacks)
                    best_future_q = np.max(self.q_table[new_state_index])
                    td_error = reward + self.gamma * best_future_q - self.q_table[current_state_index][action]
                    self.q_table[current_state_index][action] += self.alpha * td_error
                    total_td_error += abs(td_error)
                    num_td_updates += 1
                    state = new_state

            self.rewards_per_episode.append(total_reward)
            train_max_value = total_reward  # Update with the reward of the last episode
            self.max_value_per_episode.append(train_max_value)
            self.epsilon_history.append(self.epsilon)
            self.epsilon *= self.epsilon_decay
            avg_td_error = total_td_error / num_td_updates if num_td_updates > 0 else 0
            self.td_errors_per_episode.append(avg_td_error)

            q_values = [np.mean(self.q_table[state]) for state in self.q_table]
            avg_q_value = np.mean(q_values) if q_values else 0
            self.q_values_per_episode.append(avg_q_value)

        total_value = self.get_reward(state)
        knapsacks = [[] for _ in range(self.num_knapsacks)]
        for i in range(self.num_items):
            if state[i] != 0:
                knapsacks[state[i] - 1].append(i)

        return total_value, knapsacks
    
    def append_epochs(self):
        # Append the metrics from current episode to epoch lists
        self.rewards_per_episode_epoch.append(self.rewards_per_episode.copy())
        self.q_values_per_episode_epoch.append(self.q_values_per_episode.copy())
        self.epsilon_history_epoch.append(self.epsilon_history.copy())
        self.td_errors_per_episode_epoch.append(self.td_errors_per_episode.copy())
        self.max_value_per_episode_epoch.append(self.max_value_per_episode.copy())
        
        # Clear lists for current episode metrics
        self.rewards_per_episode.clear()
        self.q_values_per_episode.clear()
        self.epsilon_history.clear()
        self.td_errors_per_episode.clear()
        self.max_value_per_episode.clear()
        
    def avg_epochs(self):
        # Calculate average across epochs
        self.rewards_per_episode_epoch = np.mean(self.rewards_per_episode_epoch, axis=0)
        self.q_values_per_episode_epoch = np.mean(self.q_values_per_episode_epoch, axis=0)
        self.epsilon_history_epoch = np.mean(self.epsilon_history_epoch, axis=0)
        self.td_errors_per_episode_epoch = np.mean(self.td_errors_per_episode_epoch, axis=0)
        self.max_value_per_episode_epoch = np.mean(self.max_value_per_episode_epoch, axis=0)
    
    def save_max_values(self, file_path):
        # Calculate average of last epochs' max values
        last_epochs_max_values = self.max_value_per_episode_epoch[-self.epochs:]
        avg_last_epochs_max_value = np.mean(last_epochs_max_values)

        # Save average value to file
        with open(file_path, 'w') as file:
            file.write(f"Average Max Value of Last {self.epochs} Epochs: {avg_last_epochs_max_value}\n")
        

    def plot_and_save(self, metric_name, metric_data):
        # Normalize data
        normalized_data = (metric_data - np.min(metric_data)) / (np.max(metric_data) - np.min(metric_data))

        # Visualizations
        episodes = range(self.episodes)
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, normalized_data)
        plt.xlabel('Episodes')
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} per Episode')

        # Save to PDF
        pdf_filename = f'knapsack_rl_{metric_name}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.pdf'
        plt.savefig(pdf_filename)
        print(f"Plot saved to {pdf_filename}")

        # Close the figure to release memory
        plt.close()

        return pdf_filename

    def plot_all_metrics(self):
        # Plot and save each metric individually
        pdf_filenames = []
        pdf_filenames.append(self.plot_and_save('Total_Reward', self.rewards_per_episode_epoch))
        pdf_filenames.append(self.plot_and_save('Epsilon', self.epsilon_history_epoch))
        pdf_filenames.append(self.plot_and_save('Max_Value', self.max_value_per_episode_epoch))
        pdf_filenames.append(self.plot_and_save('Average_Q_Value', self.q_values_per_episode_epoch))

        return pdf_filenames

def read_dat_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        values = list(map(int, lines[1].split()))
        weights = list(map(int, lines[2].split()))
        capacities = list(map(int, lines[3].split()))

        return values, weights, capacities

    
if __name__ == '__main__':
    # List of dataset file paths
    dataset_files = ["weing1.dat"] 
    base_dir = r"C:\Users\Mari\Documents\Github\reinforcement_learning\All-MKP-Instances\sac94\weing"
    for file_name in dataset_files:
        # Read dataset from file
        
        file_path = os.path.join(base_dir, file_name)
        train_values, train_weights, train_capacities = read_dat_file(file_path)
        
        # Initialize the RL agent
        knapsack_rl = MultipleKnapsackRL(train_values, train_weights, train_capacities)
        
        # Train the agent for multiple epochs
        start_time = time.time()
        for e in range(knapsack_rl.epochs):
            print(f"Epoch: {e + 1}/{knapsack_rl.epochs}")
            train_max_value, train_best_combination = knapsack_rl.train()

            print(f"Training Maximum value: {train_max_value}")
            for i, knapsack in enumerate(train_best_combination):
                print(f"Training Best combination for knapsack {i + 1}: {knapsack}")

            # Append metrics for the current epoch
            knapsack_rl.append_epochs()

        # Calculate averages across epochs
        knapsack_rl.avg_epochs()

        # Print end time and runtime
        end_time = time.time()
        print("END", datetime.datetime.now())
        print(f"Runtime: {end_time - start_time:.6f} seconds")

        # Print the dataset information
        print(f"Dataset values: {train_values}")
        print(f"Dataset weights: {train_weights}")
        print(f"Dataset capacities: {train_capacities}")

        # Save max values to a text file
        max_values_file = f'max_values_{file_name.split(".")[0]}.txt'
        knapsack_rl.save_max_values(max_values_file)
        print(f"Max values saved to {max_values_file}")

        # Plot results for the dataset and save each metric to PDF
        pdf_filenames = knapsack_rl.plot_all_metrics()
        print(f"PDFs saved: {pdf_filenames}")
