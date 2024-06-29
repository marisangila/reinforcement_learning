import numpy as np
import random
import matplotlib.pyplot as plt


class MultipleKnapsackRL:
    def __init__(self, values, weights, capacities, episodes=1000, alpha=0.15, gamma=0.75, epsilon=0.01, epsilon_decay=0.99):
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
        self.q_table = {}
        self.rewards_per_episode = []
        self.q_values_per_episode = []
        self.epsilon_history = []
        self.td_errors_per_episode = []
        self.max_value_per_episode = []

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
        new_state[item] = knapsack + 1
        return new_state

    def get_reward(self, state):
        return sum(self.values[i] if state[i] != 0 else 0 for i in range(self.num_items))

    def train(self):
        for episode in range(self.episodes):
            state = [0] * self.num_items
            total_reward = 0
            total_td_error = 0
            num_td_updates = 0
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
            self.max_value_per_episode.append(max(self.max_value_per_episode[-1], total_reward) if self.max_value_per_episode else total_reward)
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

    def test(self, test_values, test_weights, test_capacities):
        self.values = test_values
        self.weights = test_weights
        self.capacities = test_capacities
        self.num_items = len(test_values)
        self.num_knapsacks = len(test_capacities)
        
        state = [0] * self.num_items
        while 0 in state:
            possible_actions = self.get_possible_actions(state)
            current_state_index = self.state_to_index(state)
            if current_state_index not in self.q_table:
                self.q_table[current_state_index] = np.zeros(self.num_items * self.num_knapsacks)
            q_values = self.q_table[current_state_index]
            valid_q_values = [q_values[action] if self.is_valid_action(state, action) else -np.inf for action in possible_actions]
            action = possible_actions[np.argmax(valid_q_values)]
            if self.is_valid_action(state, action):
                state = self.take_action(state, action)

        total_value = self.get_reward(state)
        knapsacks = [[] for _ in range(self.num_knapsacks)]
        for i in range(self.num_items):
            if state[i] != 0:
                knapsacks[state[i] - 1].append(i)

        return total_value, knapsacks

if __name__ == '__main__':
    # Instância de treinamento
    train_values = [    1898,	 440,  22507,	270,  14148,   3100,   4650,  30800,    615,	 4975,
                        1160,	4225,	510,   11880,    479,    440,    490,    330,    110,	  560,
                        24355,	2885,  11748,    4550,    750,   3720,   1950,  10500]
    train_weights = [     45,	   0,	 85,	150,	65,     95,     30,      0,    170,	    0,
   40,	  25,	 20,	  0,	 0,     25,      0,      0,     25,	    0,
  165,	   0,	 85	,  0,	 0,	0,     0,    100, 	
   30,	  20,	125,	  5,	80,     25,     35,     73,     12,	   15,
   15,	  40,	  5,	 10,	10,     12,     10,      9,      0,	   20,
   60,	  40,	 50,	 36,	49,     40,     19,    150]
    train_capacities = [600, 600] 

    knapsack_rl = MultipleKnapsackRL(train_values, train_weights, train_capacities)
    train_max_value, train_best_combination = knapsack_rl.train()

    print(f"Training Maximum value: {train_max_value}")
    for i, knapsack in enumerate(train_best_combination):
        print(f"Training Best combination for knapsack {i + 1}: {knapsack}")

    # Instância de teste
    test_values = [    1898,	 440,  22507,	270,  14148,   3100,   4650,  30800,    615,	 4975,
                        1160,	4225,	510,   11880,    479,    440,    490,    330,    110,	  560,
                        24355,	2885,  11748,    4550,    750,   3720,   1950,  10500]
    test_weights = [     45,	   0,	 85,	150,	65,     95,     30,      0,    170,	    0,
   40,	  25,	 20,	  0,	 0,     25,      0,      0,     25,	    0,
  165,	   0,	 85	,  0,	 0,	0,     0,    100, 	
   30,	  20,	125,	  5,	80,     25,     35,     73,     12,	   15,
   15,	  40,	  5,	 10,	10,     12,     10,      9,      0,	   20,
   60,	  40,	 50,	 36,	49,     40,     19,    150] 	
    test_capacities = [600, 600]

    test_max_value, test_best_combination = knapsack_rl.test(test_values, test_weights, test_capacities)

    print(f"Testing Maximum value: {test_max_value}")
    for i, knapsack in enumerate(test_best_combination):
        print(f"Testing Best combination for knapsack {i + 1}: {knapsack}")

    # Visualizações
    episodes = range(knapsack_rl.episodes)
    
    # Recompensa acumulada por episódio
    plt.figure()
    plt.plot(episodes, knapsack_rl.rewards_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')
    plt.show()

    # Decaimento de epsilon
    plt.figure()
    plt.plot(episodes, knapsack_rl.epsilon_history)
    plt.xlabel('Episodes')
    plt.ylabel('Epsilon')
    plt.title('Epsilon Decay over Episodes')
    plt.show()

    # Valor máximo por episódio
    plt.figure()
    plt.plot(episodes, knapsack_rl.max_value_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Max Value')
    plt.title('Max Value per Episode')
    plt.show()

    # Valor médio de Q por episódio
    plt.figure()
    plt.plot(episodes, knapsack_rl.q_values_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Average Q-Value')
    plt.title('Average Q-Value per Episode')
    plt.show()

    # Erro TD médio por episódio
    plt.figure()
    plt.plot(episodes, knapsack_rl.td_errors_per_episode)
    plt.xlabel('Episodes')
    plt.ylabel('Average TD Error')
    plt.title('Average TD Error per Episode')
    plt.show()
