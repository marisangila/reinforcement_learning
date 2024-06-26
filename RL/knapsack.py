from random import randint
import numpy as np

class RL:

    def __init__(self, items, capacities):
        self.items = items
        self.capacities = capacities
        self.num_items = len(items)
        self.num_knapsacks = len(capacities)
        self.state_shape = (self.num_items, self.num_knapsacks)
        self.q_table = np.zeros((self.num_items, self.num_knapsacks, 2))  # Q-table initialization with 2 possible actions (0: don't pick, 1: pick)
        
        self.episode = 1000  # Number of training episodes
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Epsilon-greedy parameter

    def _walk(self, training=True):
        state = np.zeros(self.state_shape, dtype=int)  # Start with empty knapsacks
        current_item = 0

        while current_item < self.num_items:
            item_value, item_weight = self.items[current_item]
            
            # Epsilon-greedy action selection
            if training and np.random.uniform(0, 1) < self.epsilon:
                action = randint(0, 1)  # Random action during training
            else:
                action = np.argmax(self.q_table[current_item])

            if action == 1:  # Try to pick the item
                chosen_knapsack = np.argmax(self.capacities - np.sum(state * self.items[:, 1], axis=0))  # Choose the knapsack with the most remaining capacity
                if self.capacities[chosen_knapsack] >= item_weight:
                    state[current_item, chosen_knapsack] = 1  # Pick the item

            if training:
                reward = np.sum(state[:, chosen_knapsack] * self.items[:, 0])  # Reward is the total value of items in the knapsack
                next_state = state.copy()
                self.q_table[current_item][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[current_item + 1]) - self.q_table[current_item][action])

            current_item += 1

        return state

    def _view(self, state):
        print("STATUS:")
        for knapsack in range(self.num_knapsacks):
            print(f"Knapsack {knapsack + 1}:")
            for item in range(self.num_items):
                if state[item, knapsack] == 1:
                    print(f"  Item {item + 1} (value: {self.items[item][0]}, weight: {self.items[item][1]})")
        print("")

    def train(self):
        for i in range(self.episode):
            self._walk(training=True)
        print("END TRAIN!")

    def test(self):
        state = self._walk(training=False)
        print("COMPLETE!")
        self._view(state)

if __name__ == '__main__':
    items = [(10, 2), (5, 3), (15, 5), (7, 7)]  # Example items (value, weight)
    capacities = [10, 10]  # Example knapsacks capacities
    rl = RL(items, capacities)
    rl.train()
    rl.test()
