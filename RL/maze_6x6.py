from random import randint
import numpy as np

class RL:

    def __init__(self):
        self.env = np.array([[2, 0, 1, 1, 1, 1],
                             [0, 0, 0, 1, 0, 1],
                             [0, 0, 0, 0, 0, 1],
                             [1, 0, 1, 1, 1, 1],
                             [1, 0, 0, 0, 0, 1],
                             [1, 1, 1, 1, 0, 0]])
        
        self.actions = [(0, -1), (0, 1), (-1, 0), (1, 0)] # Left | Right | Up | Down

        self.episode = 1000  # Number of training episodes
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.9  # Discount factor
        self.epsilon = 0.1  # Epsilon-greedy parameter

        self.row, self.columns = self.env.shape
        self.q_table = np.zeros((self.row, self.columns, 4))  # Q-table initialization with 4 possible actions

    def _walk(self, training=True):
        i, j = np.where(self.env == 2)
        current_state = (i[0], j[0])

        # Epsilon-greedy action selection
        if training and np.random.uniform(0, 1) < self.epsilon:
            action = randint(0, 3)  # Random action during training
        else:
            action = np.argmax(self.q_table[current_state])

        action_row, action_column = self.actions[action]

        new_row = current_state[0] + action_row
        new_column = current_state[1] + action_column

        if 0 <= new_row < self.row and 0 <= new_column < self.columns and self.env[new_row][new_column] != 1:
            self.env[new_row][new_column] = 2
            self.env[current_state] = 0

            if training:
                # Q-learning update
                reward = -1 if self.env[5][5] != 2 else 100  # Reward structure
                next_state = (new_row, new_column)
                self.q_table[current_state][action] += self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[current_state][action])

    def _out(self):
        return self.env[5][5] == 2

    def _view(self):
        print("STATUS:")
        for i in range(self.row):
            for j in range(self.columns):
                if self.env[i][j] == 0:
                    print("[   ]", end="")
                elif self.env[i][j] == 1:
                    print("[ X ]", end="")
                elif self.env[i][j] == 2:
                    print("[ R ]", end=" ")
            print("")
        print("")

    def _reset(self):
        self.env = np.array([[2, 0, 1, 1, 1, 1],
                             [0, 0, 0, 1, 0, 1],
                             [0, 0, 0, 0, 0, 1],
                             [1, 0, 1, 1, 1, 1],
                             [1, 0, 0, 0, 0, 1],
                             [1, 1, 1, 1, 0, 0]])

    def train(self):
        for i in range(self.episode):
            while not self._out():
                self._walk(training=True)
            self._reset()
        print("END TRAIN!")

    def test(self):
        self._reset()
        while not self._out():
            self._view()
            self._walk(training=False)
        print("COMPLETE!")
        self._view()

if __name__ == '__main__':
    rl = RL()
    rl.train()
    rl.test()
