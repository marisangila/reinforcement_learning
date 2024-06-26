from random import randint
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

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
        self.action_size = len(self.actions)
        self.state_shape = (self.row * self.columns,)
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_dim=self.row * self.columns, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.alpha))
        return model

    def _get_state(self):
        return np.reshape(self.env.flatten(), [1, self.row * self.columns])

    def _walk(self, training=True):
        state = self._get_state()
        if training and np.random.uniform(0, 1) < self.epsilon:
            action = randint(0, 3)  # Random action during training
        else:
            q_values = self.model.predict(state)
            action = np.argmax(q_values[0])

        action_row, action_column = self.actions[action]

        i, j = np.where(self.env == 2)
        current_state = (i[0], j[0])

        new_row = current_state[0] + action_row
        new_column = current_state[1] + action_column

        if 0 <= new_row < self.row and 0 <= new_column < self.columns and self.env[new_row][new_column] != 1:
            self.env[new_row][new_column] = 2
            self.env[current_state] = 0
            next_state = self._get_state()
            reward = -1
            done = False
            if self.env[5][5] == 2:
                reward = 100
                done = True

            if training:
                target = reward
                if not done:
                    next_q_values = self.model.predict(next_state)
                    target = reward + self.gamma * np.amax(next_q_values[0])
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)

            if done:
                self._reset()
            return done

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
            done = False
            while not done:
                done = self._walk(training=True)
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
