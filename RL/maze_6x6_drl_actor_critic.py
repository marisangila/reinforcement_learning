import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

class ActorCriticAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = 0.95    # discount rate
        self.learning_rate = 0.001
        self.actor = self._build_actor()
        self.critic = self._build_critic()

    def _build_actor(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_shape=self.state_shape, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def _build_critic(self):
        model = models.Sequential()
        model.add(layers.Dense(24, input_shape=self.state_shape, activation='relu'))
        model.add(layers.Dense(24, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        model.compile(loss='mse', optimizer=optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        target = reward
        if not done:
            target = reward + self.gamma * self.critic.predict(next_state)[0]
        target_f = self.critic.predict(state)
        target_f[0] = target
        self.critic.fit(state, target_f, epochs=1, verbose=0)

        advantages = np.zeros((1, self.action_size))
        advantages[0][action] = target - self.critic.predict(state)[0]
        self.actor.fit(state, advantages, epochs=1, verbose=0)

    def act(self, state):
        policy = self.actor.predict(state)[0]
        return np.random.choice(self.action_size, p=policy)

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
        self.row, self.columns = self.env.shape
        self.action_size = len(self.actions)
        self.state_shape = (self.row * self.columns,)
        self.agent = ActorCriticAgent(self.state_shape, self.action_size)

    def _get_state(self):
        return np.reshape(self.env.flatten(), [1, self.row * self.columns])

    def _walk(self, training=True):
        state = self._get_state()
        action = self.agent.act(state)
        acao_linha, acao_coluna = self.actions[action]

        i, j = np.where(self.env == 2)
        current_state = (i[0], j[0])

        nova_linha = current_state[0] + acao_linha
        nova_coluna = current_state[1] + acao_coluna

        if 0 <= nova_linha < self.row and 0 <= nova_coluna < self.columns and self.env[nova_linha][nova_coluna] != 1:
            self.env[nova_linha][nova_coluna] = 2
            self.env[current_state] = 0
            next_state = self._get_state()
            reward = -1
            done = False
            if self.env[5][5] == 2:
                reward = 100
                done = True

            if training:
                self.agent.remember(state, action, reward, next_state, done)

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
        for e in range(self.episode):
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
