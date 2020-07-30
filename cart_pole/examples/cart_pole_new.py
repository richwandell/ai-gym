import math
import random
from collections import deque
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import gym
import numpy as np
import tensorflow as tf
from tensorflow.core.protobuf.config_pb2 import ConfigProto, GPUOptions
from tensorflow.python.client.session import Session
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.optimizer_v2.adam import Adam


class Learner:

    @staticmethod
    def config_tensorflow():
        config = ConfigProto(
            gpu_options=GPUOptions(per_process_gpu_memory_fraction=0.8)
        )
        config.gpu_options.allow_growth = True
        session = Session(config=config)
        set_session(session)

    def __init__(self, n_tries=2000, batch_size=10, gamma=1.0, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.799):
        self.gamma = gamma
        self.config_tensorflow()
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.n_tries = n_tries
        self.env = gym.make("CartPole-v1")
        self.action_space = self.env.action_space.n
        self.observation_space_dim = self.env.observation_space.shape[0]
        self.memory = deque(maxlen=100000)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, "relu", input_dim=self.observation_space_dim),
            tf.keras.layers.Dense(48, "relu"),
            tf.keras.layers.Dense(2, "relu")
        ])
        self.model.compile(loss="mse", optimizer=Adam(lr=0.01, decay=0.01))

    def remember(self, item):
        self.memory.append(item)

    def get_action(self, state, epsilon):
        r = np.random.random()
        if r <= epsilon or (self.batch_size * 3 > len(self.memory)):
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def learn(self):
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        x_train, y_train = [], []
        for state, action, reward, observation, done in minibatch:
            y_target = self.model.predict(state)
            if not done:
                y_target[0][action] = reward + self.gamma * np.max(self.model.predict(observation))
            else:
                y_target[0][action] = reward

            x_train.append(state[0])
            y_train.append(y_target[0])

        x_batch = np.array(x_train)
        y_batch = np.array(y_train)
        history = self.model.fit(x_batch, y_batch, batch_size=len(x_train), verbose=False, epochs=5)
        print(history.history['loss'])

    def get_epsilon(self, t):
        return max(self.epsilon_min, self.epsilon)

    def run(self):
        for t in range(self.n_tries):
            """

            State is made up of
            1. cart position
            2. cart velocity
            3. pole angle
            4. pole velocity at tip
            """
            state = np.reshape(self.env.reset(), [1, 4])

            step = 0
            while True:
                step += 1
                # self.env.render()

                epsilon = self.get_epsilon(t)
                action = self.get_action(state, epsilon)
                observation, reward, done, _ = self.env.step(action)
                observation = np.reshape(observation, [1, 4])
                self.remember((state, action, reward, observation, done))
                state = observation

                if step == 500:
                    self.model.save("working_model1")

                if done:
                    print(
                        "Try: " + str(t)
                        + " Step: " + str(step)
                        + " Epsilon: " + str(epsilon)
                        + " Memory size: " + str(len(self.memory)))
                    break
            self.learn()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    learner = Learner()
    learner.run()
