import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import math
import random
from collections import deque
import gym
import numpy as np
from tensorflow.core.protobuf.config_pb2 import ConfigProto, GPUOptions
from tensorflow.python import Session
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
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
    """

    Observation is made up of
    1. cart position
    2. cart velocity
    3. pole angle
    4. pole velocity at tip
    """

    def __init__(self, n_episodes=100, batch_size=20, epsilon=1.0, epsilon_min=0.01, epsilon_log_decay=0.995):
        self.epsilon_min = epsilon_min
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_log_decay
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        self.config_tensorflow()
        self.env = gym.make("CartPole-v1")
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

        self.memory = deque(maxlen=1000000)

        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(48, "relu", input_dim=self.observation_space),
            tf.keras.layers.Dense(16, "relu"),
            tf.keras.layers.Dense(2, "linear")
        ])
        self.model.compile(loss="mse", optimizer=Adam(lr=0.01, decay=0.01))

    def remember(self, step: tuple):
        self.memory.append(step)

    def replay(self):
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        x_train, y_train = [], []
        for state, action, reward, observation, done in minibatch:
            y_target = self.model.predict(state)
            if done:
                y_target[0][action] = reward
            else:
                predicted_future = np.max(self.model.predict(observation)[0])
                y_target[0][action] = reward + predicted_future
            x_train.append(state[0])
            y_train.append(y_target[0])

        x_batch = np.array(x_train)
        y_batch = np.array(y_train)
        self.model.fit(x_batch, y_batch, batch_size=len(x_train))

    def choose_action(self, state, epsilon):
        r = np.random.random()
        if r <= epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def run(self):
        previous_step = 0
        for e in range(self.n_episodes):
            state = np.reshape(self.env.reset(), [1, 4])

            step = 0
            while True:
                step += 1
                self.env.render()

                action = self.choose_action(state, self.get_epsilon(e))
                # observation, reward, done, info
                observation, reward, done, info = self.env.step(action)
                observation = np.reshape(observation, [1, 4])
                self.remember((state, action, reward, observation, done))
                state = observation

                if step == 500:
                    self.model.save("working_model")
                if done:
                    print("Run: " + str(e) + " step: " + str(step) + " epsilon: " + str(self.epsilon) )
                    break

            self.replay()
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    learner = Learner()
    learner.run()



