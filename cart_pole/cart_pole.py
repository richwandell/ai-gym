import random
from collections import deque

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

    def __init__(self, n_tries=50, batch_size=10, e_value=1.0, e_min=0.01, e_decay=0.799):
        self.e_decay = e_decay
        self.e_min = e_min
        self.e_value = e_value
        self.batch_size = batch_size
        self.config_tensorflow()
        self.n_tries = n_tries
        self.env = gym.make("CartPole-v1")
        self.observation_space_dim = self.env.observation_space.shape[0]

    def create_model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, "relu", input_dim=self.observation_space_dim),
            tf.keras.layers.Dense(48, "relu"),
            tf.keras.layers.Dense(2, "relu")
        ])
        self.model.compile(loss="mse", optimizer=Adam(lr=0.01, decay=0.01))

    def learn(self):
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        x_train, y_train = [], []

        for state, action, observation, done in minibatch:
            y_target = self.model.predict(state)
            if not done:
                y_target[0][action] = 1.0 + np.argmax(self.model.predict(observation))
            else:
                y_target[0][action] = 1.0

            x_train.append(state[0])
            y_train.append(y_target[0])

        x_batch = np.array(x_train)
        y_batch = np.array(y_train)

        history = self.model.fit(x_batch, y_batch, batch_size=len(x_train), verbose=False, epochs=5)
        print(history.history['loss'])

    def get_action(self, state):
        r = np.random.random()
        if r <= self.e_value:
            return self.env.action_space.sample()
        return np.argmax(self.model.predict(state))

    def run(self):
        finished = False

        while not finished:
            self.e_value = 1.0
            self.create_model()
            self.memory = deque(maxlen=100000)

            for t in range(self.n_tries):
                """
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

                    action = self.get_action(state)
                    observation, _, done, _ = self.env.step(action)
                    observation = np.reshape(observation, [1, 4])

                    self.memory.append((state, action, observation, done))
                    state = observation

                    if step == 500:
                        self.model.save("working_model")
                        finished = True
                        break

                    if done:
                        print(
                            "Try: " + str(t)
                            + " Step: " + str(step)
                            + " Mem: " + str(len(self.memory))
                            + " Explore: " + str(self.e_value)
                        )
                        break
                self.learn()
                if self.e_value > self.e_min:
                    self.e_value *= self.e_decay



if __name__ == "__main__":
    learner = Learner()
    learner.run()
