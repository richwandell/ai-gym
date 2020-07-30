import gym
import numpy as np
from tensorflow.python.keras.models import load_model

from helpers import fix_mem_issue

if __name__ == "__main__":
    fix_mem_issue()
    model = load_model("working_model")

    print(model)

    env = gym.make("CartPole-v1")

    for t in range(200):
        state = np.reshape(env.reset(), [1, 4])

        step = 0
        while True:
            step += 1
            env.render()
            action = np.argmax(model.predict(state))
            observation, reward, done, _ = env.step(action)
            observation = np.reshape(observation, [1, 4])
            state = observation

            if step == 500:
                print("yay")

            if done:
                print(
                    "Try: " + str(t)
                    + " Step: " + str(step))
                break
