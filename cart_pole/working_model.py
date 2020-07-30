import gym
import numpy as np
from tensorflow.core.protobuf.config_pb2 import ConfigProto, GPUOptions
from tensorflow.python.client.session import Session
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model

if __name__ == "__main__":
    config = ConfigProto(
        gpu_options=GPUOptions(per_process_gpu_memory_fraction=0.8)
    )
    config.gpu_options.allow_growth = True
    session = Session(config=config)
    set_session(session)

    model = load_model("working_model")

    env = gym.make("CartPole-v1")

    for t in range(200):

        state = np.reshape(env.reset(), [1, 4])

        step = 0
        while True:
            step += 1
            env.render()

            action = np.argmax(model.predict(state))
            observation, _, done, _ = env.step(action)
            observation = np.reshape(observation, [1, 4])
            state = observation

            if step == 500:
                print("It Worked!")

            if done:
                print(
                    "Try: " + str(t)
                    + " Step: " + str(step)
                )
                break
