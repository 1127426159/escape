import time
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import *

from env_util import make_escape_env
from callback import SaveOnBestTrainingRewardCallback
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logdir = "result_observation_200"
    env = make_escape_env(seed=0, n_envs=1, monitor_dir=logdir, env_kwargs={'observation_range': 200})
    env = VecFrameStack(env, n_stack=4)

    model = DQN('CnnPolicy', env, verbose=1, buffer_size=20000, policy_kwargs={'net_arch': [256, 128, 32]}, batch_size=6400)

    callback = SaveOnBestTrainingRewardCallback(log_dir=logdir)
    model.learn(total_timesteps=7000000, callback=callback)

    while True:
        obs = env.reset()
        env.render()
        step = 0
        dones = False
        while not dones:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            env.render()
            step += 1
        print("total step:{}".format(step))