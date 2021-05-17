import time
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import *

from env_util import make_escape_env
from callback import SaveOnBestTrainingRewardCallback
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logdir = "result"
    env = make_escape_env(seed=0, monitor_dir=logdir, env_kwargs={'boom_count': 1000})
    env = VecFrameStack(env, n_stack=4)

    model = A2C.load(logdir+"/best_model", env)

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