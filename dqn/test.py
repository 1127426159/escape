import time
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import *

from env_util import make_escape_env
from callback import SaveOnBestTrainingRewardCallback
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logdir = "result_observation_100"
    env = make_escape_env(seed=0, n_envs=1, monitor_dir=logdir, env_kwargs={'observation_range': 100})
    # env = make_escape_env(seed=0, monitor_dir=logdir, env_kwargs={'boom_count': 500})
    env = VecFrameStack(env, n_stack=4)

    model = DQN.load(logdir+"/best_model", env)

    while True:
        obs = env.reset()
        env.render()
        # time.sleep(0.01)
        step = 0
        dones = False
        while not dones:
            action, _states = model.predict(obs, deterministic=True)
            # action = [random.randint(0, 3)]
            obs, rewards, dones, info = env.step(action)
            env.render()
            step += 1
            # time.sleep(0.01)
        print("total step:{}".format(step))