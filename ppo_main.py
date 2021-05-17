import time
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import *

from env_util import make_escape_env
from callback import SaveOnBestTrainingRewardCallback
import logging
logging.basicConfig(level=logging.DEBUG)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # There already exists an environment generator
    # that will make and wrap atari environments correctly.
    # Here we are also multi-worker training (n_envs=4 => 4 environments)
    logdir = "ppo_result"
    env = make_escape_env(monitor_dir=logdir)
    # Frame-stacking with 4 frames
    env = VecFrameStack(env, n_stack=4)

    model = PPO('CnnPolicy', env, verbose=1)

    # model = PPO.load(logdir+"/best_model", env, exploration_initial_eps=0.05)
    callback = SaveOnBestTrainingRewardCallback(log_dir=logdir)
    model.learn(total_timesteps=7000000, callback=callback)
    # model.learn(total_timesteps=0, callback=callback)

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