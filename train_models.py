from cgitb import enable
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG

import os 
from env import Hopper
import numpy as np 

import matplotlib.pyplot as plt
import argparse 
import uuid
import yaml  
import inspect

def get_kwargs():
    # https://stackoverflow.com/questions/2521901/get-a-list-tuple-dict-of-the-arguments-passed-to-a-function
    frame = inspect.currentframe().f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != 'self':
            kwargs[key] = values[key]
    return kwargs

def record_into_file(data_dict, log_dir): 
    with open(os.path.join(log_dir, data_dict), 'w') as file: 
        documents = yaml.dump(data_dict, file)

def train_model(predict_val='',
                train_timesteps=500000, 
                enable_torque=False, 
                env_overrides={}): 
    env = Hopper(enable_torque=enable_torque, predict_val=predict_val)
    for key, value in env_overrides.items(): 
        setattr(env, key, value)

    prefix_name = f"{''.join(predict_val)}_{enable_torque}_{str(uuid.uuid1())}"
    log_dir = "/nobackup/users/yunxingl/log"
    log_dir = os.path.join(log_dir, prefix_name)

    record_into_file(get_kwargs(), log_dir)

    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)

    # the noise objects for DDPG
    n_actions = 3
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Save a checkpoint every 20000 steps
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='/nobackup/users/yunxingl/ckpt/',
                                            name_prefix=prefix_name)
    model = DDPG("MlpPolicy", env, action_noise=action_noise)#, verbose=1)
    model.learn(total_timesteps=train_timesteps, log_interval=100, callback=checkpoint_callback)

train_model(predict_val='strain', 
            enable_torque=True, 
            env_overrides={"strain_cost": 0.01})