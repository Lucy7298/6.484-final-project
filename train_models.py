from cgitb import enable
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG
import matplotlib.pyplot as plt
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
import matplotlib
import matplotlib.animation as animation
from IPython.display import HTML
import imageio
from PIL import Image, ImageDraw, ImageFont  

from env import Hopper
import numpy as np 
import os 
import json 
import matplotlib.pyplot as plt
import argparse 
import uuid
import yaml  
import inspect
import torch
from pathlib import Path
from pprint import pprint

from argparse import ArgumentParser
import json

ENV_PARAM_NAMES = ['electricity_cost', 
                   'torque_cost', 
                   'joints_at_limit_cost',
                   'strain_cost', 
                   'electricity_surprise_weight', 
                   'strain_surprise_weight',
                   'lambda1_prime',
                   'lambda2_prime']

def initialize_env(args): 
    env = Hopper(args.use_progress_reward, 
                 args.use_electricity_cost, 
                 args.use_torque_cost, 
                 args.use_limits_cost, 
                 args.use_strain_cost, 
                 args.use_electricity_surprise, 
                 args.use_strain_surprise, 
                 args.use_cost_diff, 
                 eval_mode=args.eval_mode, 
                 debug=args.debug)

    argsdict = vars(args)
    for field_name in ENV_PARAM_NAMES: 
        setattr(env, field_name, argsdict[field_name])
        print(field_name, getattr(env, field_name))
    return env 

def train_model(args): 
    ##SET UP ENV
    print("torch available", torch.cuda.is_available())
    env = initialize_env(args)
    ##SET UP CHECKPOINTING AND LOGGING

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.debug: 
        raise Exception("Output directory exists, set debug flag to log to existing directory")
    log_dir = output_dir / "log"
    config_path = output_dir / "config.json"
    checkpoint_dir = output_dir / "checkpoint"

    os.mkdir(str(output_dir))
    os.mkdir(str(log_dir))
    os.mkdir(str(checkpoint_dir))

    env = Monitor(env, str(log_dir))
    with open(str(config_path), 'wt') as f: 
        json.dump(vars(args), f)

    # the noise objects for DDPG
    n_actions = 3
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Save a checkpoint every 20000 steps
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=str(checkpoint_dir))
    model = DDPG("MlpPolicy", env, action_noise=action_noise)#, verbose=1)
    model.learn(total_timesteps=args.training_timesteps, log_interval=100, callback=checkpoint_callback)


if __name__ == '__main__': 
    parser = ArgumentParser()
    ### FLAGS
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--use_progress_reward', action='store_true')
    parser.add_argument('--use_electricity_cost', action='store_true')
    parser.add_argument('--use_torque_cost', action='store_true')
    parser.add_argument('--use_limits_cost', action='store_true')
    parser.add_argument('--use_strain_cost', action='store_true')
    parser.add_argument('--use_electricity_surprise', action='store_true')
    parser.add_argument('--use_strain_surprise', action='store_true')
    parser.add_argument('--use_cost_diff', action='store_true')

    ### PARAMETERS
    parser.add_argument('--electricity_cost', type=float, default=-0.001, required=False)
    parser.add_argument('--torque_cost', type=float, default=-0.001, required=False)
    parser.add_argument('--stall_torque_cost', type=float, default=-0.1, required=False)
    parser.add_argument('--joints_at_limit_cost', type=float, default=-0.1, required=False)
    parser.add_argument('--strain_cost', type=float, default=-0.0001, required=False)
    parser.add_argument('--electricity_surprise_weight', type=float, default=1, required=False)
    parser.add_argument('--strain_surprise_weight', type=float, default=1, required=False)
    parser.add_argument('--lambda1_prime', type=float, default=2, required=False)
    parser.add_argument('--lambda2_prime', type=float, default=2, required=False)

    parser.add_argument('--training_timesteps', type=int, default=1000000, required=False)
    parser.add_argument('--eval_mode', action='store_true') # this should never be set
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    # plot_figure("/nobackup/users/yunxingl/log/strain_True_d2071332-c19a-11ec-b918-34800d664578", 900000, "graphs/strain_no_energy_reward_half_surprise.png")
    # make_video("/nobackup/users/yunxingl/ckpt/strain_True_d2071332-c19a-11ec-b918-34800d664578_900000_steps.zip", 
    #         "animations/strain_no_energy_reward_half_surprise.gif" )
    train_model(args)