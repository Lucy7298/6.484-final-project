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

import os 
from env import Hopper
import numpy as np 

import matplotlib.pyplot as plt
import argparse 
import uuid
import yaml  
import inspect
import torch
from pathlib import Path

from argparse import ArgumentParser
import json

ENV_PARAM_NAMES = ['electricity_cost', 
                   'stall_torque_cost', 
                   'joints_at_limit_cost',
                   'strain_cost', 
                   'electricity_surprise_weight', 
                   'strain_surprise_weight',
                   'lambda1_prime',
                   'lambda2_prime']

def train_model(args): 
    print("torch available", torch.cuda.is_available())
    env = Hopper(args.use_progress_reward, 
                 args.use_electricity_cost, 
                 args.use_limits_cost, 
                 args.use_strain_cost, 
                 args.use_electricity_surprise, 
                 args.use_strain_surprise)


    for field_name in ENV_PARAM_NAMES: 
        setattr(env, field_name, args[field_name])

    output_dir = Path(args.output_dir)
    if output_dir.exists(): 
        raise Exception("Output directory doesn't exist")
    log_dir = output_dir / "log"
    checkpoint_dir = output_dir / "checkpoint"
    assert log_dir.exists()
    assert checkpoint_dir.exists()

    env = Monitor(env, log_dir)

    # the noise objects for DDPG
    n_actions = 3
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Save a checkpoint every 20000 steps
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=checkpoint_dir)
    model = DDPG("MlpPolicy", env, action_noise=action_noise)#, verbose=1)
    model.learn(total_timesteps=args.training_timesteps, log_interval=100, callback=checkpoint_callback)

def display_video(frames, framerate=30):
  """Generates video from `frames`.

  Args:
    frames (ndarray): Array of shape (n_frames, height, width, 3).
    framerate (int): Frame rate in units of Hz.

  Returns:
    Display object.
  """
  height, width, _ = frames[0].shape
  dpi = 70
  orig_backend = matplotlib.get_backend()
  matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
  fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
  matplotlib.use(orig_backend)  # Switch back to the original backend.
  ax.set_axis_off()
  ax.set_aspect('equal')
  ax.set_position([0, 0, 1, 1])
  im = ax.imshow(frames[0])
  def update(frame):
    im.set_data(frame)
    return [im]
  interval = 1000/framerate
  anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                  interval=interval, blit=True, repeat=False)
  return HTML(anim.to_html5_video())

def plot_figure(log_dir, train_timesteps, save_file): 
    plot_results([log_dir], train_timesteps, results_plotter.X_TIMESTEPS, "Hopper")
    plt.savefig(save_file)

def make_video(path_to_ckpt, save_path): 
    n_steps = 200

    frames = []
    env = Hopper(enable_torque=False, predict_val="", add_additional=False)
    #env = Monitor(env, "debug_log")

    agent = DDPG.load(path_to_ckpt)
    frames = []  # Frames for video.
    rewards = [[]]  # Reward at every timestep.
    timestep = env.reset().astype(float)
    for _ in range(n_steps):
        frames.append(env.render(mode='rgb_array').copy())
        action, _ = agent.predict(timestep)
        timestep, reward, done, _ = env.step(action)

    # `timestep.reward` is None when episode terminates.
    if not done:
        # Old episode continues.
        rewards[-1].append(reward)
    else:
        # New episode begins.
        rewards.append([])

    imageio.mimsave(save_path, [np.array(img) for i, img in enumerate(frames) if i%2 == 0], fps=29)


if __name__ == '__main__': 
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--use_progress_reward', action='store_true')
    parser.add_argument('--use_electricity_reward', action='store_true')
    parser.add_argument('--use_limits_cost', action='store_true')
    parser.add_argument('--use_strain_cost', action='store_true')
    parser.add_argument('--use_electricity_surprise', action='store_true')
    parser.add_argument('--use_strain_surprise', action='store_true')
    parser.add_argument('--electricity_cost', type=float, default=-0.001, required=False)
    parser.add_argument('--stall_torque_cost', type=float, default=-0.1, required=False)
    parser.add_argument('--joints_at_limit_cost', type=float, default=-0.1, required=False)
    parser.add_argument('--strain_cost', type=float, default=-0.0001, required=False)
    parser.add_argument('--electricity_surprise_weight', type=float, default=1, required=False)
    parser.add_argument('--strain_surprise_weight', type=float, default=1, required=False)
    parser.add_argument('--lambda1_prime', type=float, default=2, required=False)
    parser.add_argument('--lambda2_prime', type=float, default=2, required=False)

    parser.add_argument('--training_timesteps', type=int)

    args = parser.parse_args()
    # plot_figure("/nobackup/users/yunxingl/log/strain_True_d2071332-c19a-11ec-b918-34800d664578", 900000, "graphs/strain_no_energy_reward_half_surprise.png")
    # make_video("/nobackup/users/yunxingl/ckpt/strain_True_d2071332-c19a-11ec-b918-34800d664578_900000_steps.zip", 
    #         "animations/strain_no_energy_reward_half_surprise.gif" )
    train_model(args)