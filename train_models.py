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
    with open(os.path.join(log_dir, "args.yaml"), 'w') as file: 
        documents = yaml.dump(data_dict, file)

def train_model(predict_val='',
                train_timesteps=500000, 
                enable_torque=False, 
                add_additional=True,
                env_overrides={}): 
    print("torch available", torch.cuda.is_available())
    env = Hopper(enable_torque=enable_torque, predict_val=predict_val, add_additional=add_additional)
    for key, value in env_overrides.items(): 
        setattr(env, key, value)

    prefix_name = f"{predict_val}_{enable_torque}_{str(uuid.uuid1())}"
    log_dir = "/nobackup/users/yunxingl/log"
    log_dir = os.path.join(log_dir, prefix_name)
    os.makedirs(log_dir, exist_ok=True)

    record_into_file(get_kwargs(), log_dir)

    env = Monitor(env, log_dir)

    # the noise objects for DDPG
    n_actions = 3
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Save a checkpoint every 20000 steps
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path='/nobackup/users/yunxingl/ckpt/',
                                            name_prefix=prefix_name)
    model = DDPG("MlpPolicy", env, action_noise=action_noise)#, verbose=1)
    model.learn(total_timesteps=train_timesteps, log_interval=100, callback=checkpoint_callback)

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
  # Run env for n_steps, apply random actions, and show video.
    n_steps = 200

    frames = []
    env = Hopper(enable_torque=False, predict_val=False, add_additional=False)
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



# plot_figure("/nobackup/users/yunxingl/log/strain_True_0f9b994c-bec2-11ec-b6eb-0894ef8099d1", 900000, "graphs/strain_no_energy_reward_half_surprise.png")
# make_video("/nobackup/users/yunxingl/ckpt/strain_True_0f9b994c-bec2-11ec-b6eb-0894ef8099d1_900000_steps.zip", 
#            "animations/strain_no_energy_reward_half_surprise.gif" )
train_model(predict_val='strain',
                train_timesteps=1000000, 
                enable_torque=True, 
                add_additional=True,
                env_overrides={"electricity_cost": 0, 
                               "stall_torque_cost": 0, 
                               "surprise_coeff": 1})