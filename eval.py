from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3 import DDPG
import matplotlib.pyplot as plt
import imageio
from PIL import Image, ImageDraw, ImageFont  

from env import Hopper
import numpy as np 
import json 
import matplotlib.pyplot as plt
from pathlib import Path

from argparse import ArgumentParser, Namespace
import json
from train_models import initialize_env
import pandas as pd
from decimal import Decimal
import gc


TTF_PATH = "/nobackup/users/yunxingl/final_project/arial.ttf"

def make_video(path_to_ckpt, save_path, data_save_path, debug_config=''): 
    """
    debug_config: path to some config file you would like to display
    """
    n_steps = 200

    frames = []

    if debug_config: 
        with open(debug_config, 'r') as f: 
            parser = ArgumentParser()
            t_args = Namespace()
            config = json.load(f)
            config['eval_mode'] = True
            config['debug'] = True
            t_args.__dict__.update(config)
            args = parser.parse_args(namespace=t_args)
            env = initialize_env(args)
    
    else: 
        env = Hopper(enable_torque=False, 
                     predict_val="", 
                     add_additional=False, 
                     eval_mode=True, 
                     debug=True)
    #env = Monitor(env, "debug_log")
    font  = ImageFont.truetype(TTF_PATH, 10, encoding="unic")
    agent = DDPG.load(path_to_ckpt)
    frames = []  # Frames for video.
    all_data = []
    timestep = env.reset().astype(float)
    data = {}
    for i in range(n_steps):
        action, _ = agent.predict(timestep)
        timestep, _, done, data = env.step(action)

        frame_numpy = env.render(mode='rgb_array').copy()
        frame_pil = Image.fromarray(np.uint8(frame_numpy*255))
        draw_pil  = ImageDraw.Draw(frame_pil)
        if debug_config: 
            for idx, (k, v) in enumerate(data.items()): 
                display_value = '%.2E' % Decimal(v)
                draw_pil.text( (0, idx*10), f"{k}={display_value}", fill="#00ffff", font=font)
            
        frames.append(frame_pil)

        all_data.append(data)
        if done: 
            break 

    imageio.mimsave(save_path, [np.asarray(img) for i, img in enumerate(frames) if i%2 == 0], fps=29)
    df = pd.DataFrame(all_data)
    df.to_csv(data_save_path)


def plot_figure(log_dir, train_timesteps, save_file): 
    plot_results([log_dir], train_timesteps, results_plotter.X_TIMESTEPS, "Hopper")
    plt.savefig(save_file)

if __name__ == '__main__':
    data_dir = Path('/nobackup/users/yunxingl/models/experiment_1') 
    for directory in data_dir.iterdir(): 
        if directory.is_dir() and directory.name != 'slurm_logs': 
            video_output = str((directory / "900000_animation.mp4"))
            data_output = str((directory / "evaluation_metrics.pkl"))
            ckpt_path = str((directory / 'checkpoint' / "rl_model_900000_steps.zip"))
            debug_config = str((directory / "config.json"))
            make_video(ckpt_path, video_output, data_output, debug_config)
            print(directory)
    
            log_dir = directory / "log"
            plot_figure(log_dir, 900000, directory / "final_graph.png")
            
            gc.collect()