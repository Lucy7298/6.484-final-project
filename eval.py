from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3 import DDPG, SAC
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
import re


TTF_PATH = "/nobackup/users/yunxingl/final_project/arial.ttf"

def get_data(path_to_ckpt, save_path, data_save_path, n_steps=200, debug_config='', save_video=True, save_df=True): 
    """
    debug_config: path to some config file you would like to display
    """

    frames = []

    config = {}

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
    agent = SAC.load(path_to_ckpt) # change this to DDPG if evaluating DDPG! 
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

    if save_video: 
        imageio.mimsave(save_path, [np.asarray(img) for i, img in enumerate(frames) if i%2 == 0], fps=29)
    df = pd.DataFrame(all_data)

    if save_df: 
        df.to_csv(data_save_path)
    return df, config


def plot_figure(log_dir, train_timesteps, save_file): 
    plot_results([log_dir], train_timesteps, results_plotter.X_TIMESTEPS, "Hopper")
    plt.savefig(save_file)

def eval_all(possible_paths, idxes):
    all_data = []
    for idx in idxes: 
        for path in possible_paths: 
            path = Path(path)
            if (path / str(idx)).exists(): 
                ckpt_dir = (path / str(idx) / "checkpoint")
                config_path = (path / str(idx) / "config.json")
                for ckpt in ckpt_dir.iterdir(): 
                    for i in range(4): 
                        fname = ckpt.stem
                        num_epochs = re.findall("\d+", fname)
                        assert len(num_epochs) == 1 
                        num_epochs = int(num_epochs[0])
                        save_path = (path / str(idx) / f"{num_epochs}_{i}_video.gif")
                        df, args = get_data(ckpt, save_path, None, 1000, config_path, True, False)
                        df = df[["progress", "eval_electricity", "eval_torque", "eval_strain"]]
                        df['num_epochs'] = num_epochs
                        for k, v in args.items(): 
                            df[k] = v
                        df["index"] = idx
                        all_data.append(df)
                        gc.collect()

    return pd.concat(all_data)

if __name__ == '__main__':
    # eval_dirs = [6]
    # data_dir = Path('/nobackup/users/yunxingl/models/experiment_1') 
    # # for directory in data_dir.iterdir(): 
    # #     if directory.is_dir() and directory.name != 'slurm_logs': 
    # for dir in eval_dirs: 
    #     directory = data_dir / str(dir)
    #     for i in range(4): 
    #         video_output = str((directory / f"900000_animation_{i}.mp4"))
    #         data_output = str((directory / f"evaluation_metrics_{i}.csv"))
    #         ckpt_path = str((directory / 'checkpoint' / "rl_model_900000_steps.zip"))
    #         debug_config = str((directory / "config.json"))
    #         get_data(ckpt_path, video_output, data_output, debug_config)
    #         print(directory)
    
    #         log_dir = directory / "log"
    #         plot_figure(log_dir, 900000, directory / "final_graph.png")
            
    #         gc.collect()
    # df = eval_all(["/home/getnakul/models/experiment_1", 
    #           "/home/mnocito/models/exp4", 
    #           "/home/yunxingl/models/experiment_1"], [0, 3, 6, 17, 4, 30])
    # df.to_csv("/nobackup/users/yunxingl/all_data.csv")
    # for i in range(4): 
    #     get_data("/home/yunxingl/models/experiment_1/0/checkpoint/rl_model_500000_steps.zip", 
    #            f"animations/0_500000_clip_{i}.mp4", 
    #             f"graphs/0_500000_data_{i}.csv", 
    #             n_steps=200, debug_config="/home/yunxingl/models/experiment_1/0/config.json")
    #     gc.collect()
    df = eval_all(["/nobackup/users/yunxingl/models/sac_exp"], [0, 17, 3, 30, 4, 6])
    df = df.to_csv("/nobackup/users/yunxingl/models/sac_exp/all_data.csv")