import os
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from mazemdp.toolbox import egreedy, egreedy_loc, softmax
from mazemdp.maze import build_maze, create_random_maze
from mazemdp.maze_plotter import show_videos
from random import seed
from mazemdp import create_random_maze
from replay_sim import Agent


import argparse

# For visualization
os.environ["VIDEO_FPS"] = "5"

parser = argparse.ArgumentParser()
parser.add_argument("--pre_explore", type = bool, default = True)
parser.add_argument("--start_random", type = bool, default = False)
parser.add_argument("--greedy_eval", type = bool, default = False)
parser.add_argument("--transi_goal_to_start", type = bool, default = True)
parser.add_argument("--tau", type = float, default = 0.2)
parser.add_argument("--alpha", type = float, default = 1)
parser.add_argument("--lambda_", type = float, default = 0)
parser.add_argument("--T_learning_rate", type = float, default = 0.9)
parser.add_argument("--plan_only_start_end", type = bool, default = True)
parser.add_argument("--planning_steps", type = int, default = 20)
parser.add_argument("--max_episode_steps", type = int, default = 1e5)
parser.add_argument("--gamma", type = float, default = 0.9)
parser.add_argument("--set_all_need_to_1", type = bool, default = False)
parser.add_argument("--baseline_gain", type = float, default = 1e-10)
parser.add_argument("--EVM_thresh", type = float, default = 0)
parser.add_argument("--allow_loops", type = bool, default = False)
parser.add_argument("--expand_further", type = bool, default = True)
parser.add_argument("--episodes", type = int, default = 50)
parser.add_argument("--online_offline", type = str, default = "online")
parser.add_argument("--remove_samestate", type = bool, default = True)
parser.add_argument("--simulations", type = int, default = 1)
args = parser.parse_args()



mdp = create_random_maze(9, 6, 0.13)
res_train = []
res_eval = []
for i in range(args.simulations):
    print("#### SIM NB: {}".format(i))
    rat = Agent(mdp, args)
    data = rat.learn(args)
    res_train.append(data["train"])
    res_eval.append(data["eval"])


def to_plot(data, label):
    mean_data = np.mean(data, axis = 0)
    std_data = np.std(data, axis = 0)
    plt.plot(mean_data, label = label)
    plt.fill_between(np.arange(len(mean_data)), mean_data + std_data, mean_data - std_data, alpha = 0.4)


to_plot(np.array(res_train), "train")
to_plot(np.array(res_eval), "eval")
plt.legend()
filename = ""
for _, val in args._get_kwargs():
    filename += "_" + str(val)
plt.savefig("results/Mattar/" + filename + ".png")
