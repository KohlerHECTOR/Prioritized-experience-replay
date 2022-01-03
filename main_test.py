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
parser.add_argument("--start_random", type = bool, default = True)
parser.add_argument("--transi_goal_to_start", type = bool, default = True)
parser.add_argument("--tau", type = float, default = 0.2)
parser.add_argument("--alpha", type = float, default = 1)
parser.add_argument("--lambda_", type = float, default = 1)
parser.add_argument("--T_learning_rate", type = float, default = 0.9)
parser.add_argument("--plan_only_start_end", type = bool, default = True)
parser.add_argument("--planning_steps", type = int, default = 20)
parser.add_argument("--max_episode_steps", type = int, default = 3000)
args = parser.parse_args()

mdp = create_random_maze(9, 6, 0.13)
rat = Agent(mdp)
rat.learn(args)
