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
from dynaq import DynaQAgent
from utils import plot_need
# For visualization
os.environ["VIDEO_FPS"] = "5"



mdp = create_random_maze(9, 6, 0.13)

# Sutton and Barto parameters (figure 8.5 from Intro to RL) with eps_greedy pol
# alpha = 0.1, eps = 0.1, gamma = 0.95

# Mattar paraemeters with softmax policy
ALPHA = 1
BETA = 5 # inverse temperature
TAU = 1/BETA
GAMMA = 0.9
NB_EXP = 1 #30
to_plot_planning_50 = []

rat = DynaQAgent(mdp, alpha = ALPHA, gamma = GAMMA)
for i in range(NB_EXP):
    to_plot_planning_50.append(rat.learn(tau = TAU, nb_episodes = 50, render = False, n = 50, timeout = 3000, seed = i))

print(np.argmax(rat.M[0, :]))
# plot_need(9, 6, rat.M)
to_plot_planning_50 = np.array(to_plot_planning_50).mean(axis = 0)

plt.clf()

plt.plot(to_plot_planning_50, label = "50 planning steps")
plt.xlabel("steps per epsiode")
plt.legend()
plt.savefig("results/test_gain_need_softmax")
plt.clf()
rat.mdp.current_state = 0
rat.mdp.render(rat.Q, np.argmax(rat.Q, axis=1), title="Dyna-Q softmax")
plt.savefig("results/q_table_dyna_q_50_steps_planning_gain_need_softmax")
