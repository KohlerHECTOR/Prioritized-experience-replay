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

# For visualization
os.environ["VIDEO_FPS"] = "5"

"""
Just launch python3 main.py to reproduce fig 8.5 from Sutton and Barto's Intro to RL
2nd edition, figure 8.5 .
"""

mdp = create_random_maze(9, 6, 0.13)

# Sutton and Barto parameters (figure 8.5 from Intro to RL) with eps_greedy pol
# alpha = 0.1, eps = 0.1, gamma = 0.95

# Mattar paraemeters with softmax policy
ALPHA = 0.1
EPS = 0.1
# BETA = 5 # inverse temperature
# TAU = 1/BETA
GAMMA = 0.95
NB_EXP =  50

print("-------- Rat does 0 steps of planning ----------")
to_plot_planning_0 = []
rat = DynaQAgent(mdp, alpha = ALPHA, gamma = GAMMA)
for i in range(NB_EXP):
    to_plot_planning_0.append(rat.learn(eps = EPS, nb_episodes = 50, render = False, n = 0, timeout = 20000, seed = i ))
    print("Experience number: {}".format(i + 1))
to_plot_planning_0 = np.array(to_plot_planning_0).mean(axis = 0)

print("-------- Rat does 5 steps of planning ----------")
to_plot_planning_5 = []
rat = DynaQAgent(mdp, alpha = ALPHA, gamma = GAMMA)
for i in range(NB_EXP):
    to_plot_planning_5.append(rat.learn(eps = EPS, nb_episodes = 50, render = False, n = 5, timeout = 20000, seed = i))
    print("Experience number: {}".format(i + 1))
to_plot_planning_5 = np.array(to_plot_planning_5).mean(axis = 0)

print("-------- Rat does 50 steps of planning ----------")
to_plot_planning_50 = []
rat = DynaQAgent(mdp, alpha = ALPHA, gamma = GAMMA)
for i in range(NB_EXP):
    to_plot_planning_50.append(rat.learn(eps = EPS, nb_episodes = 50, render = False, n = 50, timeout = 20000, seed = i))
    print("Experience number: {}".format(i + 1))
to_plot_planning_50 = np.array(to_plot_planning_50).mean(axis = 0)


# print(np.argmax(rat.M[0, :]))
# plot_need(9, 6, rat.M)


plt.clf()
plt.plot(to_plot_planning_0[:], label = "0 planning steps")
plt.plot(to_plot_planning_5[:], label = "5 planning steps")
plt.plot(to_plot_planning_50[:], label = "50 planning steps")
plt.xlabel("steps per epsiode")
plt.legend()
plt.savefig("results/Sutton&Barto/figure_8_5.png")
plt.clf()
# rat.mdp.current_state = 0
# rat.mdp.render(rat.Q, np.argmax(rat.Q, axis=1), title="Dyna-Q softmax")
# plt.savefig("results/q_table_dyna_q_50_steps_planning_gain_need_softmax")
