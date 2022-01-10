import os

import numpy as np
import matplotlib.pyplot as plt

from mazemdp.toolbox import egreedy, egreedy_loc, softmax
from mazemdp.maze import build_maze, create_random_maze
from mazemdp.maze_plotter import show_videos
from random import seed
from mazemdp import create_random_maze
from dynaq import DynaQAgent

# For visualization
os.environ["VIDEO_FPS"] = "5"


mdp = create_random_maze(9, 6, 0.2)

# Sutton and Barto parameters (figure 8.5 from Intro to RL) with eps_greedy pol
# alpha = 0.1, eps = 0.1, gamma = 0.95

ALPHA = 0.1
EPS = 0.1
GAMMA = 0.95
NB_EXP =  300

print("-------- Rat does 0 steps of planning ----------")
to_plot_planning_0 = []

rat = DynaQAgent(mdp, alpha = ALPHA, gamma = GAMMA)

for i in range(NB_EXP):
    data_train, data_eval = rat.learn(eps = EPS, nb_episodes = 30, render = False, n = 0, timeout = 20000, seed = i )
    to_plot_planning_0.append(data_eval)
    print("Experience number: {}".format(i + 1))

to_plot_planning_0 = np.array(to_plot_planning_0).mean(axis = 0)

print("-------- Rat does 5 steps of planning ----------")
to_plot_planning_5 = []

rat = DynaQAgent(mdp, alpha = ALPHA, gamma = GAMMA)

for i in range(NB_EXP):
    data_train, data_eval = rat.learn(eps = EPS, nb_episodes = 30, render = False, n = 5, timeout = 20000, seed = i)
    to_plot_planning_5.append(data_eval)
    print("Experience number: {}".format(i + 1))

to_plot_planning_5 = np.array(to_plot_planning_5).mean(axis = 0)

print("-------- Rat does 50 steps of planning ----------")
to_plot_planning_50 = []

rat = DynaQAgent(mdp, alpha = ALPHA, gamma = GAMMA)

for i in range(NB_EXP):
    data_train, data_eval = rat.learn(eps = EPS, nb_episodes = 30, render = False, n = 50, timeout = 20000, seed = i)
    to_plot_planning_50.append(data_eval)
    print("Experience number: {}".format(i + 1))

to_plot_planning_50 = np.array(to_plot_planning_50).mean(axis = 0)

plt.plot(to_plot_planning_0[:], label = "0 planning steps")
plt.plot(to_plot_planning_5[:], label = "5 planning steps")
plt.plot(to_plot_planning_50[:], label = "50 planning steps")
plt.xlabel("episode")
plt.ylabel("steps to goal")
plt.title("Evaluation of Sutton & Barto's Dyna-Q agent")
plt.legend()
plt.savefig("results/Sutton&Barto/figure_8_5.png")
