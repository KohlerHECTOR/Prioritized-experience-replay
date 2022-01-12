import os

import numpy as np
import matplotlib.pyplot as plt

from mazemdp.maze import build_maze, create_random_maze
from random import seed
from mazemdp import create_random_maze
from replay_sim import Agent
from utils import evaluate, to_plot, Replay, SimuData

from arguments import get_args, get_args_string

import os

import numpy as np
import matplotlib.pyplot as plt

from mazemdp.maze import build_maze, create_random_maze
from random import seed
from mazemdp import create_random_maze
from replay_sim import Agent
from utils import evaluate, to_plot, Replay, SimuData

from arguments import get_args, get_args_string


args = get_args()

mdp = create_random_maze(9, 6, 0.2)

# using e-greedy pol for all models
args.action_policy = "greedy"
args.epsilon = 0.
args.start_random = True
print(get_args_string(args))
#### MATTAR MODEL ####
print("WITH PRIORITIZED REPLAY")
res_train_prio = []
res_list_Q_prio = []
nb_exps_prio = []
rew_per_exp_prio = []
for i in range(args.simulations):
    print("#### SIM NB: {}".format(i))
    replay = Replay()
    saver = SimuData(replay)
    rat = Agent(mdp, args, saver)
    rat.learn(args, seed = i)
    res_train_prio.append(saver.steps_to_exit)
    res_list_Q_prio.append(saver.list_Q)
    rew_per_exp_prio.append(saver.rew_per_exp)



#### Q-learning######
print("Q-learning")
args.set_all_gain_to_1 = False
args.set_all_need_to_1 = False
args.planning_steps = 0
res_train_nothing = []
res_list_Q_nothing = []
nb_exps_nothing = []
rew_per_exp_nothing = []
for i in range(args.simulations):
    print("#### SIM NB: {}".format(i))
    replay = Replay()
    saver = SimuData(replay)
    rat = Agent(mdp, args, saver)
    rat.learn(args, seed = i)
    res_train_nothing.append(saver.steps_to_exit)
    res_list_Q_nothing.append(saver.list_Q)
    nb_exps_nothing.append(saver.nb_exps)
    rew_per_exp_nothing.append(saver.rew_per_exp)



to_plot(rew_per_exp_prio, label = "Prioritized replay")
to_plot(rew_per_exp_nothing, label = "Q-learning")
plt.xlabel("experiences accessed")
plt.ylabel("total reward")
plt.legend()
plt.savefig("results/rew_per_exps.png")
plt.clf()
