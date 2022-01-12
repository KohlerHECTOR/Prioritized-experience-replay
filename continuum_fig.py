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
args.simulations = 10
args.epsiodes = 20
args.action_policy = "greedy"
args.epsilon = 0.
args.start_random = True
print(get_args_string(args))
#### MATTAR MODEL ####
print("WITH PRIORITIZED REPLAY")
res_train_prio = []
res_list_Q_prio = []
for i in range(args.simulations):
    print("#### SIM NB: {}".format(i))
    replay = Replay()
    saver = SimuData(replay)
    rat = Agent(mdp, args, saver)
    rat.learn(args, seed = i)
    res_train_prio.append(saver.steps_to_exit)
    res_list_Q_prio.append(saver.list_Q)

#### Dyna - Q ######
print("DYNA-Q")
args.set_all_gain_to_1 = True
args.set_all_need_to_1 = True
res_train_dyna = []
res_list_Q_dyna = []
for i in range(args.simulations):
    print("#### SIM NB: {}".format(i))
    replay = Replay()
    saver = SimuData(replay)
    rat = Agent(mdp, args, saver)
    rat.learn(args, seed = i)
    res_train_dyna.append(saver.steps_to_exit)
    res_list_Q_dyna.append(saver.list_Q)

#### Q-learning######
print("Q-learning")
args.set_all_gain_to_1 = False
args.set_all_need_to_1 = False
args.planning_steps = 0
res_train_nothing = []
res_list_Q_nothing = []
for i in range(args.simulations):
    print("#### SIM NB: {}".format(i))
    replay = Replay()
    saver = SimuData(replay)
    rat = Agent(mdp, args, saver)
    rat.learn(args, seed = i)
    res_train_nothing.append(saver.steps_to_exit)
    res_list_Q_nothing.append(saver.list_Q)


#### Q(lambda)-learning######
print("Q-lambda")
args.set_all_gain_to_1 = False
args.set_all_need_to_1 = False
args.planning_steps = 0
args.lambda_ = 0.9
res_train_qlambda = []
res_list_qlambda = []
for i in range(args.simulations):
    print("#### SIM NB: {}".format(i))
    replay = Replay()
    saver = SimuData(replay)
    rat = Agent(mdp, args, saver)
    rat.learn(args, seed = i)
    res_train_qlambda.append(saver.steps_to_exit)
    res_list_Q_qlambda.append(saver.list_Q)


#### Q(lambda)-learning######
print("Dyna Q-lambda")
args.set_all_gain_to_1 = True
args.set_all_need_to_1 = True
args.planning_steps = 20
args.lambda_ = 0.9
res_train_dynaqlambda = []
res_list_dynaqlambda = []
for i in range(args.simulations):
    print("#### SIM NB: {}".format(i))
    replay = Replay()
    saver = SimuData(replay)
    rat = Agent(mdp, args, saver)
    rat.learn(args, seed = i)
    res_train_dynaqlambda.append(saver.steps_to_exit)
    res_list_Q_dynaqlambda.append(saver.list_Q)

#### Q(lambda)-learning######
print("Prioritized replay Q-lambda")
args.set_all_gain_to_1 = False
args.set_all_need_to_1 = False
args.planning_steps = 20
args.lambda_ = 0.9
res_train_prio_qlambda = []
res_list_prio_qlambda = []
for i in range(args.simulations):
    print("#### SIM NB: {}".format(i))
    replay = Replay()
    saver = SimuData(replay)
    rat = Agent(mdp, args, saver)
    rat.learn(args, seed = i)
    res_train_prio_qlambda.append(saver.steps_to_exit)
    res_list_Q_prio_qlambda.append(saver.list_Q)

to_plot(np.array(res_train_prio), label = "Prioritized replay")
to_plot(np.array(res_train_dyna), label = "Dyna-Q")
to_plot(np.array(res_train_nothing), label = "Q-learning")
to_plot(np.array(res_train_qlambda), label = "Watkin's Q-lambda")
to_plot(np.array(res_train_dynaqlambda), label = "Dyna-Q lambda")
to_plot(np.array(res_train_prio_qlambda), label = "Pioritized replay Q-lambda")
plt.xlabel("episode")
plt.ylabel("steps until goal")
plt.legend()
plt.savefig("/results/continuum_fig.png")
plt.clf()
