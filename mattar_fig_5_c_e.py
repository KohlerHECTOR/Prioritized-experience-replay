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



mdp = create_random_maze(5, 5, 0.2)

args.reward_change_proba = 0.5
args.reward_multiplicator = 4 #0 for fig 5_e
print(get_args_string(args))
list_data = []
for i in range(args.simulations):
    print("#### SIM NB: {}".format(i))
    replay = Replay()
    saver = SimuData(replay)
    rat = Agent(mdp, args, saver)
    rat.learn(args, seed = i)
    list_data.append(saver)


### ANALYSIS ###

min_num_cells = 5
min_frac_cells = 0
forward_count_baseline = np.zeros((len(list_data), mdp.nb_states))
reverse_count_baseline = np.zeros((len(list_data), mdp.nb_states))
forward_count_rew_shift = np.zeros((len(list_data), mdp.nb_states))
reverse_count_rew_shift = np.zeros((len(list_data), mdp.nb_states))
next_state = np.full((mdp.nb_states, 4), np.NaN)

for s in range(mdp.nb_states):
    for a in range(mdp.action_space.size):
        mdp.reset()
        mdp.current_state = s
        s_next, _, _, _ = mdp.step(a)
        next_state[s, a] = s_next

def get_candidate_events(data, mdp, min_frac_cells, min_num_cells):
    #Identify candidate replay events: timepoints in which the number of replayed states is greater than minFracCells,minNumCells
    candaidate_events = []
    for idx, r_s in enumerate(data.replay.state):
        if len(r_s) >= max(mdp.nb_states * min_frac_cells , min_num_cells):
            candaidate_events.append(idx)

    return candaidate_events

to_plot = [np.nanmean(preplay_forward_baseline), np.nanmean(replay_forward_baseline), np.nanmean(preplay_reverse_baseline), np.nanmean(replay_reverse_baseline), np.nanmean(preplay_forward_rew_shift), np.nanmean(replay_forward_rew_shift), np.nanmean(preplay_reverse_rew_shift), np.nanmean(replay_reverse_rew_shift)]
plt.bar(range(len(to_plot)), to_plot, width = 0.6, color = 'black')
plt.xticks(range(len(to_plot)), ["PF1", 'RF1', 'PB1', 'RB4', "PF4", 'RF4', 'PB4', 'RB4'])
plt.savefig("results/Mattar/mattar_fig_5_c_open_maze.pdf")
