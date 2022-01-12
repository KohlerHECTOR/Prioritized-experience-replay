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
args.set_all_need_to_1 = True
args.start_random = True
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
forward_count = np.zeros((len(list_data), mdp.nb_states))
reverse_count = np.zeros((len(list_data), mdp.nb_states))
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

for k, data in enumerate(list_data):
    print("ANALYSING SIMU NB: {}".format(k + 1))
    data.replay.state = np.array(data.replay.state).T
    print(data.replay.state.shape)
    data.replay.action = np.array(data.replay.action).T
    candidate_events = get_candidate_events(data, mdp, min_frac_cells, min_num_cells)
    print(len(candidate_events))
    print(len(data.list_exp))
    agent_pos = np.array(data.list_exp)[candidate_events]# agent position during each candidate event
    agent_pos = agent_pos[:,0]
    for e, evt in enumerate(candidate_events):
        event_state = np.array(data.replay.state[evt]) # In a multi-step sequence, simData.replay.state has 1->2 in one row, 2->3 in another row, etc
        event_action = np.array(data.replay.action[evt]) # In a multi-step sequence, simData.replay.action has the action taken at each step of the trajectory
        # Identify break points in this event, separating event into sequences
        event_dir = np.full(len(event_state)- 1, "")
        break_points = [0] # Save breakpoints that divide contiguous replay events

        for i, evt_state in enumerate(event_state[:-1]):
            # If state(i) and action(i) leads to state(i+1): FORWARD
            if next_state[int(event_state[i]), int(event_action[i])] == event_state[i +1]:
                event_dir[i] = "F"

            # If state(i+1) and action(i+1) leads to state(i): REVERSE
            if next_state[int(event_state[i + 1]), int(event_action[i + 1])] == event_state[i]:
                event_dir[i] = "R"

            if event_dir[i] == "": # If this transition was neither forward nor backward
                break_points.append(i - 1) # Then, call this a breakpoint

            elif i > 0:
                if event_dir[i] != event_dir[i - 1]:
                    break_points.append(i - 1)

            if (i) == (len(event_state) - 1):
                break_points.append(i) # Add a breakpoint after the last transition

        # Break this event into segments of sequential activity
        for j, b_pt in enumerate(break_points[:-1]):
            this_chunk = list(range(break_points[j] + 1, break_points[j + 1] + 1))
            if (len(this_chunk) + 1) >= min_num_cells:
                # Extract information from this sequential event
                replay_dir = event_dir[this_chunk] # Direction of transition
                replay_state = event_state[this_chunk + [this_chunk[-1] + 1]] # start state
                replay_action = event_action[this_chunk + [this_chunk[-1] + 1]] # action

                if replay_dir[0] == "F":
                    forward_count[k, int(agent_pos[e])] += 1
                elif replay_dir[0] == "R":
                    reverse_count[k, int(agent_pos[e])] += 1

#
# Compute the number of significant events BEFORE (preplay) and AFTER (replay) an event (which could be larger than 1)
# PS: Notice that this is not a measure of the percent of episodes with a significant event (which would produce a smaller numbers)

preplay_forward = np.nansum(forward_count[:,: -1 ], axis=0) / args.episodes # goal state is always last
replay_forward = np.nansum(forward_count[:, -1], axis=0) / args.episodes
preplay_reverse = np.nansum(reverse_count[:,: -1 ],axis=0) / args.episodes # goal state is always last
replay_reverse = np.nansum(reverse_count[:, -1], axis=0) / args.episodes
print([np.nanmean(preplay_forward), np.nanmean(replay_forward), np.nanmean(preplay_reverse), np.nanmean(replay_reverse)])
to_plot = [np.nanmean(preplay_forward), np.nanmean(replay_forward), np.nanmean(preplay_reverse), np.nanmean(replay_reverse)]
plt.bar(range(len(to_plot)), to_plot, width = 0.6, color = 'black')
plt.xticks(range(len(to_plot)), ["Pre-For", 'Repl-For', 'Pre-Back', 'Repl-Back'])
plt.savefig("results/Mattar/mattar_fig_3_a_open_maze.pdf")
