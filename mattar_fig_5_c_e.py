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

## FIG 5 C or E ##
for k, data in enumerate(list_data):
    data.replay.state = np.array(data.replay.state).T
    data.replay.action = np.array(data.replay.action).T
    candidate_events = get_candidate_events(data, mdp, min_frac_cells, min_num_cells)
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

                reward_s_n = [data.list_exp[h][3] in [mdp.nb_states - 1] for h in range(evt)] #reward_tsi identifies the timepoint corresponding to the last reward received (at or prior to the current chunk)
                last_reward_s_n = 0
                for idx , bol in enumerate(reward_s_n):
                    if bol:
                        last_reward_s_n = idx

                last_reward_mag = data.list_exp[last_reward_s_n][3] # lastReward_mag is the magnitude of the last reward received, prior to this chunk
                if replay_dir[0] == "F":
                    if last_reward_mag != args.reward_multiplicator:
                        forward_count_baseline[k, int(agent_pos[e])] +=1
                    else:
                        forward_count_rew_shift[k, int(agent_pos[e])] +=1

                elif replay_dir[0] == "R":
                    if last_reward_mag != args.reward_multiplicator:
                        reverse_count_baseline[k, int(agent_pos[e])] +=1
                    else:
                        reverse_count_rew_shift[k, int(agent_pos[e])] +=1


num_ep_baseline = args.episodes * (1 - args.reward_change_proba)
num_ep_rew_shift = args.episodes * (args.reward_change_proba)

preplay_forward_baseline = np.nansum(forward_count_baseline [:, : -1], axis = 0 ) / num_ep_baseline
replay_forward_baseline = np.nansum(forward_count_baseline [:,-1], axis = 0 ) / num_ep_baseline
preplay_reverse_baseline = np.nansum(reverse_count_baseline [:, : -1], axis = 0 ) / num_ep_baseline
replay_reverse_baseline =  np.nansum(reverse_count_baseline [:,-1], axis = 0 ) / num_ep_baseline

preplay_forward_rew_shift = np.nansum(forward_count_rew_shift [:, : -1], axis = 0 ) / num_ep_rew_shift
replay_forward_rew_shift = np.nansum(forward_count_rew_shift [:,-1], axis = 0 ) / num_ep_rew_shift
preplay_reverse_rew_shift = np.nansum(reverse_count_rew_shift [:, : -1], axis = 0 ) / num_ep_rew_shift
replay_reverse_rew_shift =  np.nansum(reverse_count_rew_shift [:,-1], axis = 0 ) / num_ep_rew_shift

to_plot = [np.nanmean(preplay_forward_baseline), np.nanmean(replay_forward_baseline), np.nanmean(preplay_reverse_baseline), np.nanmean(replay_reverse_baseline), np.nanmean(preplay_forward_rew_shift), np.nanmean(replay_forward_rew_shift), np.nanmean(preplay_reverse_rew_shift), np.nanmean(replay_reverse_rew_shift)]
plt.bar(range(len(to_plot)), to_plot, width = 0.6, color = 'black')
plt.xticks(range(len(to_plot)), ["PF1", 'RF1', 'PB1', 'RB4', "PF4", 'RF4', 'PB4', 'RB4'])
plt.savefig("results/Mattar/mattar_fig_5_c_open_maze.pdf")
