import numpy as np
from utils import normalize_mat
from mazemdp.toolbox import softmax, egreedy
from mazemdp.mdp import Mdp
from need_term import need_term
from gain_term import gain_term
from EVM import get_EVM
import sys
from proba import proba
import numpy.matlib
class Agent():

    def __init__(self, mdp, params, data_saver):
        self.saver = data_saver
        self.mdp = mdp
        self.mdp.gamma = params.gamma
        self.Q = np.zeros((self.mdp.nb_states, self.mdp.action_space.size)) # state-action value function
        self.T = np.zeros((self.mdp.nb_states, self.mdp.nb_states)) # state-state transition proba
        self.eTr = np.zeros((self.mdp.nb_states, self.mdp.action_space.size)) # eligibility matrix
        self.list_exp = [] # list to store individual experiences (s, a, r, s')
        self.exp_last_s_next = np.empty((self.mdp.nb_states, self.mdp.action_space.size))#
        self.exp_last_s_next[:] = np.NaN
        self.exp_last_rew = np.empty((self.mdp.nb_states, self.mdp.action_space.size))
        self.exp_last_rew[:] = np.NaN
        self.nb_episodes = 0 # keep track of nb times we reached end of maze

# TODO: code to store sim data using a sim_data class

    def pre_explore(self):
        # To get initial Model

        for s in range(self.mdp.nb_states):
            if s not in self.mdp.terminal_states:
                for a in range(self.mdp.action_space.size):
                    self.mdp.reset()
                    self.mdp.current_state = s # to force the execution of every action in each state
                    s_next, _, _, _ = self.mdp.step(a)
                    self.list_exp.append([s, a, 0, s_next]) # update list of experiences
                    self.exp_last_s_next[s, a] = s_next # update next_state model
                    self.exp_last_rew[s, a] = 0 # update reward model
                    self.T[s, s_next] += 1 # update transition matrix

    def get_plan_exp(self, params):
        # Data structure stuff
        # Create a list of 1-step backups based on 1-step models
        plan_exp = np.concatenate((np.matlib.repmat(np.arange(self.mdp.nb_states), 1, self.mdp.action_space.size).reshape(
            self.mdp.action_space.size * self.mdp.nb_states, 1),
                                   np.repeat(np.arange(self.mdp.action_space.size), self.mdp.nb_states, axis=0).reshape(
                                       self.mdp.action_space.size * self.mdp.nb_states, 1),
                                   np.column_stack(self.exp_last_rew).reshape(self.mdp.action_space.size * self.mdp.nb_states,
                                                                              1),
                                   np.column_stack(self.exp_last_s_next).reshape(self.mdp.action_space.size * self.mdp.nb_states,
                                                                               1)),
            axis=1)
        # Remove NaNs -- e.g. actions starting from invalid states, such as walls:
        plan_exp = plan_exp[np.invert(np.isnan(plan_exp).any(axis=1))]
        # Remove actions that lead to same state (optional) -- e.g. hitting the wall:
        if params.remove_samestate:
            plan_exp = plan_exp[plan_exp[:, 0] != plan_exp[:, 3]]
        plan_exp = list(plan_exp)  # use plan_exp to hold all steps of any n-step trajectory

        return plan_exp

    def tie_break(self, max_EVM_idx, plan_exp):

        nb_steps = np.array([len(plan_exp[i]) for i in range(len(plan_exp))]) #  number of total steps on this trajectory
        max_EVM_idx = max_EVM_idx[nb_steps[max_EVM_idx] == min(nb_steps[max_EVM_idx])]# Select the one corresponding to a shorter trajectory
        if len(max_EVM_idx)>1: # If there are still multiple items with equal gain (and equal length)
            max_EVM_idx = np.random.choice(max_EVM_idx)# ... select one at random

        return max_EVM_idx

    def get_max_EVM_idx(self, EVM, plan_exp):

        max_EVM_idx = np.argwhere(EVM == max(EVM))

        if len(max_EVM_idx) > 1:  # If there are multiple items with equal gain
            max_EVM_idx = self.tie_break(max_EVM_idx, plan_exp)
        else:
            max_EVM_idx = max_EVM_idx[0][0]

        return int(max_EVM_idx)

    def expand(self, params, plan_exp, planning_backups):

        # Find the last entry in planning_backups with that started an n-step backup
        seq_start = np.argwhere(planning_backups[:, 4] == 1)[-1]
        seq_so_far = planning_backups[seq_start[0]:, 0:4]
        s_n = int(seq_so_far[-1, 3])  # Final state reached in the last planning step
        probs = np.zeros(self.Q[s_n].shape)
        # Appended experience is sampled greedily:
        probs[self.Q[s_n] == max(self.Q[s_n])] = 1 / np.sum(self.Q[s_n] == max(self.Q[s_n]))
        #  Select action to append using the same action selection policy used in real experience
        a_n = np.random.choice(self.mdp.action_space.size, p = probs)
        s_n_next = self.exp_last_s_next[s_n, a_n]  # Resulting state from taking action an in state sn
        r_n = self.exp_last_rew[s_n, a_n]  # Reward received on this step only
        next_step_is_nan = np.isnan(s_n_next) or np.isnan(
            r_n)  # Check whether the retrieved rew and stp1 are NaN
        # Check whether a loop is formed
        next_step_is_repeated = np.isin(s_n_next, [seq_so_far[:, 0], seq_so_far[:, 3]])
        # p.s. Notice that we can't enforce that planning is done only when the next state is not
        # repeated or doesn't form a loop. The reason is that the next step needs to be derived
        # 'on-policy', otherwise the Q-values may not converge.

        # If loops are  allowed and next state is not repeated, then expand this backup
        if not next_step_is_nan and (params.allow_loops or not next_step_is_repeated):
            # Add one row to seq_updated (i.e., append one transition). Notice that seq_updated has many
            # rows, one for each appended step
            seq_updated = np.append(seq_so_far, np.array([[s_n, a_n, r_n, s_n_next]]), axis=0)
            plan_exp.append(seq_updated)
        return plan_exp

    def do_planning(self, params, s):
        planning_backups = np.empty((0, 5))
        backups_gain = []  # List of GAIN for backups executed
        backups_need = []  # List of NEED for backups executed
        backups_EVM = []  # List of EVM for backups executed
        backups_TD = []  # List of (abs(TD)) for backups executed (in case of PS)

        for p in range(params.planning_steps):

            plan_exp = self.get_plan_exp(params)

            # Expand previous backup with one extra action
            if params.expand_further and planning_backups.shape[0] > 0:
                plan_exp = self.expand(params, plan_exp, planning_backups)

            gain, sa_gain = gain_term(plan_exp, params, self.Q.copy())

            need, SR_or_SD = need_term(params ,plan_exp, s, self.T.copy())

            EVM = get_EVM(params, plan_exp, gain, need)

            # PERFORM THE UPDATE
            opport_cost = np.nanmean(np.array(self.list_exp)[:, 2])  # Average expected reward from a random act
            EVM_thresh = min(opport_cost, params.EVM_thresh)  # if EVM_thresh==Inf, threshold is opport_cost

            if max(EVM) > EVM_thresh:
                # Identify state-action pairs with highest priority
                max_EVM_idx = self.get_max_EVM_idx(EVM, plan_exp)

                plan_exp_arr = np.array(plan_exp, dtype=object)
                if len(plan_exp_arr[max_EVM_idx].shape) == 1:
                    plan_exp_arr_max = np.expand_dims(plan_exp_arr[max_EVM_idx], axis=0)
                else:
                    plan_exp_arr_max = np.expand_dims(plan_exp_arr[max_EVM_idx][-1], axis=0)

                for n in range(plan_exp_arr_max.shape[0]):
                    # Retrieve information from this experience
                    s_plan = int(plan_exp_arr_max[n][0])
                    a_plan = int(plan_exp_arr_max[n][1])
                    # Individual rewards from this step to end of trajectory
                    rew_to_end = plan_exp_arr_max[n:][:, 2]
                    # Notice the use of '-1' instead of 'n', meaning that stp1_plan is the final state of the
                    # trajectory
                    stp1_plan = int(plan_exp_arr_max[-1][3])

                    # Discounted cumulative reward from this step to end of trajectory
                    n_plan = np.size(rew_to_end)
                    r_plan = np.dot(np.power(params.gamma, np.arange(0, n_plan)), rew_to_end)

                    # ADD PLAN Q_LEARNING UPDATES TO Q_LEARNING FUNCTION
                    stp1_value = np.max(self.Q[stp1_plan])
                    Q_target = r_plan + (params.gamma ** n_plan) * stp1_value
                    self.Q[s_plan, a_plan] += params.alpha * (Q_target - self.Q[s_plan, a_plan])

                # self.times_for_EVB[self.num_episodes][p - 1] = time.perf_counter() - times_for_EVB

                # List of planning backups (to be used in creating a plot with the full planning trajectory/trace)
                backups_gain.append(gain[max_EVM_idx][0])  # List of GAIN for backups executed
                backups_need.append(need[max_EVM_idx][0])  # List of NEED for backups executed
                backups_EVM.append(EVM[max_EVM_idx])  # List of EVM for backups executed

                if planning_backups.shape[0] > 0:
                    planning_backups = np.vstack(
                        [planning_backups, np.append(plan_exp_arr_max, plan_exp_arr_max.shape[0])])
                elif planning_backups.shape[0] == 0:
                    planning_backups = np.append(plan_exp_arr_max,
                                                 plan_exp_arr_max.shape[0]).reshape(1, planning_backups.shape[1])
                else:
                    err_msg = 'planning_backups does not have the correct shape. It is {} but should have a ' \
                              'length equal to 1 or 2, e.g. (5,) or (2, 5)'.format(planning_backups.shape)
                    raise ValueError(err_msg)
                p += 1  # Increment planning counter
            else:
                break
        self.saver.replay.state.append(planning_backups[:, 0])
        self.saver.replay.action.append(planning_backups[:, 1])


    def target(self, gamma, s_next, a, r):
        """
        1-step target
        """
        return r + gamma * np.max(self.Q[s_next,:])

    def start(self, params):
        s = self.mdp.reset(uniform = True)
        while s in self.mdp.terminal_states:
            s = self.mdp.reset(uniform = True)
        done = self.mdp.done()
        if not params.start_random:
            self.mdp.current_state = 0 # start state
            s = self.mdp.current_state
        return s, done

    def update_transi(self, s, s_next, T_learning_rate):
            target_vector = np.zeros(self.mdp.nb_states)
            target_vector[s_next] = 1 # Update transition matrix
            self.T[s, : ] += T_learning_rate  * (target_vector - self.T[s, :]) # Shift corresponding row of T towards targVec

    def update_exp(self, s, a, r, s_next):
            self.list_exp.append([s, a, r, s_next]) # Add transition to expList
            self.saver.list_exp.append([s, a, r, s_next])
            self.exp_last_s_next[s, a] = s_next # next state from last experience of this state/action
            self.exp_last_rew[s, a] = r # rew from last experience of this state/action

    def do_backup_all_trace(self, alpha, delta):
        self.Q += (alpha * self.eTr ) * delta # TD learning

    def update_elig_traces(self, s, a):
        self.eTr[s, :] = 0
        self.eTr[s, a] = 1

    def decay_elig_traces(self, gamma, lambda_):
        self.eTr *= gamma * lambda_ # Decay eligibility trace

    def learn(self, params, seed):
        self.mdp.timeout = params.max_episode_steps

        if params.pre_explore:
            self.pre_explore()
            self.T = normalize_mat(self.T)
            self.T = np.nan_to_num(self.T)

        if params.transi_goal_to_start:
            for term_state in self.mdp.terminal_states:
                if not params.start_random:
                    self.T[term_state, :] = 0 # transition from goal to anywhere but start is not allowed
                    self.T[term_state, 0] = 1 # transition from goal to start

                else:
                    self.T[term_state, :-1] = 1/(self.mdp.nb_states - 1) # transition from goal to any state is uniform
                    assert self.T[term_state,  -1] == 0

        tot_reward = 0
        ep = 0
        steps_to_done = 0
        ep_reward = 0
        previous_was_goal = False
        np.random.seed(seed)
        s, done = self.start(params)
        while ep < params.episodes:

            while not done:
                # action selection
                proba_a = proba(self.Q[s], params.action_policy, params)
                a = np.random.choice(self.mdp.action_space.size, p = proba_a)
                # perform action
                s_next, r, done, _ = self.mdp.step(a)
                ep_reward += r

                # update transi matrix and experience list
                self.update_transi(s, s_next, params.T_learning_rate)
                self.update_exp(s, a, r, s_next)

                ### Q-Learning ###
                delta = self.target(params.gamma, s_next, a, r) - self.Q[s,a] # prediction error
                self.update_elig_traces(s, a)
                self.do_backup_all_trace(params.alpha , delta)
                self.decay_elig_traces(params.gamma, params.lambda_)

                ## Planning ###
                if params.plan_only_start_end: #Only do replay if either current or last trial was a goal state
                    if previous_was_goal or s_next in self.mdp.terminal_states:
                        self.do_planning(params, s)
                else:
                    self.do_planning(params, s)

                # move
                s = s_next
                self.saver.num_episodes.append(ep)
                steps_to_done +=1
                previous_was_goal = False
                if s in self.mdp.terminal_states:
                    previous_was_goal = True

            # END EPISODE #
            #get next start location
            s_next, done = self.start(params) # agent is currently already in s_next in mdp
            # print(s_next)
            if s in self.mdp.terminal_states:
                if params.transi_goal_to_start:
                    self.update_transi(s, s_next, params.T_learning_rate)
                    # self.update_exp(s, a, r, s_next)
                    self.list_exp.append([s, np.NaN, np.NaN, s_next])

            self.eTr = np.zeros((self.mdp.nb_states, self.mdp.action_space.size))
            tot_reward += ep_reward
            self.saver.steps_to_exit.append(steps_to_done)
            self.saver.list_Q.append(self.Q)


            print("#### EPISODE {} ####".format(ep + 1))
            print("TRAIN: {}".format(steps_to_done))

            ep += 1
            steps_to_done = 0
            ep_reward = 0
            # move agent to start location
            s = s_next

        # return {"train" : steps_to_exit_or_timeout , "list_Q" : list_Q}
