import numpy as np
from utils import normalize_mat
from mazemdp.toolbox import softmax, egreedy
from mazemdp.mdp import Mdp
from scipy.linalg import eig
import sys

class Agent():

    def __init__(self, mdp, params):
        self.mdp = mdp
        self.mdp.gamma = params.gamma
        self.Q = np.zeros((self.mdp.nb_states, self.mdp.action_space.size)) # state-action value function
        self.T = np.zeros((self.mdp.nb_states, self.mdp.nb_states)) # state-state transition proba
        self.eTr = np.zeros((self.mdp.nb_states, self.mdp.action_space.size)) # eligibility matrix
        self.list_exp = [] # list to store individual experiences (s, a, r, s')
        self.exp_last_s_next = np.zeros((self.mdp.nb_states, self.mdp.action_space.size), dtype = int) # list to store next states
        self.exp_last_reward = np.zeros((self.mdp.nb_states, self.mdp.action_space.size), dtype = int) # list to store next rewards
        self.nb_episodes = 0 # keep track of nb times we reached end of maze

# TODO: code to store sim data using a sim_data class

    def pre_explore(self):
        # To get initial Model
        self.mdp.reset()
        for s in range(self.mdp.nb_states):
            if s not in self.mdp.terminal_states:
                for a in range(self.mdp.action_space.size):
                    self.mdp.current_state = s # to force the execution of every action in each state
                    s_next, _, _, _ = self.mdp.step(a)
                    self.list_exp.append([s, a, 0, s_next]) # update list of experiences
                    self.exp_last_s_next[s, a] = s_next # update next_state model
                    self.exp_last_reward[s, a] = 0 # update reward model
                    self.T[s, s_next] += 1 # update transition matrix

    def need_term(self, params, plan_exp, s):
        need = []
        if params.online_offline == "offline":
            D, W, V = eig(self.T, left = True) # Calculate eigenvectors and eigenvalues of Transition matrix
            SD = np.abs(W[:,1].T) # Stationary distribution of the MDP
            SR_or_SD = SD;
        elif params.online_offline == "online":
             # Calculate Successor Representation
             SR = np.linalg.inv(np.eye(len(self.T)) - params.gamma * self.T)
             SRi = SR[s,:]; # Calculate the Successor Representation for the current state
             SR_or_SD = SRi;

        # Calculate need-term for each experience in nStepExps
        for i, exps in enumerate(plan_exp):
            need_i = []
            for j , e in enumerate(exps):
                need_i.append(SR_or_SD[int(e[0])])
            need.append(need_i)

        return need, SR_or_SD

    def gain_term(self, plan_exp, params):
        Gain = []
        sa_Gain = np.zeros_like(self.Q)
        for i, exps  in enumerate(plan_exp):
            gain_i = []
            for j, e in enumerate(exps): # remember a single experience e is (s, a , r , s_next)
                s_e  = int(e[0]) # e[0] correspons to state s of the experience
                a_e = int(e[1]) # e[1] corresponds to action a of the experience
                s_next_e = int(e[-1]) # e[-1] corresponds to next state of experience
                Q_mean = self.Q[s_e].copy()
                Q_pre = Q_mean.copy()
                # Policy before
                if params.plan_policy == "softmax":
                    prob_a_pre = np.exp((Q_mean / params.tau)).round(5) / np.sum(np.exp((Q_mean / params.tau)).round(5))

                elif params.plan_policy == "greedy":
                    r = np.random
                    if r.random() < params.epsilon:
                        prob_a_pre = np.ones(self.mdp.action_space.size) / self.mdp.action_space.size
                    else:
                        # choose randomly about the actions with a max value
                        prob_a_pre = np.zeros(self.mdp.action_space.size)
                        prob_a_pre[np.random.choice(np.where(Q_mean == Q_mean.max())[0])] = 1

                # if int(exps[-1,-1]) <  self.mdp.nb_states:
                s_next_val = np.max(self.Q[int(exps[-1, -1])]) # value of s_next of the last experience in the seq
                # else:
                #     s_next_val = 1

                steps_to_end = len(exps) - (j+1) # remaining steps to end of trajectory
                rew_to_end = np.sum((params.gamma ** np.arange(steps_to_end + 1)) * exps[j: , 2])
                Q_target = rew_to_end + (params.gamma ** (steps_to_end + 1)) * s_next_val
                Q_mean[a_e] += params.alpha * (Q_target - Q_mean[a_e])

                # Policy after backup
                if params.plan_policy == "softmax":
                    prob_a_post = np.exp((Q_mean / params.tau)).round(5) / np.sum(np.exp((Q_mean / params.tau)).round(5))

                elif params.plan_policy == "greedy":
                    r = np.random
                    if r.random() < params.epsilon:
                        prob_a_post = np.ones(self.mdp.action_space.size) / self.mdp.action_space.size
                    else:
                        # choose randomly about the actions with a max value
                        prob_a_post = np.zeros(self.mdp.action_space.size)
                        prob_a_post[np.random.choice(np.where(Q_mean == Q_mean.max())[0])] = 1

                # Calculate Gain
                EV_pre = np.sum(prob_a_pre * Q_mean)
                EV_post = np.sum(prob_a_post * Q_mean)
                gain_i.append(EV_post - EV_pre)
                # print(gain_i[-1])
                Q_post = Q_mean.copy()
                # save on Gain[s, a]
                sa_Gain[s_e, a_e] = max(sa_Gain[s_e, a_e], gain_i[-1])
                # print(sa_Gain[s_e, a_e])
            Gain.append(gain_i)

        return Gain, sa_Gain
    def get_plan_exp(self, params):
        # Data structure stuff
        # Create a list of 1-step backups based on 1-step models
        tmp = np.tile(np.arange(self.mdp.nb_states), self.mdp.action_space.size)
        tmp_bis = np.repeat(np.arange(self.mdp.action_space.size), self.mdp.nb_states)
        plan_exp = np.column_stack((tmp, tmp_bis , self.exp_last_reward.flatten(), self.exp_last_s_next.flatten()))
        plan_exp = plan_exp.reshape((plan_exp.shape[0], 1 , plan_exp.shape[-1]))
        # print(self.exp_last_s_next)
        if params.remove_samestate: # Remove actions that lead to same state (optional) -- e.g. hitting the wall
            idx = []
            for i in range(len(plan_exp)):
                if plan_exp[i][0][0] != plan_exp[i][0][3]:
                    idx.append(i)
            plan_exp = plan_exp[idx]

        plan_exp = np.nan_to_num(plan_exp)
        plan_exp = list(plan_exp)

        return plan_exp

    def tie_break(self, max_EVM_idx, plan_exp):

        nb_steps = np.array([len(plan_exp[i]) for i in range(len(plan_exp))]) #  number of total steps on this trajectory
        max_EVM_idx = max_EVM_idx[nb_steps[max_EVM_idx] == min(nb_steps[max_EVM_idx])]# Select the one corresponding to a shorter trajectory
        if len(max_EVM_idx)>1: # If there are still multiple items with equal gain (and equal length)
            max_EVM_idx = np.random.choice(max_EVM_idx)# ... select one at random

        return max_EVM_idx

    def get_max_EVM_idx(self, EVM, plan_exp):

        max_EVM_idx = np.where(EVM == np.max(EVM))[0]
        if len(max_EVM_idx)>1: # If there are multiple items with equal gain
            max_EVM_idx = self.tie_break(max_EVM_idx, plan_exp)

        return int(max_EVM_idx)

    def expand(self, params, plan_exp, planning_backups):

        seq_start = np.where(np.array(planning_backups)[:,-1] == 1)[0][-1]
        seq_so_far = np.array(planning_backups)[seq_start: , : 4]
        s_n = seq_so_far[-1, -1]

        # if not s_n >= self.mdp.nb_states:
        probs = np.zeros_like(self.Q[s_n])
        probs[np.where(self.Q[s_n] == np.max(self.Q[s_n]))[0]] = 1 / len(np.where(self.Q[s_n] == np.max(self.Q[s_n]))[0]) # appended experience is sampled greedily

        a_n = np.random.choice(self.mdp.action_space.size, p = probs) # Select action to append using the same action selection policy used in real experience
        s_n_next = self.exp_last_s_next[s_n, a_n] # Resulting state from taking action an in state sn
        r_n = self.exp_last_reward[s_n, a_n] # Reward received on this step only

        next_step_is_nan = np.isnan(self.exp_last_s_next[s_n, a_n]) or np.isnan(self.exp_last_reward[s_n, a_n]) # is a bool

        next_step_is_repeated = s_n_next in seq_so_far[:, 0] or s_n_next in seq_so_far[:, 3] # Check whether a loop is formed. Bool as well
        # Notice that we cant enforce that planning is done only when the next state is not repeated or don't form aloop. The reason is that the next step needs to be derived 'on-policy', otherwise the Q-values may not converge.
        if not next_step_is_nan and (params.allow_loops or not next_step_is_repeated): # If loops are not allowed and next state is repeated, don't expand this backup
            seq_updated = np.concatenate((seq_so_far, np.array([s_n, a_n, r_n, s_n_next]).reshape(1, 4)), axis = 0)
            plan_exp.append(seq_updated)

        return plan_exp

    def do_planning(self, params, s):
        planning_backups = []
        backups_need = []
        backups_gain = []
        backups_EVM = []
        # print("---- Planning ----")

        for plan in range(params.planning_steps):

            plan_exp = self.get_plan_exp(params)

            #Expand previous backup with one extra action
            if params.expand_further and len(planning_backups) > 0:
                plan_exp = self.expand(params, plan_exp , planning_backups)

            Gain, sa_Gain = self.gain_term(plan_exp, params)
            need, SR_or_SD = self.need_term(params ,plan_exp, s)

            mask_need = 1
            if params.set_all_need_to_1:
                mask_need = 0

            mask_gain = 1
            if params.set_all_gain_to_1:
                mask_gain = 0


            EVM = [] # Expected value of memories
            for i, exps in enumerate(plan_exp):
                EVM.append(np.sum((need[i][-1] ** mask_need) * (np.maximum(Gain[i], params.baseline_gain))) ** mask_gain) # Use the need from the last (appended) state

            opport_cost = np.array(self.list_exp)[:,2].mean() # Average expected reward from a random act
            EVM_thresh = min(opport_cost, params.EVM_thresh) # if EVMthresh==Inf, threshold is opportCost

            if np.max(EVM) > EVM_thresh:
                # Identify state-action pairs with highest priority
                max_EVM_idx = self.get_max_EVM_idx(EVM, plan_exp)
                # N-step backup with most useful traj
                for n, exp in enumerate(plan_exp[max_EVM_idx]):
                    s_plan = int(exp[0])
                    a_plan =int(exp[1])
                    s_next_plan = int(plan_exp[max_EVM_idx][-1, -1]) # Notice the use of 'end' instead of 'n', meaning that stp1_plan is the final state of the trajectory
                    rew_to_end = plan_exp[max_EVM_idx][n:, 2] # Individual rewards from this step to end of trajectory
                    r_plan = np.sum(params.gamma ** np.arange(len(rew_to_end)) * rew_to_end)
                    n_plan = len(rew_to_end)

                    s_next_value = np.max(self.Q[s_next_plan])

                    Q_target = r_plan + (params.gamma ** n_plan) * s_next_value

                    self.Q[s_plan, a_plan] += params.alpha * (Q_target - self.Q[s_plan, a_plan])

                backups_gain.append(Gain[max_EVM_idx]) # List of GAIN for backups executed
                backups_need.append(need[max_EVM_idx]) # List of NEED for backups executed
                backups_EVM.append(EVM[max_EVM_idx]) # List of EVM for backups executed
                planning_backups.append(np.concatenate((plan_exp[max_EVM_idx][-1], [len(plan_exp[max_EVM_idx])]))) # Notice that the first column of planning_backups corresponds to the start state of the final transition on a multistep sequence

            else:
                break

    def do_backup(self, s, a, r, s_next):
        """
        Bellman backup
        """
        self.Q[s,a] = self.Q[s,a] + self.alpha * (self.target(s_next, a, r) - self.Q[s,a])

    def target(self, s_next, a, r):
        """
        1-step target
        """
        return r + self.mdp.gamma * np.max(self.Q[s_next,:])

    def start(self, params):
        s = self.mdp.reset(uniform = True)
        while s in self.mdp.terminal_states:
            s = self.mdp.reset(uniform = True)
        done = self.mdp.done()
        if not params.start_random:
            self.mdp.current_state = 0 # start state
            s = self.mdp.current_state
        return s, done

    def select_action(self, s, params):
        # action selection
        if params.action_policy == "softmax":
            prob_a = softmax(self.Q, s, params.tau)
            a = np.random.choice(np.arange(self.mdp.action_space.size), p = prob_a)
        elif params.action_policy =="greedy":
            a = egreedy(self.Q, s, params.epsilon)
        return a

    def update_transi(self, s, s_next, params):
            target_vector = np.zeros(self.mdp.nb_states)
            target_vector[s_next] = 1 # Update transition matrix
            self.T[s, : ] += params.T_learning_rate  * (target_vector - self.T[s, :]) # Shift corresponding row of T towards targVec

    def update_exp(self, s, a, r, s_next):
            self.list_exp.append([s, a, r, s_next]) # Add transition to expList
            self.exp_last_s_next[s, a] = s_next # next state from last experience of this state/action
            self.exp_last_reward[s, a] = r # rew from last experience of this state/action

    def do_backup_all_trace(self, params, delta):
        self.Q += (params.alpha * self.eTr ) * delta # TD learning

    def update_elig_traces(self, s, a):
        self.eTr[s, :] = 0
        self.eTr[s, a] = 1

    def decay_elig_traces(self, params):
        self.eTr *= self.mdp.gamma * params.lambda_ # Decay eligibility trace

    def learn(self, params):
        list_Q = []
        steps_to_exit_or_timeout = []
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
                    self.T[term_state, :] = 1/self.mdp.nb_states # transition from goal to any state is uniform

        tot_reward = 0

        ep = 0
        steps_to_done = 0
        ep_reward = 0
        starting = True
        s, done = self.start(params)
        while ep < params.episodes:

            while not done:

                # action selection
                a = self.select_action(s, params)
                # perform action
                s_next, r, done, _ = self.mdp.step(a)
                # if s_next == 47: print(s, "ok")
                ep_reward += r

                # update transi matrix and experience list

                self.update_transi(s, s_next, params)
                self.update_exp(s, a, r, s_next)

                ### Q-Learning ###
                # if s in self.mdp.terminal_states:
                #     delta = r
                # else:
                delta = self.target(s_next, a, r) - self.Q[s,a] # prediction error

                self.update_elig_traces(s, a)
                self.do_backup_all_trace(params , delta)
                self.decay_elig_traces(params)


                if params.plan_only_start_end: #Only do replay if either current or last trial was a goal state
                    if (starting and tot_reward > 0) or s in self.mdp.terminal_states:
                        self.do_planning(params, s)
                else:
                    self.do_planning(params, s)


                # move

                s = s_next
                starting = False
                steps_to_done +=1

            # END EPISODE #
            #get next start location
            s_next, done = self.start(params)
            # print(s_next)
            if s in self.mdp.terminal_states:
                if params.transi_goal_to_start:
                    self.update_transi(s, s_next, params)
                    # self.update_exp(s, a, r, s_next)
                    self.list_exp.append([s, np.nan, np.nan, s_next])

            self.eTr = np.zeros((self.mdp.nb_states, self.mdp.action_space.size))
            tot_reward += ep_reward
            list_Q.append(self.Q)
            steps_to_exit_or_timeout.append(steps_to_done)

            print("#### EPISODE {} ####".format(ep))
            print("TRAIN: {}".format(steps_to_done))
            # bug dans  Simple Maze MDP
            if steps_to_done == 1:
                print("last episode not counted")
                ep-=1
                steps_to_exit_or_timeout.pop()
                list_Q.pop()

            ep += 1
            steps_to_done = 0
            ep_reward = 0
            starting = True
            # move agent to start location
            s = s_next

        return {"train" : steps_to_exit_or_timeout , "list_Q" : list_Q}
