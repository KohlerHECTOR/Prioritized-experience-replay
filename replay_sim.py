import numpy as np
from utils import normalize_mat
from mazemdp.toolbox import softmax
from mazemdp.mdp import Mdp

class Agent():
    def __init__(self, mdp):
        self.mdp = mdp
        self.Q = np.zeros((self.mdp.nb_states, self.mdp.action_space.size)) # state-action value function
        self.T = np.zeros((self.mdp.nb_states, self.mdp.nb_states)) # state-state transition proba
        self.eTr = self.Q = np.zeros((self.mdp.nb_states, self.mdp.action_space.size)) # eligibility matrix
        self.list_exp = [] # list to store individual experiences (s, a, r, s')
        self.exp_last_s_next = np.empty((self.mdp.nb_states, self.mdp.action_space.size)) # list to store next states
        self.exp_last_reward = np.empty((self.mdp.nb_states, self.mdp.action_space.size)) # list to store next rewards
        self.nb_episodes = 0 # keep track of nb times we reached end of maze

# TODO: code to store sim data using a sim_data class


    def pre_explore(self):
        self.mdp.reset()
        for s in range(self.mdp.nb_states):
            for a in range(self.mdp.action_space.size):
                self.mdp.current_state = s # to force the execution of every action in each state
                s_next, _, _, _ = self.mdp.step(a)
                self.list_exp.append([s, a, 0, s_next]) # update list of experiences
                self.exp_last_s_next[s, a] = s_next # update next_state model
                self.exp_last_reward[s, a] = 0 # update reward model
                if s not in self.mdp.terminal_states:
                    self.T[s, s_next] += 1 # update transition matrix
                    
    def do_planning(self, planning_steps):
        # create list of 1-step backups based on 1-step models
        # plan_exp =
        pass
    def backup(self, s, a, r, s_next):
        """
        k-step Bellman backup
        """
        self.Q[s,a] = self.Q[s,a] + self.alpha * (self.target(s_next, a, r) - self.Q[s,a])

    # def start_state(self, params):
    #     if not params.start_random:
    #         self.mdp.current_state = 0 # start state
    #         s = self.mdp.current_state
    #     else:
    #         pass

    def target(self, s_next, a, r):
        """
        1-step target
        """
        return r + self.mdp.gamma * np.max(self.Q[s_next,:])
    def transition_from_goal(self, params, s_next):
        if params.transi_goal_to_start:
            s_next = self.start_state(params)

        return s_next

    def learn(self, params):
        if params.pre_explore:
            self.pre_explore()
            self.T = normalize_mat(self.T)

        # # add transition from goal to start
        # if params.transi_goal_to_start:
        #     if params.start_random:
        #         self.T[-1, :] += 1  # goal state is last state
        #         self.T[-1, :] /= self.mdp.nb_states # transition from goal state to any starting state is uniform
        #     else:
        #         self.T[-1: 0] = 1 # goal state is last state, start state is first state


        s = self.mdp.reset()

        if not params.start_random:
            self.mdp.current_state = 0 # start state
            s = self.mdp.current_state
        ep_reward = 0
        # start episode

        for steps in range(params.max_episode_steps):

            # action selection
            prob_a = softmax(self.Q, s, params.tau)
            a = np.random.choice(np.arange(self.mdp.action_space.size), p = prob_a)
            # perform action
            s_next, r, done, _ = self.mdp.step(a)
            # if s in self.mdp.terminal_states:
            #     s_next = self.transition_from_goal(params)
            ep_reward += r

            # update transi matrix and experience list
            target_vector = np.zeros(self.mdp.nb_states)
            target_vector[s_next] = 1
            self.T[s_next, : ] += params.T_learning_rate  * (target_vector - self.T[s_next, :])
            self.list_exp.append([s, a, r, s_next])
            self.exp_last_s_next[s, a] = s_next
            self.exp_last_reward[s, a] = r

            # TODO: log data

            ### Q-Learning ###
            delta = self.target(s_next, a, r) - self.Q[s,a] # prediction error
            # update elig traces
            self.eTr[s, :] = 0
            self.eTr[s, a] = 1
            self.Q += (params.alpha * self.eTr ) * delta # TD learning
            self.eTr *= self.mdp.gamma * params.lambda_

            ### Planning ###
            if params.plan_only_start_end:
                if (s == 0 and ep_reward > 0) or s in self.mdp.terminal_states: # current state is either start or goal
                    self.do_planning(params.planning_steps)
            else:
                self.do_planning(params.planning_steps)
