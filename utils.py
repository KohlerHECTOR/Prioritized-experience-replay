import numpy as np
from mazemdp.toolbox import softmax, egreedy
import matplotlib.pyplot as plt
def normalize_mat(matrix):
    row_sums = matrix.sum(axis=1)
    new_matrix = matrix / row_sums[:, np.newaxis]

    return new_matrix

def evaluate(list_q , mdp, params):
    """
    simple function to evaluate a policy from a Q table
    """
    mdp.gamma = params.gamma
    mdp.timeout = params.max_episode_steps
    all_steps_to_exit = []
    for i, Q in enumerate(list_q):

        ## BEGIN EPISODE ##
        s = mdp.reset()
        done = mdp.done()
        # np.random.seed(i)
        steps_to_exit = 0
        while not done:
            # Draw an action using a softmax policy
            if params.action_policy == "softmax":
                prob_a = softmax(Q, s, params.tau)
                a = np.random.choice(np.arange(mdp.action_space.size), p = prob_a)
            elif params.action_policy =="greedy":
                a = egreedy(Q, s, params.epsilon)

            s, r, done, _  = mdp.step(a)
            # s, r, done, _  = self.mdp.step(np.argmax(self.Q[s,:]))
            steps_to_exit += 1
        all_steps_to_exit.append(steps_to_exit)

    return all_steps_to_exit

def to_plot(data, label = None):
    mean_data = np.mean(data, axis = 0)
    std_data = np.std(data, axis = 0)
    plt.plot(mean_data, label = label)
    plt.fill_between(np.arange(len(mean_data)), mean_data + std_data, mean_data - std_data, alpha = 0.4)

# PLACEHOLDERS FOR ANALYSIS

class SimuData():
    def __init__(self, replay):
        self.num_episodes = []
        self.replay = replay
        self.list_exp = []
        self.steps_to_exit = []
        self.list_Q = []

class Replay():
    def __init__(self):
        self.action = []
        self.state = []
