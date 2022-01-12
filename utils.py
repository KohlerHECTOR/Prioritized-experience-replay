import numpy as np
from mazemdp.toolbox import softmax, egreedy
import matplotlib.pyplot as plt
from scipy.linalg import eig

def proba(Q_mean, policy, params):
    probs = np.zeros_like(Q_mean)
    if policy == "softmax":
        probs = np.exp(Q_mean / params.tau) / np.sum(np.exp(Q_mean / params.tau));
    elif policy == "greedy":
        Q_best_idx = np.where(Q_mean == Q_mean.max())[0]
        if (Q_mean == Q_mean.max()).all():
            probs[Q_best_idx] = 1/len(Q_best_idx)
        else:
            probs[Q_best_idx] = (1-params.epsilon)/len(Q_best_idx)
            probs = np.nan_to_num(probs)
            probs += params.epsilon/len(Q_mean)

    assert np.sum(probs) < 1.01 and np.sum(probs) > 0.99, "proba dont sum to 1"
    return probs
########################### EVM STUFF ###############################################
def get_EVM(params, plan_exp, gain, need):

    mask_need = 1
    if params.set_all_need_to_1:
        mask_need = 0

    mask_gain = 1
    if params.set_all_gain_to_1:
        mask_gain = 0
    # Expected value of memories

    EVM = np.full((len(plan_exp)), np.nan)
    for i in range(len(plan_exp)):
        if len(plan_exp[i].shape) == 1:
            EVM[i] = (need[i][-1] ** mask_need) * (max(gain[i], params.baseline_gain) ** mask_gain)
        elif len(plan_exp[i].shape) == 2:
            EVM[i] = 0
            for x in range(len(plan_exp[i])):
                EVM[i] += (need[i][-1] ** mask_need) * (max(gain[i][-1], params.baseline_gain) ** mask_gain)
        else:
            err_msg = 'plan_exp[i] does not have the correct shape. It is {} but should have a ' \
                      'length equal to 1 or 2, e.g. (4,) or (2, 4)'.format(plan_exp[i].shape)
            raise ValueError(err_msg)
    return EVM

def gain_term(plan_exp, params, Q):

    gain = []
    sa_gain = np.empty(Q.shape)
    sa_gain.fill(np.nan)
    for i in range(len(plan_exp)):
        this_exp = plan_exp[i]
        if len(this_exp.shape) == 1:
            this_exp = np.expand_dims(this_exp, axis=0)
        gain.append(np.repeat(np.nan, this_exp.shape[0]))

        for j in range(this_exp.shape[0]):
            Q_mean = np.copy(Q[int(this_exp[j, 0])])
            Qpre = Q_mean.copy()
            # Policy BEFORE backup
            pA_pre = proba(Q_mean, params.plan_policy, params)

            # Value of state stp1
            stp1i = int(this_exp[-1, 3])
            stp1_value = np.max(Q[stp1i])

            act_taken = int(this_exp[j, 1])
            steps_to_end = this_exp.shape[0] - (j + 1)
            rew = np.dot(np.power(params.gamma, np.arange(0, steps_to_end + 1)), this_exp[j:, 2])
            Q_target = rew + np.power(params.gamma, steps_to_end + 1) * stp1_value
            Q_mean[act_taken] += params.alpha * (Q_target - Q_mean[act_taken])

            # policy AFTER backup
            pA_post = proba(Q_mean, params.plan_policy, params)

            # calculate gain
            EV_pre = np.sum(np.multiply(pA_pre, Q_mean))
            EV_post = np.sum(np.multiply(pA_post, Q_mean))
            gain[i][j] = EV_post - EV_pre
            Qpost = Q_mean.copy()
            # Save on gain[s, a]
            sti = int(this_exp[j, 0])
            if np.isnan(sa_gain[sti, act_taken]):
                sa_gain[sti, act_taken] = gain[i][j]
            else:
                sa_gain[sti, act_taken] = max(sa_gain[sti, act_taken], gain[i][j])
    return gain, sa_gain

def need_term(params, plan_exp, s, T):
    need = []
    if params.online_offline == "offline":
        D, W, V = eig(T, left = True) # Calculate eigenvectors and eigenvalues of Transition matrix
        SD = np.abs(W[:,1].T) # Stationary distribution of the MDP
        SR_or_SD = SD;
    elif params.online_offline == "online":
        # Calculate Successor Representation
        SR = np.linalg.inv(np.eye(len(T)) - params.gamma * T)
        SRi = SR[s,:]; # Calculate the Successor Representation for the current state
        SR_or_SD = SRi;

    # Calculate need-term for each experience in nStepExps
    for i in range(len(plan_exp)):
        this_exp = plan_exp[i]
        if len(this_exp.shape) == 1:
            this_exp = np.expand_dims(this_exp, axis=0)
        need.append(np.repeat(np.nan, this_exp.shape[0]))
        for j in range(this_exp.shape[0]):
            need[i][j] = SR_or_SD[int(this_exp[j, 0])]
    return need, SR_or_SD
#################################################################################
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
    plt.fill_between(np.arange(len(mean_data)), mean_data + std_data, mean_data - std_data, alpha = 0.1)

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
