import numpy as np
from scipy.linalg import eig

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
