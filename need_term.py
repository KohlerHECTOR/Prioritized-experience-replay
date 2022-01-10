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
    for i, exps in enumerate(plan_exp):
        need_i = []
        for j , e in enumerate(exps):
            need_i.append(SR_or_SD[int(e[0])])
        need.append(need_i)

    return need, SR_or_SD
