import numpy as np

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
