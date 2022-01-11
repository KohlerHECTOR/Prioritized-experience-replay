import numpy as np
from proba import proba

def gain_term(plan_exp, params, Q):
    # Gain = []
    # sa_Gain = np.zeros_like(Q)
    # for i, exps  in enumerate(plan_exp):
    #     gain_i = []
    #     for j, e in enumerate(exps): # remember a single experience e is (s, a , r , s_next)
    #         print(e)
    #         s_e  = int(e[0]) # e[0] correspons to state s of the experience
    #         a_e = int(e[1]) # e[1] corresponds to action a of the experience
    #         s_next_e = int(e[-1]) # e[-1] corresponds to next state of experience
    #         Q_mean = Q[s_e].copy()
    #         Q_pre = Q_mean.copy()
    #         # Policy before
    #         prob_a_pre = proba(Q_mean, params.plan_policy, params)
    #
    #         s_next_val = np.max(Q[int(exps[-1, -1])]) # value of s_next of the last experience in the seq
    #
    #         steps_to_end = len(exps) - (j+1) # remaining steps to end of trajectory
    #         rew_to_end = np.sum((params.gamma ** np.arange(steps_to_end + 1)) * exps[j: , 2])
    #         Q_target = rew_to_end + (params.gamma ** (steps_to_end + 1)) * s_next_val
    #         Q_mean[a_e] += params.alpha * (Q_target - Q_mean[a_e])
    #
    #         # Policy after backup
    #         prob_a_post = proba(Q_mean, params.plan_policy, params)
    #         # Calculate Gain
    #         EV_pre = np.sum(prob_a_pre * Q_mean)
    #         EV_post = np.sum(prob_a_post * Q_mean)
    #         gain_i.append(EV_post - EV_pre)
    #         Q_post = Q_mean.copy()
    #         # save on Gain[s, a]
    #         sa_Gain[s_e, a_e] = max(sa_Gain[s_e, a_e], gain_i[-1])
    #     Gain.append(gain_i)
    #
    # return Gain, sa_Gain
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
