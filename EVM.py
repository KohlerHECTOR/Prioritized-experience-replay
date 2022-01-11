import numpy as np

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
