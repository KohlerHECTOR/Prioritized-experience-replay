import argparse

def get_args_string(args):
    my_str = ""
    for _, val in args._get_kwargs():
        my_str += str(val) + "_"
    return my_str

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--pre_explore", type = bool, default = True)
    parser.add_argument("--start_random", type = bool, default = False)
    parser.add_argument("--transi_goal_to_start", type = bool, default = True)

    parser.add_argument("--plan_policy", type = str, default = "softmax")
    parser.add_argument("--plan_only_start_end", type = bool, default = True)
    parser.add_argument("--planning_steps", type = int, default = 20)
    parser.add_argument("--expand_further", type = bool, default = True)

    parser.add_argument("--set_all_need_to_1", type = bool, default = False)
    parser.add_argument("--set_all_gain_to_1", type = bool, default = False)
    parser.add_argument("--baseline_gain", type = float, default = 1e-10)
    parser.add_argument("--EVM_thresh", type = float, default = 0)

    parser.add_argument("--allow_loops", type = bool, default = False)
    parser.add_argument("--online_offline", type = str, default = "online")
    parser.add_argument("--remove_samestate", type = bool, default = True)


    parser.add_argument("--gamma", type = float, default = 0.9)
    parser.add_argument("--alpha", type = float, default = 1)
    parser.add_argument("--lambda_", type = float, default = 0)
    parser.add_argument("--T_learning_rate", type = float, default = 0.9)

    parser.add_argument("--action_policy", type = str, default = "softmax")
    parser.add_argument("--epsilon", type = float, default = 0.05)
    parser.add_argument("--tau", type = float, default = 0.2)

    parser.add_argument("--simulations", type = int, default = 50)
    parser.add_argument("--episodes", type = int, default = 50)
    parser.add_argument("--max_episode_steps", type = int, default = 1e5)

    parser.add_argument("--reward_change_proba", type = float, default = 0)
    parser.add_argument("--reward_multiplicator", type = float, default = 1)
    args = parser.parse_args()
    return args
