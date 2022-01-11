import os

import numpy as np
import matplotlib.pyplot as plt

from mazemdp.maze import build_maze, create_random_maze
from random import seed
from mazemdp import create_random_maze
from replay_sim import Agent
from utils import evaluate, to_plot, SimuData, Replay

from arguments import get_args, get_args_string


def save_figure(filename, data, title):
    to_plot(data)
    plt.xlabel("episode")
    plt.ylabel("steps until goal")
    plt.title(title)
    plt.legend()
    plt.savefig(filename)
    plt.clf()


def train_model(mdp, args):
    res_train = []
    res_list_Q = []
    for i in range(args.simulations):
        print("#### SIM NB: {}".format(i))
        replay = Replay()
        saver = SimuData(replay)
        rat = Agent(mdp, args, saver)
        rat.learn(args, seed = i)
        res_train.append(saver.steps_to_exit)
        res_list_Q.append(saver.list_Q)

    return res_train, res_list_Q

def eval_model(list_Qs, mdp, args):
    res_eval = []
    for list_Q in list_Qs:
        res_eval.append(evaluate(list_Q, mdp, args))
    return res_eval

if __name__ == '__main__':
    args = get_args()


    # Dyna q #
    # args.set_all_gain_to_1 = True
    # args.set_all_need_to_1 = True
    # args.action_policy = "greedy"
    # args.plan_policy = "greedy"
    # args.alpha = 0.1
    # args.epsilon = 0.1
    # args.transi_goal_to_start = False
    # args.max_episode_steps = 3000
    # args.expand_further = False
    # Dyna q #
    print(args)
    mdp = create_random_maze(9, 6, 0.13)
    train_data, list_Qs = train_model(mdp, args)
    filename = get_args_string(args)
    save_figure("results/train_"+filename+".png", train_data, "training")
    eval_data = eval_model(list_Qs, mdp, args)
    save_figure("results/eval_"+filename+".png", eval_data, "evaluation")
