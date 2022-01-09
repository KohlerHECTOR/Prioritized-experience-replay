import os

import numpy as np
import matplotlib.pyplot as plt

from mazemdp.maze import build_maze, create_random_maze
from random import seed
from mazemdp import create_random_maze
from replay_sim import Agent
from utils import evaluate

from arguments import get_args, get_args_string



def to_plot(data):
    mean_data = np.mean(data, axis = 0)
    std_data = np.std(data, axis = 0)
    plt.plot(mean_data)
    plt.fill_between(np.arange(len(mean_data)), mean_data + std_data, mean_data - std_data, alpha = 0.4)

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
        rat = Agent(mdp, args)
        data = rat.learn(args)
        res_train.append(data["train"])
        res_list_Q.append(data["list_Q"])

    return res_train, res_list_Q

def eval_model(list_Qs, mdp, args):
    res_eval = []
    for list_Q in list_Qs:
        res_eval.append(evaluate(list_Q, mdp, args))
    return res_eval

if __name__ == '__main__':
    args = get_args()
    print(args)
    mdp = create_random_maze(9, 6, 0.13)
    train_data, list_Qs = train_model(mdp, args)
    filename = get_args_string(args)
    save_figure("results/train_"+filename+".png", train_data, "training")
    eval_data = eval_model(list_Qs, mdp, args)
    save_figure("results/eval_"+filename+".png", eval_data, "evaluation")