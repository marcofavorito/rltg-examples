import argparse

import matplotlib.pyplot as plt
import numpy as np
from RLGames.gym_wrappers.GymBreakout import GymBreakout
from RLGames.gym_wrappers.GymSapientino import GymSapientino
from rltg.agents.brains.TDBrain import QLearning, Sarsa

BREAKOUT = "breakout"
SAPIENTINO = "sapientino"
MINECRAFT = "minecraft"

name2env = {
    BREAKOUT : GymBreakout,
    SAPIENTINO: GymSapientino
}

QLEARNING = "q-learning"
SARSA = "sarsa"

name2algorithm = {
    QLEARNING: QLearning,
    SARSA: Sarsa
}

class Config(object):
    def __init__(self, episodes=10000, algorithm=QLEARNING, gamma=0.99, alpha=0.1, epsilon=0.1,
                 lambda_=1.0, reward_shaping=True, on_the_fly=False, eval=False, resume=False,
                 render=False, datadir="data", verbosity=1):
        self.episodes = episodes
        self.algorithm = algorithm
        self.gamma =    gamma
        self.alpha =    alpha
        self.epsilon =  epsilon
        self.lambda_ =  lambda_
        self.reward_shaping = reward_shaping
        self.on_the_fly = on_the_fly
        self.eval = eval
        self.resume = resume
        self.render =   render
        self.datadir = datadir
        self.verbosity = verbosity

    def __str__(self):
        return "Configs:\n" + "\n".join(["\t{} = {}".format(k,v) for k, v in sorted(self.__dict__.items())])


def mean_std_plot(y_data):
    x = list(range(len((y_data[0]))))
    y = np.mean(y_data, axis=0)
    e = np.std(y_data, axis=0)

    plt.errorbar(x, y, e, fmt="o-", ecolor='k', errorevery=5)
    plt.show()

def check_in_float_range(min_, max_, min_open=False, max_open=False):
    f = lambda x: _check_in_float_range(x, min_, max_, min_open=min_open, max_open=max_open)
    return f

def _check_in_float_range(value, min_, max_, min_open=False, max_open=False):
    ivalue = float(value)
    left_ =  ivalue > min_ if min_open else ivalue >= min_
    right_ = ivalue < max_ if max_open else ivalue <= max_
    if not left_ or not right_:
        raise argparse.ArgumentTypeError("{} is not included in {} {}, {} {}".format(
            value, "(" if min_open else "[", min_, max_, "]" if max_open else "]"
        ))
    return ivalue