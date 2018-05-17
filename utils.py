import os

import click
from RLGames.gym_wrappers.GymBreakout import GymBreakout
from RLGames.gym_wrappers.GymSapientino import GymSapientino
from rltg.agents.brains.TDBrain import QLearning, Sarsa

BREAKOUT = "breakout"
SAPIENTINO = "sapientino"

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


plugin_folder = os.path.join(os.path.dirname(__file__), 'commands')
class MyCLI(click.MultiCommand):
    def list_commands(self, ctx):
        rv = []
        for filename in os.listdir(plugin_folder):
            if filename.endswith('.py'):
                rv.append(filename[:-3])
        rv.sort()
        return rv

    def get_command(self, ctx, name):
        if name in ["sapientino", "minecraft"]:
            return
        ns = {}
        fn = os.path.join(plugin_folder, name + '.py')
        with open(fn) as f:
            code = compile(f.read(), fn, 'exec')
            eval(code, ns, ns)
        return ns['cli']


class Config(object):
    def __init__(self, game="", episodes=10000, algorithm=QLEARNING, gamma=0.99, alpha=0.1, epsilon=0.1,
                 nsteps=1, render=False):
        self.game = game
        self.episodes = episodes
        self.algorithm = algorithm
        self.gamma =    gamma
        self.alpha =    alpha
        self.epsilon = epsilon
        self.nsteps = nsteps
        self.render =   render

    def __str__(self):
        return "Configs:\n" + "\n".join(["\t{} = {}".format(k,v) for k, v in sorted(self.__dict__.items())])