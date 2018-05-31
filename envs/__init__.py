from envs import breakout, sapientino, minecraft
from utils import Config, BREAKOUT, SAPIENTINO, MINECRAFT

name2module = {
    BREAKOUT:   breakout,
    SAPIENTINO: sapientino,
    MINECRAFT:  minecraft
}

# dispatcher
# should be called from 'play'
def run_experiment(config:Config, args):
    cmd = args.cmd
    name2module[cmd].run_experiment(config, args)
