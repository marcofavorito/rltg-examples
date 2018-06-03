import argparse

from rltg.trainers.Trainer import Trainer

parser = argparse.ArgumentParser(description='Execute a Reinforcement Learning process')
parser.add_argument('--render',     action='store_true',                   help='Enable rendering.')
parser.add_argument('--verbosity',  default=1, type=int, choices=[0,1,2],  help='Verbosity {0,1,2} Default: 1')
parser.add_argument('--datadir',    default="data",             help='Directory to store the output of the process.')


args = parser.parse_args()

if __name__ == '__main__':
    trainer = Trainer.eval(datadir=args.datadir, render=args.render, verbosity=args.verbosity)