import argparse

from rltg.trainers.Trainer import Trainer

parser = argparse.ArgumentParser(description='Resume a Reinforcement Learning process')
parser.add_argument('--render',     action='store_true',                   help='Enable rendering.')
parser.add_argument('--verbosity',  default=1, type=int, choices=[0,1,2],  help='Verbosity {0,1,2} Default: 1')
parser.add_argument('--datadir',    default="data",             help='Directory where to retrieve data from.')


args = parser.parse_args()

if __name__ == '__main__':
    trainer = Trainer.resume(datadir=args.datadir, render=args.render, verbosity=args.verbosity)