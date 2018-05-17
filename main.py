import click
from click import pass_context

from utils import MyCLI, Config, name2env, BREAKOUT, name2algorithm, QLEARNING


@click.command(cls=MyCLI)
@click.option('--game', default=BREAKOUT,
              type=click.Choice(sorted(name2env.keys())),       help="The game for run the experiments")
@click.option('--episodes',  default=10000,                     help='Number of episodes.')
@click.option('--algorithm', default=QLEARNING,
              type=click.Choice(sorted(name2algorithm.keys())), help="The RL algorihtm.")
@click.option('--gamma',     default=1.0,   type=click.FLOAT,   help="The discount factor. Must fall into (0, 1].")
@click.option('--alpha',     default=None,  type=click.FLOAT,   help="The learning rate. Must fall into (0, 1].")
@click.option('--epsilon',   default=0.1,   type=click.FLOAT,   help="The epsilon in eps-greedy. Must fall into [0, 1].")
@click.option('--nsteps',    default=1,     type=click.INT,     help="The number of step for the update rule.")
@click.option('--render',    is_flag=True,                      help='Enable rendering.')
@pass_context
def main(ctx, game, episodes, algorithm, gamma, alpha, epsilon, nsteps, render):
    ctx.obj = \
        Config(game, episodes, algorithm, gamma, alpha, epsilon, nsteps, render)
    print(str(ctx.obj))
    print()


if __name__ == '__main__':
    main()
