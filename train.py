import argparse
import shlex
import sys

from envs import minecraft, sapientino, breakout
from envs.breakout import breakout_name2robot_feature_ext, breakout_name2temp_goals
from envs.minecraft import minecraft_name2temp_goals, minecraft_name2robot_feature_ext
from envs.sapientino import sapientino_name2temp_goals, sapientino_name2robot_feature_ext
from utils import name2algorithm, check_in_float_range, BREAKOUT, SAPIENTINO, MINECRAFT, Config, SARSA

parser = argparse.ArgumentParser(description='Execute a Reinforcement Learning process')
parser.add_argument('--algorithm',      default=SARSA, choices=list(name2algorithm.keys()),              help="TD algorithm variant. The possible values are: '" + "', '".join(list(name2algorithm.keys())) + "'")
parser.add_argument('--episodes',       default=10000, type=int,                                         help='Number of episodes.')
parser.add_argument('--gamma',          default=1.0,   type=check_in_float_range(0.0, 1.0, True,  True), help="The discount factor. Must fall into [0, 1].")
parser.add_argument('--alpha',          default=0.1,   type=check_in_float_range(0.0, 1.0, False, True), help="The learning rate. Must fall into (0, 1], or None (adaptive). Default: 0.1")
parser.add_argument('--epsilon',        default=0.1,   type=check_in_float_range(0.0, 1.0, True,  True), help="The epsilon in eps-greedy. Must fall into [0, 1].")
parser.add_argument('--lambda',         default=0.0,   type=check_in_float_range(0.0, 1.0, True,  True), help="Lambda in TD(Lambda). Default: 0 (classical TD)", dest="lambda_")
parser.add_argument('--reward_shaping', action='store_true',        help="Enable reward shaping")
parser.add_argument('--on_the_fly',     action='store_true',        help="Enable on-the-fly construction")
parser.add_argument('--render',         action='store_true',        help='Enable rendering.')
parser.add_argument('--datadir',        default="data",             help='Directory to store the output of the process.')
parser.add_argument('--verbosity',      default=1,      type=int, choices=[0,1,2],  help='Verbosity {0,1,2}')

# Environment subparser
env_subparser = parser.add_subparsers(title="Environment selection", metavar="ENVIRONMENT", description="choose the environment", dest='cmd')
env_subparser.required = True

# Breakout
breakout_parser = env_subparser.add_parser(BREAKOUT, help="use the Breakout environment")
breakout_parser.add_argument("--robot_feature_space", default="N", choices=sorted(breakout_name2robot_feature_ext.keys()), help="Specify the feature space for the robot. N=normal, S=reduced")
breakout_parser.add_argument("--brick_rows", type=int, default=3, help="The number of brick rows.")
breakout_parser.add_argument("--brick_cols", type=int, default=3, help="The number of brick columns.")
breakout_parser.add_argument("--temp_goal", choices=sorted(breakout_name2temp_goals.keys()), help="Temporal goal. Remove bricks by columns, by rows, or both.")
breakout_parser.add_argument("--left_right", default=False, action='store_true',        help="From the left column to the right one. If not specified, the order is inverted.")
breakout_parser.add_argument("--bottom_up",  default=False, action='store_true',        help="From the bottom row to the top one. If not specified, the order is inverted.")

# Sapientino
sapientino_subparser = env_subparser.add_parser(SAPIENTINO, help="use the Sapientino environment")
sapientino_subparser.add_argument("--robot_feature_space", default="N", choices=sorted(sapientino_name2robot_feature_ext.keys()), help="Specify the feature space for the robot. N=normal, D=differential")
sapientino_subparser.add_argument("--temp_goal", choices=sorted(sapientino_name2temp_goals.keys()), help="Temporal goal. Ordered visit of colors.")


# Minecraft
minecraft_subparser = env_subparser.add_parser(MINECRAFT, help="use the Minecraft environment")
minecraft_subparser.add_argument("--robot_feature_space", default="N", choices=sorted(minecraft_name2robot_feature_ext.keys()), help="Specify the feature space for the robot. N=normal, D=differential")
minecraft_subparser.add_argument("--temp_goal", choices=sorted(minecraft_name2temp_goals.keys()), help="Temporal goal. All the tasks.")


name2module = {
    BREAKOUT: breakout,
    SAPIENTINO: sapientino,
    MINECRAFT: minecraft
}

def print_info(config, args):
    with open("experiment.info", "w") as f:
        f.write(str(config))
        f.write("\n")
        f.write(str(args))


def main(cli_input):
    if type(cli_input)==str:
        arguments = shlex.split(cli_input)
    elif type(cli_input) == list:
        arguments = cli_input
    else:
        raise Exception

    args = parser.parse_args(arguments)

    config = Config(episodes=args.episodes,
                    algorithm=args.algorithm,
                    gamma=args.gamma,
                    alpha=args.alpha,
                    epsilon=args.epsilon,
                    lambda_=args.lambda_,
                    reward_shaping=args.reward_shaping,
                    on_the_fly=args.on_the_fly,
                    datadir=args.datadir,
                    render=args.render,
                    verbosity=args.verbosity)

    print(config)
    # dispatching
    cmd = args.cmd
    name2module[cmd].run_experiment(config, args)


if __name__ == '__main__':
    main(sys.argv[1:])