import argparse

from envs import breakout, sapientino, minecraft
from envs.breakout import name2robot_feature_ext, name2temp_goals
from utils import name2algorithm, check_in_float_range, BREAKOUT, SAPIENTINO, MINECRAFT, Config, SARSA

parser = argparse.ArgumentParser(description='Execute a Reinforcement Learning process')
parser.add_argument('--algorithm',      default=SARSA, choices=list(name2algorithm.keys()),              help="TD algorithm variant. The possible values are: '" + "', '".join(list(name2algorithm.keys())) + "'")
parser.add_argument('--episodes',       default=10000,                                                   help='Number of episodes.')
parser.add_argument('--gamma',          default=1.0,   type=check_in_float_range(0.0, 1.0, True,  True), help="The discount factor. Must fall into [0, 1].")
parser.add_argument('--alpha',          default=0.1,   type=check_in_float_range(0.0, 1.0, False, True), help="The learning rate. Must fall into (0, 1], or None (adaptive). Default: 0.1")
parser.add_argument('--epsilon',        default=0.1,   type=check_in_float_range(0.0, 1.0, True,  True), help="The epsilon in eps-greedy. Must fall into [0, 1].")
parser.add_argument('--lambda',         default=0.0,   type=check_in_float_range(0.0, 1.0, True,  True), help="Lambda in TD(Lambda). Default: 0 (classical TD)", dest="lambda_")
parser.add_argument('--reward_shaping', action='store_true',    help="Enable reward shaping")
parser.add_argument('--on_the_fly',     action='store_true',    help="Enable on-the-fly construction")
parser.add_argument('--render',         action='store_true',    help='Enable rendering.')
parser.add_argument('--resume',         action='store_true',    help='Resume the learning process.')
parser.add_argument('--eval',           action='store_true',    help='Enable evaluation mode.')
parser.add_argument('--datadir',        default="data",         help='Directory to store the output of the process.')


# Environment subparsers
env_subparser = parser.add_subparsers(title="Environment selection", metavar="ENVIRONMENT", description="choose the environment", dest='cmd')
env_subparser.required = True

# Breakout
breakout_parser = env_subparser.add_parser(BREAKOUT, help="use the Breakout environment")
breakout_parser.add_argument("--robot-feature-space", default="N", choices=sorted(name2robot_feature_ext.keys()), help="Specify the feature space for the robot. N=normal, S=reduced")
breakout_parser.add_argument("--brick_rows", default=3, help="The number of brick rows.")
breakout_parser.add_argument("--brick_cols", default=3, help="The number of brick columns.")
breakout_parser.add_argument("--temp_goal",  default="cols", choices=sorted(name2temp_goals.keys()), help="Temporal goal.")

# Sapientino
sapientino_subparser = env_subparser.add_parser(SAPIENTINO, help="use the Sapientino environment")

# Minecraft
minecraft_subparser = env_subparser.add_parser(MINECRAFT, help="use the Minecraft environment")

args = parser.parse_args()

name2module = {
    BREAKOUT:   breakout,
    SAPIENTINO: sapientino,
    MINECRAFT:  minecraft
}


if __name__ == '__main__':
    config = Config(episodes=       args.episodes,
                    algorithm=      args.algorithm,
                    gamma=          args.gamma,
                    alpha=          args.alpha,
                    epsilon=        args.epsilon,
                    lambda_=        args.lambda_,
                    reward_shaping= args.reward_shaping,
                    on_the_fly=     args.on_the_fly,
                    resume=         args.resume,
                    eval=           args.eval,
                    datadir=        args.datadir,
                    render=         args.render)

    print("Configurations:", config)
    # dispatching
    cmd = args.cmd
    name2module[cmd].run_experiment(config, args)


