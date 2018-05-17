import click
from click import pass_context
from rltg.agents.TGAgent import TGAgent
from rltg.agents.exploration_policies.RandomPolicy import RandomPolicy
from rltg.trainer import Trainer
from rltg.utils.Renderer import PygameRenderer

from games.breakout import BreakoutNRobotFeatureExtractor, BreakoutSRobotFeatureExtractor, \
    BreakoutCompleteColumnsTemporalEvaluator, BreakoutCompleteRowsTemporalEvaluator
from utils import name2env, name2algorithm

name2robot_feature_ext = {
    "N": BreakoutNRobotFeatureExtractor,
    "S": BreakoutSRobotFeatureExtractor
}

name2temp_goals = {
    "cols": [BreakoutCompleteColumnsTemporalEvaluator],
    "rows": [BreakoutCompleteRowsTemporalEvaluator],
    "both": [BreakoutCompleteColumnsTemporalEvaluator, BreakoutCompleteRowsTemporalEvaluator]
}

@click.command()
@click.option("--robot-feature-space", default="N", type=click.Choice(sorted(name2robot_feature_ext.keys())),
              help="Specify the feature space for the robot. N=normal, S=reduced")
@click.option("--rows", default=3, help="The number of brick rows.")
@click.option("--cols", default=3, help="The number of brick columns.")
@click.option("--temp_goal", default="cols", type=click.Choice(["cols", "rows", "both"]), help="Temporal goal.")
@pass_context
def cli(ctx, robot_feature_space, rows, cols, temp_goal):
    config = ctx.obj
    env = name2env[config.game](brick_rows=rows, brick_cols=cols)

    robot_feat_ext = name2robot_feature_ext[robot_feature_space](env.observation_space)
    temp_goals = [class_temp_goal(env.observation_space, bricks_rows=env.brick_rows, bricks_cols=env.brick_cols,
                                  gamma=config.gamma) for class_temp_goal in name2temp_goals[temp_goal]]

    exploration_policy = RandomPolicy(env.action_space, epsilon=config.epsilon)
    brain = name2algorithm[config.algorithm](None, env.action_space, alpha=config.alpha,
                                             gamma=config.gamma, nsteps=config.nsteps)

    agent = TGAgent(robot_feat_ext, exploration_policy, brain, temp_goals)

    t = Trainer(env, agent,
                n_episodes=config.episodes,
                resume=False,
                eval=False,
                # resume = True,
                # eval = True,
                renderer=PygameRenderer(delay=0.01) if config.render else None
                )
    t.main()

