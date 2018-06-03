"""Summary:

#################################################################################################
FEATURE EXTRACTORS
#################################################################################################


Two FeatureExtractor:
 - BreakoutRobotFeatureExtractor:           Used by the robot for the main task.
                                            returns (ball_x, ball_y, ball_dir, paddle_x)

 - BreakoutGoalFeatureExtractor:            Used by the temporal goal evaluators.
                                            returns a matrix of booleans: columns * rows
                                            representing the state of each brick

#################################################################################################
TEMPORAL EVALUATORS
#################################################################################################

- BreakoutCompleteLinesTemporalEvaluator:   Abstract class for temporal goals which involve the ordered deletion of
                                            lines of bricks, rows or columns.
                                            Uses BreakoutGoalFeatureExtractor

- BreakoutCompleteRowsTemporalEvaluator:    extends from BreakoutCompleteLinesTemporalEvaluator
                                            deletion of rows bottom-up or top-down

- BreakoutCompleteColumnsTemporalEvaluator: extends from BreakoutCompleteLinesTemporalEvaluator
                                            deletion of columns from left-to-right or right-to-left

#################################################################################################
"""
from abc import abstractmethod

import numpy as np
from RLGames.gym_wrappers.GymBreakout import GymBreakout
from flloat.base.Symbol import Symbol
from flloat.parser.ldlf import LDLfParser
from gym.spaces import Box, Tuple
from rltg.agents.RLAgent import RLAgent
from rltg.agents.TGAgent import TGAgent
from rltg.agents.feature_extraction import FeatureExtractor, RobotFeatureExtractor
from rltg.agents.policies.EGreedy import EGreedy
from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator
from rltg.trainers.GenericTrainer import GenericTrainer
from rltg.trainers.TGTrainer import TGTrainer
from rltg.utils.StoppingCondition import GoalPercentage

from utils import Config, name2algorithm


class BreakoutRobotFeatureExtractor(RobotFeatureExtractor):
    pass

class BreakoutNRobotFeatureExtractor(BreakoutRobotFeatureExtractor):

    def __init__(self, obs_space):
        # features considered by the robot in this learning task: (ball_x, ball_y, ball_dir, paddle_x)
        robot_feature_space = Tuple((
            obs_space.spaces["ball_x"],
            obs_space.spaces["ball_y"],
            obs_space.spaces["ball_dir"],
            obs_space.spaces["paddle_x"],
        ))
        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return (input["ball_x"],
                input["ball_y"],
                input["ball_dir"],
                input["paddle_x"])


class BreakoutSRobotFeatureExtractor(BreakoutRobotFeatureExtractor):

    def __init__(self, obs_space):
        robot_feature_space = Tuple((
            obs_space.spaces["diff_paddle_ball"],
        ))
        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return input["diff_paddle_ball"],


class BreakoutGoalFeatureExtractor(FeatureExtractor):
    def __init__(self, obs_space, bricks_rows=3, bricks_cols=3):
        output_space = Box(low=0, high=1, shape=(bricks_cols, bricks_rows), dtype=np.uint8)
        super().__init__(obs_space, output_space)

    def _extract(self, input, **kwargs):
        return input["bricks_matrix"]


def get_breakout_lines_formula(lines_symbols):
    # Generate the formula string
    # E.g. for 3 line symbols:
    # "<(!l0 & !l1 & !l2)*;(l0 & !l1 & !l2);(l0 & !l1 & !l2)*;(l0 & l1 & !l2); (l0 & l1 & !l2)*; l0 & l1 & l2>tt"
    pos = list(map(str, lines_symbols))
    neg = list(map(lambda x: "!" + str(x), lines_symbols))

    s = "(%s)*" % " & ".join(neg)
    for idx in range(len(lines_symbols)-1):
        step = " & ".join(pos[:idx + 1]) + " & " + " & ".join(neg[idx + 1:])
        s += ";({0});({0})*".format(step)
    s += ";(%s)" % " & ".join(pos)
    s = "<%s>tt" % s

    return s


class BreakoutCompleteLinesTemporalEvaluator(TemporalEvaluator):
    """Breakout temporal evaluator for delete columns from left to right"""

    def __init__(self, input_space, bricks_cols=3, bricks_rows=3, lines_num=3, gamma=0.99, on_the_fly=False):
        assert lines_num == bricks_cols or lines_num == bricks_rows
        self.line_symbols = [Symbol("l%s" % i) for i in range(lines_num)]
        lines = self.line_symbols

        parser = LDLfParser()


        string_formula = get_breakout_lines_formula(lines)
        print(string_formula)
        f = parser(string_formula)
        reward = 10000

        super().__init__(BreakoutGoalFeatureExtractor(input_space, bricks_cols=bricks_cols, bricks_rows=bricks_rows),
                         set(lines),
                         f,
                         reward,
                         gamma=gamma,
                         on_the_fly=on_the_fly)

    @abstractmethod
    def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
        """map the matrix bricks status to a propositional formula
        first dimension: columns
        second dimension: row
        """
        matrix = features
        lines_status = np.all(matrix == 0.0, axis=kwargs["axis"])
        result = set()
        sorted_symbols = reversed(self.line_symbols) if kwargs["is_reversed"] else self.line_symbols
        for rs, sym in zip(lines_status, sorted_symbols):
            if rs:
                result.add(sym)

        return frozenset(result)


class BreakoutCompleteRowsTemporalEvaluator(BreakoutCompleteLinesTemporalEvaluator):
    """Temporal evaluator for complete rows in order"""

    def __init__(self, input_space, bricks_cols=3, bricks_rows=3, bottom_up=True, gamma=0.99, on_the_fly=False):
        super().__init__(input_space, bricks_cols=bricks_cols, bricks_rows=bricks_rows, lines_num=bricks_rows, gamma=gamma, on_the_fly=on_the_fly)
        self.bottom_up = bottom_up

    def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
        """complete rows from bottom-to-up or top-to-down, depending on self.bottom_up"""
        return super().fromFeaturesToPropositional(features, action, axis=0, is_reversed=self.bottom_up)


class BreakoutCompleteColumnsTemporalEvaluator(BreakoutCompleteLinesTemporalEvaluator):
    """Temporal evaluator for complete columns in order"""

    def __init__(self, input_space, bricks_cols=3, bricks_rows=3, left_right=True, gamma=0.99, on_the_fly=False):
        super().__init__(input_space, bricks_cols=bricks_cols, bricks_rows=bricks_rows, lines_num=bricks_cols, gamma=gamma, on_the_fly=on_the_fly)
        self.left_right = left_right

    def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
        """complete columns from left-to-right or right-to-left, depending on self.left_right"""
        return super().fromFeaturesToPropositional(features, action, axis=1, is_reversed=not self.left_right)


name2robot_feature_ext = {
    "N": BreakoutNRobotFeatureExtractor,
    "S": BreakoutSRobotFeatureExtractor
}

name2temp_goals = {
    "cols": [BreakoutCompleteColumnsTemporalEvaluator],
    "rows": [BreakoutCompleteRowsTemporalEvaluator],
    "both": [BreakoutCompleteColumnsTemporalEvaluator, BreakoutCompleteRowsTemporalEvaluator]
}


def _set_up_temporal_breakout(config, args, env, robot_feature_extractor, brain):
    temporal_goals = []
    if args.temp_goal == "cols" or args.temp_goal == "both":
        by_cols = BreakoutCompleteColumnsTemporalEvaluator(env.observation_space, bricks_rows=args.brick_rows,
                                                           bricks_cols=args.brick_cols, left_right=args.left_right,
                                                           gamma=config.gamma, on_the_fly=config.on_the_fly)
        temporal_goals.append(by_cols)

    if args.temp_goal == "rows" or args.temp_goal == "both":
        by_rows = BreakoutCompleteRowsTemporalEvaluator(env.observation_space, bricks_rows=args.brick_rows,
                                                        bricks_cols=args.brick_cols, bottom_up=args.bottom_up,
                                                        gamma=config.gamma, on_the_fly=config.on_the_fly)
        temporal_goals.append(by_rows)

    agent = TGAgent(robot_feature_extractor,
                    brain,
                    temporal_goals,
                    reward_shaping=config.reward_shaping)

    tr = TGTrainer(env, agent, n_episodes=config.episodes,
                   stop_conditions=(GoalPercentage(100, 1.0),),
                   data_dir=config.datadir
                   )
    return agent, tr


def _set_up_simple_breakout(config, args, env, robot_feature_extractor, brain):
    agent = RLAgent(robot_feature_extractor, brain)
    tr = GenericTrainer(env, agent, n_episodes=config.episodes,
                        stop_conditions=(GoalPercentage(100, 1.0),),
                        data_dir=config.datadir)
    return agent, tr


def run_experiment(config:Config, args):
    env = GymBreakout(brick_cols=args.brick_cols, brick_rows=args.brick_rows)

    render = config.render

    if config.resume:
        trainer = GenericTrainer if args.temp_goal is None else TGTrainer
        stats, optimal_stats = trainer.resume(render=render, verbosity=args.verbosity)
    elif config.eval:
        trainer = GenericTrainer if args.temp_goal is None else TGTrainer
        stats, optimal_stats = trainer.eval(render=render, verbosity=args.verbosity)
    else:
        robot_feature_extractor = name2robot_feature_ext[args.robot_feature_space](env.observation_space)
        brain = name2algorithm[config.algorithm](None, env.action_space, policy=EGreedy(config.epsilon),
                                         alpha=config.alpha, gamma=config.gamma, lambda_=config.lambda_)


        if args.temp_goal is None:
            print("No temporal goal - simple Breakout")
            agent, trainer = _set_up_simple_breakout(config, args, env, robot_feature_extractor, brain)
        else:
            agent, trainer = _set_up_temporal_breakout(config, args, env, robot_feature_extractor, brain)

        stats, optimal_stats = trainer.main(render=render, verbosity=args.verbosity)

    return stats, optimal_stats