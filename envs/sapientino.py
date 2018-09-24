from RLGames.Sapientino import COLORS
from RLGames.gym_wrappers.GymPygameWrapper import PygameVideoRecorder
from RLGames.gym_wrappers.GymSapientino import GymSapientino
from flloat.base.Symbol import Symbol
from flloat.parser.ldlf import LDLfParser
from gym.spaces import Tuple
from rltg.agents.RLAgent import RLAgent
from rltg.agents.TGAgent import TGAgent

from rltg.agents.feature_extraction import RobotFeatureExtractor
from rltg.agents.policies.EGreedy import EGreedy
from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator
from rltg.trainers.GenericTrainer import GenericTrainer
from rltg.trainers.TGTrainer import TGTrainer
from rltg.utils.StoppingCondition import GoalPercentage

from utils import name2algorithm, Config


class SapientinoRobotFeatureExtractor(RobotFeatureExtractor):
    pass


class SapientinoNRobotFeatureExtractor(SapientinoRobotFeatureExtractor):

    def __init__(self, obs_space):
        robot_feature_space = Tuple((
            obs_space.spaces["x"],
            obs_space.spaces["y"],
        ))

        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return (input["x"],
                input["y"],)

class SapientinoDRobotFeatureExtractor(SapientinoRobotFeatureExtractor):

    def __init__(self, obs_space):
        robot_feature_space = Tuple((
            obs_space.spaces["x"],
            obs_space.spaces["y"],
            obs_space.spaces["theta"],
        ))

        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return (input["x"],
                input["y"],
                input["theta"])


class SapientinoTEFeatureExtractor(SapientinoRobotFeatureExtractor):

    def __init__(self, obs_space):
        robot_feature_space = Tuple((
            obs_space.spaces["color"],
        ))

        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return (input["color"],)



class SapientinoTemporalEvaluator(TemporalEvaluator):
    """Sapientino temporal evaluator for visit colors in a given order.
    If relaxed is False, stop if an illegal bip is done."""

    def __init__(self, input_space, gamma=0.99, on_the_fly=False, relaxed=True):
        self.color_syms = [Symbol(c) for c in COLORS] + [Symbol("no_color")]
        self.bip = Symbol("bip")

        parser = LDLfParser()

        if not relaxed:
            # the formula
            sb = str(self.bip)
            not_bip = ";(!%s)*;"%sb
            and_bip = lambda x: str(x) + " & " + sb
            # every color-bip in sequence, no bip between colors.
            formula_string = "<(!%s)*;"%sb + not_bip.join(map(and_bip, self.color_syms[:-1])) + ">tt"
        else:
            sb = str(self.bip)
            not_bip = ";true*;"
            and_bip = lambda x: str(x) + " & " + sb
            # every color-bip in sequence, no bip between colors.
            formula_string = "<true*;" + not_bip.join(map(and_bip, self.color_syms[:-1])) + ">tt"

        print(formula_string)
        f = parser(formula_string)

        reward = 1

        super().__init__(SapientinoTEFeatureExtractor(input_space),
                         set(self.color_syms).union({self.bip}),
                         f,
                         reward,
                         gamma=gamma,
                         on_the_fly=on_the_fly)


    def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
        res = set()
        # bip action
        if action == 4:
            res.add(self.bip)

        c = features[0]
        color_sym = self.color_syms[c]
        res.add(color_sym)

        return res


sapientino_name2robot_feature_ext = {
    "N": SapientinoNRobotFeatureExtractor,
    "D": SapientinoDRobotFeatureExtractor
}

sapientino_name2temp_goals = {
    "colors": [SapientinoTemporalEvaluator],
    "colors_relaxed": [SapientinoTemporalEvaluator],
}


def _set_up_temporal_sapientino(config, args, env, robot_feature_extractor, brain):
    temporal_goals = []

    relaxed = False if args.temp_goal == "colors" else True if args.temp_goal == "colors_relaxed" else None
    temporal_goals.append(SapientinoTemporalEvaluator(env.observation_space,
                                                           gamma=config.gamma, on_the_fly=config.on_the_fly,
                                                           relaxed=relaxed))
    agent = TGAgent(robot_feature_extractor,
                    brain,
                    temporal_goals,
                    reward_shaping=config.reward_shaping)

    tr = TGTrainer(env, agent, n_episodes=config.episodes,
                   stop_conditions=(GoalPercentage(100, 1.0),),
                   data_dir=config.datadir
                   )
    return agent, tr


def run_experiment(config:Config, args):
    env = GymSapientino()

    render = config.render
    if render:
        env = PygameVideoRecorder(env, config.datadir+"/videos")

    if config.resume:
        trainer = GenericTrainer if args.temp_goal is None else TGTrainer
        stats, optimal_stats = trainer.resume(render=render, verbosity=args.verbosity)
    elif config.eval:
        trainer = GenericTrainer if args.temp_goal is None else TGTrainer
        stats, optimal_stats = trainer.eval(render=render, verbosity=args.verbosity)
    else:
        robot_feature_extractor = sapientino_name2robot_feature_ext[args.robot_feature_space](env.observation_space)
        brain = name2algorithm[config.algorithm](None, env.action_space, policy=EGreedy(config.epsilon),
                                         alpha=config.alpha, gamma=config.gamma, lambda_=config.lambda_)

        agent, trainer = _set_up_temporal_sapientino(config, args, env, robot_feature_extractor, brain)

        print(agent.temporal_evaluators[0].formula)
        stats, optimal_stats = trainer.main(render=render, verbosity=args.verbosity)

    return stats, optimal_stats