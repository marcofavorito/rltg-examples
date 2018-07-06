from RLGames.Minecraft import LOCATIONS, TASKS, LOCATION2ENTITY, RESOURCES, TOOLS
from RLGames.gym_wrappers.GymMinecraft import GymMinecraft
from RLGames.gym_wrappers.GymPygameWrapper import PygameVideoRecorder
from flloat.base.Symbol import Symbol
from flloat.parser.ldlf import LDLfParser
from flloat.parser.ltlf import LTLfParser
from gym.spaces import Tuple, Dict

from rltg.agents.TGAgent import TGAgent
from rltg.agents.brains.TDBrain import Sarsa
from rltg.agents.feature_extraction import RobotFeatureExtractor
from rltg.agents.policies.EGreedy import EGreedy
from rltg.agents.temporal_evaluator.TemporalEvaluator import TemporalEvaluator
from rltg.trainers.GenericTrainer import GenericTrainer
from rltg.trainers.TGTrainer import TGTrainer
from rltg.utils.StoppingCondition import GoalPercentage

from utils import Config, name2algorithm


class MinecraftRobotFeatureExtractor(RobotFeatureExtractor):
    pass

class MinecraftNRobotFeatureExtractor(MinecraftRobotFeatureExtractor):

    def __init__(self, obs_space):
        # features considered by the robot in this learning task: (ball_x, ball_y, ball_dir, paddle_x)
        robot_feature_space = Tuple((
            obs_space.spaces["x"],
            obs_space.spaces["y"],
        ))

        super().__init__(obs_space, robot_feature_space)

    def _extract(self,   input, **kwargs):
        return (input["x"],
                input["y"])


class MinecraftTEFeatureExtractor(MinecraftRobotFeatureExtractor):

    def __init__(self, obs_space):
        robot_feature_space = Tuple((obs_space.spaces["location"], obs_space.spaces["actionlocation"]))
        super().__init__(obs_space, robot_feature_space)

    def _extract(self, input, **kwargs):
        return (input["location"],input["actionlocation"])

class MinecraftTemporalEvaluator(TemporalEvaluator):
    def __init__(self, input_space, formula_string, gamma=0.99, on_the_fly=False):
        # self.location_syms = [Symbol(l[0]) for l in LOCATIONS]
        # self.get, self.use = Symbol("get"), Symbol("use")
        self.locations = [l[0] for l in LOCATIONS]

        parser = LDLfParser()
        # parser = LTLfParser()
        print(formula_string)
        f = parser(formula_string)
        reward = 1

        super().__init__(MinecraftTEFeatureExtractor(input_space),
                         f.find_labels(),
                         f,
                         reward,
                         gamma=gamma,
                         on_the_fly=on_the_fly)

    def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
        res = set()

        if action == 4:
            # res.add(self.get)
            cur_action = "get"
        elif action == 5:
            # res.add(self.use)
            cur_action = "use"
        else:
            cur_action = ""

        l = features[0]
        if l < len(self.locations):
            location_sym = self.locations[l]
            # res.add(location_sym)

            s = Symbol(cur_action + "_" + location_sym)
            if s in self.alphabet.symbols: res.add(s)

        for entity, used in features[1].items():
            if used == 0: continue
            if entity in RESOURCES:
                s = Symbol("get" + "_" + entity)
            elif entity in TOOLS:
                s = Symbol("use" + "_" + entity)
            else:
                raise Exception
            if s in self.alphabet.symbols: res.add(s)

        return res

class MinecraftSafetyTemporalEvaluator(MinecraftTemporalEvaluator):
    def __init__(self, input_space, gamma=0.99, on_the_fly=False):

        # the formula
        self.location_syms = [Symbol(l[0]) for l in LOCATIONS]
        ngnu = "(<(get | use) -> (%s) >tt | [true]ff)" % " | ".join(map(str,self.location_syms))
        formula_string = "[true*]" + ngnu

        super().__init__(input_space,
                         formula_string,
                         gamma=gamma,
                         on_the_fly=on_the_fly)

class MinecraftTaskTemporalEvaluator(MinecraftTemporalEvaluator):
    """Breakout temporal evaluator for delete columns from left to right"""

    def __init__(self, input_space, task, gamma=0.99, on_the_fly=False):
        """:param task: a list of subgoals of the following form:
        ['action_location', 'action_location'...]
        e.g.
        ['get_wood', 'use_toolshed', 'get_grass', 'use_workbench']"""


        # the formula
        # ngnu = "true*"
        # split_task = lambda x: " & ".join(x.split("_"))
        # formula_string = "<true*>(<%s;" % ngnu + (";" + ngnu + ";").join(map(split_task, task)) + ">tt)"

        # formula_string = ""
        # for i, t in enumerate(task):
        #     formula_string += "F(" + t + (" & " if i!=len(task)-1 else "")
        # formula_string += ")" * len(task)

        # the formula
        ngnu = "true*"
        formula_string = "<true*>(<true*;"
        prop = []
        for i, t in enumerate(task):
            prop.append(t)
            k = "(" + " & ".join(task[:i+1]) + " & " + " & ".join(map(lambda x: "!"+x, task[i+1:])) + ")"
            # formula_string +=  k + (";" + k + "*;" if i != len(task)-1 else "")
            formula_string +=  k + (";" + k + "*;") if i != len(task)-1 else prop[-1]

        formula_string += ">tt)"

        super().__init__(input_space,
                         formula_string,
                         gamma=gamma,
                         on_the_fly=on_the_fly)


    def fromFeaturesToPropositional(self, features, action, *args, **kwargs):
        return super().fromFeaturesToPropositional(features, action, *args, **kwargs)

def temporal_evaluators_from_task(env, tasks, gamma=0.99, on_the_fly=False):
    # res = [MinecraftTaskTemporalEvaluator(env.observation_space, t, gamma=gamma, on_the_fly=on_the_fly) for k, t in list(tasks.items())[:4] if "make" in k]
    # res.append((MinecraftSafetyTemporalEvaluator(env.observation_space, gamma=gamma, on_the_fly=on_the_fly)))
    # for a in res:
    #     a._automaton.to_dot(str(a.formula))
    # res = [MinecraftTaskTemporalEvaluator(env.observation_space, t, gamma=gamma, on_the_fly=on_the_fly)]

    res = [MinecraftTaskTemporalEvaluator(env.observation_space, t, gamma=gamma, on_the_fly=on_the_fly) for k, t in list(tasks.items())]
    return res


minecraft_name2robot_feature_ext = {
    "N": MinecraftNRobotFeatureExtractor,
    # "D": MinecraftDRobotFeatureExtractor
}

minecraft_name2temp_goals = {
    "all": [],
    # "rows": [BreakoutCompleteRowsTemporalEvaluator],
    # "both": [BreakoutCompleteColumnsTemporalEvaluator, BreakoutCompleteRowsTemporalEvaluator]
}


def _set_up_temporal_sapientino(config, args, env, robot_feature_extractor, brain):
    temporal_goals = []
    if args.temp_goal == "all":
        temporal_goals = temporal_evaluators_from_task(env, TASKS, gamma=args.gamma, on_the_fly=args.on_the_fly)
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
    env = GymMinecraft()

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
        robot_feature_extractor = minecraft_name2robot_feature_ext[args.robot_feature_space](env.observation_space)
        brain = name2algorithm[config.algorithm](None, env.action_space, policy=EGreedy(config.epsilon),
                                         alpha=config.alpha, gamma=config.gamma, lambda_=config.lambda_)

        agent, trainer = _set_up_temporal_sapientino(config, args, env, robot_feature_extractor, brain)

        stats, optimal_stats = trainer.main(render=render, verbosity=args.verbosity)

    return stats, optimal_stats