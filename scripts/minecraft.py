import os
import sys
from collections import OrderedDict

NUM_EXPERIMENTS_PER_CONFIG = 10


conf = OrderedDict([
    ("episodes", [10000]),
    ("reward_shaping",[False, True]),
    ("on_the_fly", [False, True]),

    # ("temp_goal", ["cols --left_right"])
    # ("temp_goal", ["cols", "rows", "both"])
    ("temp_goal", ["all"])
])

def _make_cmd(ep, rs, otf, goal):
    main_goal_string = goal
    s = "data/minecraft" + "__".join([a + "_" + b for a, b in zip(keys, map(str, [ep, rs, otf, main_goal_string]))])
    experiment_string = "--episodes {} --verbosity 0 --epsilon 0.25 --gamma 0.99 --lambda 0.3 {} {} --datadir {} minecraft --temp_goal {}" \
        .format(ep, "--reward_shaping" if rs else "", "--on_the_fly" if otf else "", s, goal)
    return experiment_string

def run_train(cmd):
    train.main(cmd)

if __name__ == '__main__':
    # EXECUTE THIS SCRIPT FROM THE ROOT DIRECTORY
    sys.path.append(os.curdir)

    import train
    processes = []

    import itertools
    keys, ranges = zip(*conf.items())
    configurations = itertools.product(*ranges)

    sh = "{\n"
    for ep, rs, otf, goal in list(configurations):
        if otf and not rs: continue

        cmd = _make_cmd(ep, rs, otf, goal=goal)
        print(cmd)

        # run_train(cmd)
        # os.system("python3 train.py " + cmd + " &;")
        sh +=" python3 train.py " + cmd + "\n"


    sh += "} &"
    for i in range(NUM_EXPERIMENTS_PER_CONFIG):
        print(sh)
        os.system(sh)
    os.system("wait")


