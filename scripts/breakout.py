import os
import sys
from collections import OrderedDict

NUM_EXPERIMENTS_PER_CONFIG = 10


conf = OrderedDict([
    ("episodes", [15000]),
    ("brick_rows", [3]),
    ("brick_cols", [3]),
    ("reward_shaping",[True]),
    ("on_the_fly", [False]),
    # ("temp_goal", ["cols --left_right"])
    # ("temp_goal", ["cols", "rows", "both"])
    ("temp_goal", ["both", "both --bottom_up", "both --left_right", "both --left_right --bottom_up",
                   "cols", "cols --left_right", "rows", "rows --bottom_up"])
])

def _make_cmd(ep, rows, cols, rs, otf, goal="cols"):
    goal_string_command = goal.split()
    main_goal_string = goal_string_command[0]
    orderings = "-".join(map(lambda x: x[2:], sorted(goal_string_command[1:])))
    s = "data/breakout" + "__".join([a + "_" + b for a, b in zip(keys, map(str, [ep, rows, cols, rs, otf, main_goal_string + "-" + orderings]))])
    experiment_string = "--episodes {} --gamma 0.999 --lambda 0.99 --verbosity 0 {} {} --datadir {} breakout --brick_cols {} --brick_rows {} --temp_goal {}" \
        .format(ep, "--reward_shaping" if rs else "", "--on_the_fly" if otf else "", s, cols, rows, goal)
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
    for ep, rows, cols, rs, otf, goal in list(configurations):
        if otf and not rs: continue
        if rows > cols: continue

        cmd = _make_cmd(ep, rows, cols, rs, otf, goal=goal)
        print(cmd)

        # run_train(cmd)
        # os.system("python3 train.py " + cmd + " &;")
        sh +=" python3 train.py " + cmd + "\n"


    sh += "} &"
    for i in range(NUM_EXPERIMENTS_PER_CONFIG):
        print(sh)
        os.system(sh)
    os.system("wait")


