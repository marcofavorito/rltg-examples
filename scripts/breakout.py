import os
import sys
from collections import OrderedDict
# from multiprocessing import Process
# from multiprocessing.dummy import Pool as ThreadPool
# from multiprocessing.pool import ThreadPool

from multiprocessing import Pool
# from src import train

NUM_EXPERIMENTS_PER_CONFIG = 10


# pool = ThreadPool(NUM_EXPERIMENTS_PER_CONFIG)
# pool = Pool(processes=10)

conf = OrderedDict([
    ("episodes", [5000]),
    ("brick_cols", [3]),
    ("brick_rows", [3]),
    ("reward_shaping",[True, False]),
    ("on_the_fly", [False, True]),
])

def _make_cmd(ep, rows, cols, rs, otf):
    s = "data/" + "__".join([a + "_" + b for a, b in zip(keys, map(str, [ep, rows, cols, rs, otf]))])
    experiment_string = "--episodes {} --gamma 0.999 --lambda 0.99 --verbosity 0 {} {} --datadir {} breakout --brick_cols {} --brick_rows {} --temp_goal cols --left_right" \
        .format(ep, "--reward_shaping" if rs else "", "--on_the_fly" if otf else "", s, cols, rows)
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
    for ep, rows, cols, rs, otf in list(configurations):
        if otf and not rs: continue

        cmd = _make_cmd(ep, rows, cols, rs, otf)
        print(cmd)

        # run_train(cmd)
        # os.system("python3 train.py " + cmd + " &;")
        sh +=" python3 train.py " + cmd + "\n"


    sh += "} &"
    for i in range(NUM_EXPERIMENTS_PER_CONFIG):
        print(sh)
        os.system(sh)
    os.system("wait")



        # pool.map(run_train, [cmd]*NUM_EXPERIMENTS_PER_CONFIG)
        # pool.map(lambda x: train.main(x), [cmd] * NUM_EXPERIMENTS_PER_CONFIG)

        # pool.map(run_train, (cmd, )*10)
        # run_train(cmd)
        # ps = [Process(target=run_train, args=(cmd,) ) for i in range(NUM_EXPERIMENTS_PER_CONFIG)]
        # for p in ps:
        #     p.start()
        #
        # for p in ps:
        #     p.join()
