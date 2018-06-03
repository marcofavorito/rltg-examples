
import csv
import os
import sys

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

colors = iter(["b", "g", "r"])

def _align_to_same_length(sequences_per_experiments):

    result = []

    maxes = []
    for sequences in sequences_per_experiments:
        cur_max = max(list(map(lambda x: len(x), sequences)))
        maxes.append((cur_max))
    max_ = max(maxes)

    for sequences in sequences_per_experiments:
        new_sequences = []
        for rs in sequences:
            rs += (max_ - len(rs)) * [rs[-1]]
            new_sequences.append(rs)
        result.append(new_sequences)

    return result

def extract_experiment(experiment_folder_path):
    data_rewards = []
    f = os.listdir(experiment_folder_path)
    for datafile in f:
        rs = []
        if datafile[:4] != "eval" or not datafile[-4:] == ".csv":
            continue
        file = csv.reader(open(experiment_folder_path + "/" + datafile).readlines()[1:], delimiter=';')
        for ep, _, r, _, _ in file:
            rs.append(float(r))

        data_rewards.append(rs)

    return data_rewards


def plot_experiment(sequences, legend=""):
    processed_data = []
    l = len(sequences[0])
    for seq in sequences:
        data = {"score": seq}
        df = pd.DataFrame(data)

        moving_average = df.rolling(window=25).mean()
        processed_data.append(moving_average.values.tolist())

    sns.tsplot(time=np.arange(l), data=processed_data, color=next(colors), linestyle="--", condition=legend)

if __name__ == '__main__':
    experiment_folder_paths = sys.argv[1:]
    fig = plt.figure()

    datas = []

    for efp in experiment_folder_paths:
        data = extract_experiment(efp)
        datas.append(data)

    datas = _align_to_same_length(datas)
    for idx, data in enumerate(datas):
        plot_experiment(data, legend=experiment_folder_paths[idx])

    plt.show()



