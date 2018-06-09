import argparse
import csv
import os
import sys

import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

all_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

parser = argparse.ArgumentParser(description='Plot the output from experiments')
parser.add_argument('--title', type=str, default="")
parser.add_argument('--labels', metavar='label', nargs='*', help="Labels to assign for every plot, must be len(labels)==len(paths).")
parser.add_argument('--colors', metavar='color', nargs='*', default=all_colors, help="Colors to assign for every plot.")
parser.add_argument('path_to_png', type=str, help="Where to save the plot")
parser.add_argument('paths',    metavar='datadir', nargs='+', help="Directory where to retrieve statistics from")
args = parser.parse_args()

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


def plot_experiment(sequences, color, legend=""):
    processed_data = []
    l = len(sequences[0])
    for seq in sequences:
        data = {"score": seq}
        df = pd.DataFrame(data)

        moving_average = df.rolling(window=100).mean()
        processed_data.append(moving_average.values.tolist())
        # sns.tsplot(time=np.arange(l), data=[moving_average.values.tolist()], color=color, linestyle="-")

    sns.tsplot(time=np.arange(l), data=processed_data, color=color, linestyle="-", condition=legend, ci=90)

if __name__ == '__main__':

    experiment_folder_paths = args.paths
    labels = args.labels
    if labels is None or len(labels) != len(experiment_folder_paths):
        labels = experiment_folder_paths
    colors = args.colors
    if colors is None or len(colors) != len(experiment_folder_paths):
        colors = all_colors

    fig = plt.figure()

    datas = []

    for efp in experiment_folder_paths:
        data = extract_experiment(efp)
        datas.append(data)

    datas = _align_to_same_length(datas)
    for idx, data in enumerate(datas):
        plot_experiment(data, colors[idx], legend=labels[idx])

    plt.title(args.title)
    # plt.show()
    plt.savefig(args.path_to_png)



