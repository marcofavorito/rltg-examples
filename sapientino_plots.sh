#!/usr/bin/env bash

DATADIR=data_sapientino
python3 scripts/plot.py plots/sapientino-relaxed\
    --labels "No Reward Shaping" "Off-line Reward Shaping" "On-the-fly Reward shaping"\
    --title "Sapientino 'Relaxed'"\
    ./data_sapientino/sapientino_episodes_750__reward_shaping_False__on_the_fly_False__temp_goal_colors_relaxed /home/marcofavorito/Workfolder/rltg-examples/data_sapientino/sapientino_episodes_750__reward_shaping_True__on_the_fly_False__temp_goal_colors_relaxed /home/marcofavorito/Workfolder/rltg-examples/data_sapientino/sapientino_episodes_750__reward_shaping_True__on_the_fly_True__temp_goal_colors_relaxed

DATADIR=data_sapientino
python3 scripts/plot.py plots/sapientino-full\
    --labels "No Reward Shaping" "Off-line Reward Shaping" "On-the-fly Reward shaping"\
    --title "Sapientino 'Full'"\
    ./data_sapientino/sapientino_episodes_750__reward_shaping_False__on_the_fly_False__temp_goal_colors /home/marcofavorito/Workfolder/rltg-examples/data_sapientino/sapientino_episodes_750__reward_shaping_True__on_the_fly_False__temp_goal_colors /home/marcofavorito/Workfolder/rltg-examples/data_sapientino/sapientino_episodes_750__reward_shaping_True__on_the_fly_True__temp_goal_colors