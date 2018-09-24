#!/usr/bin/env bash

DATADIR=data_minecraft
python3 scripts/plot.py plots/minecraft\
    --labels "No Reward Shaping" "Off-line Reward Shaping" "On-the-fly Reward shaping"\
    --title "Minecraft"\
    ./data_minecraft/minecraftepisodes_10000__reward_shaping_False__on_the_fly_False__temp_goal_all /home/marcofavorito/Workfolder/rltg-examples/data_minecraft/minecraftepisodes_10000__reward_shaping_True__on_the_fly_False__temp_goal_all /home/marcofavorito/Workfolder/rltg-examples/data_minecraft/minecraftepisodes_10000__reward_shaping_True__on_the_fly_True__temp_goal_all