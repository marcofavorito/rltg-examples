#!/usr/bin/env bash

DATADIR=data_44tasks
python3 scripts/plot.py plots/b44-rows-comparison\
    --labels "No Reward Shaping" "Off-line Reward Shaping" "On-the-fly Reward shaping"\
    --title "Breakout 4x4, Break-Rows-BT"\
    $DATADIR/breakoutepisodes_10000__brick_rows_4__brick_cols_4__reward_shaping_False__on_the_fly_False__temp_goal_rows $DATADIR/breakoutepisodes_10000__brick_rows_4__brick_cols_4__reward_shaping_True__on_the_fly_False__temp_goal_rows $DATADIR/breakoutepisodes_10000__brick_rows_4__brick_cols_4__reward_shaping_True__on_the_fly_True__temp_goal_rows

python3 scripts/plot.py plots/b44-cols-comparison\
    --labels "No Reward Shaping" "Off-line Reward Shaping" "On-the-fly Reward shaping"\
    --title "Breakout 4x4, Break-Cols-LR"\
    $DATADIR/breakoutepisodes_10000__brick_rows_4__brick_cols_4__reward_shaping_False__on_the_fly_False__temp_goal_cols $DATADIR/breakoutepisodes_10000__brick_rows_4__brick_cols_4__reward_shaping_True__on_the_fly_False__temp_goal_cols $DATADIR/breakoutepisodes_10000__brick_rows_4__brick_cols_4__reward_shaping_True__on_the_fly_True__temp_goal_cols

python3 scripts/plot.py plots/b44-both-comparison\
    --labels "No Reward Shaping" "Off-line Reward Shaping" "On-the-fly Reward shaping"\
    --title "Breakout 4x4, Break-Cols-LR & Break-Rows-BT"\
    $DATADIR/breakoutepisodes_10000__brick_rows_4__brick_cols_4__reward_shaping_False__on_the_fly_False__temp_goal_both $DATADIR/breakoutepisodes_10000__brick_rows_4__brick_cols_4__reward_shaping_True__on_the_fly_False__temp_goal_both $DATADIR/breakoutepisodes_10000__brick_rows_4__brick_cols_4__reward_shaping_True__on_the_fly_True__temp_goal_both



DATADIR=data_new
python3 scripts/plot.py plots/rs-comparison_b33\
    --labels "No Reward Shaping" "Off-line Reward Shaping" "On-the-fly Reward shaping"\
    --title "Breakout 3x3, Break-Cols-LR"\
     $DATADIR/breakoutepisodes_15000__brick_rows_3__brick_cols_3__reward_shaping_False__on_the_fly_False__temp_goal_cols-left_right $DATADIR/breakoutepisodes_15000__brick_rows_3__brick_cols_3__reward_shaping_True__on_the_fly_False__temp_goal_cols-left_right $DATADIR/breakoutepisodes_15000__brick_rows_3__brick_cols_3__reward_shaping_True__on_the_fly_True__temp_goal_cols-left_right

python3 scripts/plot.py plots/rs-comparison_b34\
    --labels "No Reward Shaping" "Off-line Reward Shaping" "On-the-fly Reward shaping"\
    --title "Breakout 3x4, Break-Cols-LR"\
    $DATADIR/breakoutepisodes_15000__brick_rows_3__brick_cols_4__reward_shaping_False__on_the_fly_False__temp_goal_cols-left_right $DATADIR/breakoutepisodes_15000__brick_rows_3__brick_cols_4__reward_shaping_True__on_the_fly_False__temp_goal_cols-left_right $DATADIR/breakoutepisodes_15000__brick_rows_3__brick_cols_4__reward_shaping_True__on_the_fly_True__temp_goal_cols-left_right

python3 scripts/plot.py plots/rs-comparison_b44\
    --labels "No Reward Shaping" "Off-line Reward Shaping" "On-the-fly Reward shaping"\
    --title "Breakout 4x4, Break-Cols-LR"\
    $DATADIR/breakoutepisodes_15000__brick_rows_4__brick_cols_4__reward_shaping_False__on_the_fly_False__temp_goal_cols-left_right $DATADIR/breakoutepisodes_15000__brick_rows_4__brick_cols_4__reward_shaping_True__on_the_fly_False__temp_goal_cols-left_right $DATADIR/breakoutepisodes_15000__brick_rows_4__brick_cols_4__reward_shaping_True__on_the_fly_True__temp_goal_cols-left_right
