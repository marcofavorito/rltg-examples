# rltg-examples
Reinforcement Learning examples using RLTG

## Examples

Training:

```
python train.py --gamma 0.999 --lambda 0.99 --reward_shaping --datadir my_experiment breakout --temp_goal cols
```

Evaluation:

```
python eval.py --render --datadir my_experiment
```

Plot result (reward per episode with moving average):

```
python script/plot.py my_experiment
```