# Reinforcement Learning examples using RLTG

In this repository you can find some examples of Reinforcement Learning tasks (with temporally extended goals) by using the [RLTG](https://github.com/MarcoFavorito/rltg) framework.

Further details in [this document](https://github.com/MarcoFavorito/master-thesis/blob/1.0.0/thesis.pdf) (Chapter 8: Experiments).

### Breakout: remove bricks in a given order
The goal is given by using temporal logic formulas.
Examples of temporal goals can be:

- columns from left to right

![](./docs/breakout-left-right.gif)

- rows from the bottom to the top

![](./docs/breakout-bottom-top.gif)

- Both from the top to the bottom and from right to left

![](docs/breakout-top-bottom-right-left.gif)

## Repository structure

You have three command line utilities:
- `train.py`: a command-line utility to run a training job. Usage (`python train.py --help`):

      usage: train.py [-h] [--algorithm {q-learning,sarsa}] [--episodes EPISODES]
                [--gamma GAMMA] [--alpha ALPHA] [--epsilon EPSILON]
                [--lambda LAMBDA_] [--reward_shaping] [--on_the_fly]
                [--render] [--datadir DATADIR] [--verbosity {0,1,2}]
                ENVIRONMENT ...

    ...

- `resume.py`: resume a previously stopped training job. Run `python resume.py --help` to see the usage.

- `eval.py`: run the learnt policy. Run `python eval.py --help` to see the usage.

Other stuff:
- `scripts/`: contains a set of preconfigured experiments, used for benchmarking among different configurations.
- `plots/`: contains some plots of the benchmarking. Please refer to Chapter 8 of this [thesis](https://github.com/MarcoFavorito/master-thesis/blob/1.0.0/thesis.pdf).



You can use three environments (the implementation in [this repo](https://github.com/MarcoFavorito/RLgames)):
- Breakout: Reimplementation of the well-known Atari game.
- Sapientino: a kid game where pairs of colors have to be matched.
- Minecraft: a 2D implementation of a Minecraft-like environment.



## Examples
First of all, install the dependencies:

    pip install -r requirements.txt

Training (Breakout environment):

```
python train.py --gamma 0.999 --lambda 0.99 --reward_shaping --datadir my_experiment breakout --temp_goal cols
```

Evaluation:

```
python eval.py --render --datadir my_experiment
```

Resume the training job:
```
python resume.py --datadir my_experiment
```


Plot result (reward per episode with moving average):

```
python script/plot.py my_experiment
```