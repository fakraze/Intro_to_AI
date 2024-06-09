# Introduction to AI Final Project
> This project is part of the Introduction to AI course at NYCU. It involves developing three types of agents to play Mahjong.

## Installation

### Environment

We utilize the `mahjong_env` from [Github](https://github.com/ailab-pku/PyMahjongGB),  which clearly defines the rules and environment of Mahjong.

### Requirements

```
numpy==1.26.4
PyMahjongGB==1.2.0
setuptools==68.2.2
torch==2.3.0
tqdm==4.66.4
```

To install the requirement packages, use the following command:

```
pip install -r requirements.txt
```

## Dataset

We use the dataset from the [IJCAI 2024 Mahjong AI Competition](https://botzone.org.cn/static/gamecontest2024a_cn.html#).
Run this command to unzip the mahjong_data/data.zip
```
python unzip.py
```
Then, run process{1,2}.py
```
python process1.py 
# prepare dataset for NN(I)
# Dataset chi     peng    gang    bugang  angang  pass    play
# Size    232248  177092  7037    9655    5110    2511705 2338069

python process2.py 
# prepare dataset for NN(II) and NN(III)
# Dataset Discard chi     peng    gang    bugang  angang  pass
# Size    224365  232248  177092  7037    9655    5110    240804
```

## Train
run train.py to train the models. For example:
```
python train.py --model 1 # Train the discard model for NN23.
```
* `--model`:  `0`,`1` Discard{1,23} `2`,`3`,`4` Claiming{1,2,3} 
* You can see other args in train.py.

## Related Work

[Building a 3-Player Mahjong AI using Deep Reinforcement Learning](https://arxiv.org/abs/2202.12847?fbclid=IwZXh0bgNhZW0CMTAAAR0YjcYbLQcKdE3nHg887u7unZWGCm9znNFdwsnMOyK5wBfx9G9eQYzyujY_aem_AWoehQKIlg1YNNDJ5cfaHrxgHJvLQwlN1A7wy0_yN7aLtXAIYyYx9JDt0k0avpP25EWTTbWoLVXFSUtwgfqEUZxx)


## Agents

### 1. A* Agent

This agent uses `Shanten()` as a heuristic function to calculate the shortest path to win the game (HU).

### 2. Expectimax Agent

The state evaluation function for this agent is defined as: $8 - Shanten()+tiles*0.01$

### 3. Supervised Learning (SL) Agent

This agent uses a neural network to train an agent to mimic human actions when playing Mahjong.

> The detail of main approach is provided in the presentation report slide.


## Run/Test The Whole Project

To run or test the project, use the following command:
```
python eval.py # Astar agent against 3 randomBot
python eval1.py # NN(I) against 3 randomBot
python eval2.py # NN(II&III) against 3 randomBot
```
