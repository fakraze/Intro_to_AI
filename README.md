# Introduction to AI Final Project
> This project is part of the Introduction to AI course at NYCU. It involves developing three types of agents to play Mahjong.

## Installation

### Environment

We utilize the `mahjong_env` from [Github](https://github.com/ailab-pku/PyMahjongGB),  which clearly defines the rules and environment of Mahjong.

### Package

To run the environment and our final project, you need to install `PyMahjongGB`, a useful package with many defined functions.

Install `MahjongGB` using the following command:
```
pip install PyMahjongGB
```

## Dataset

We use the dataset from the [IJCAI 2024 Mahjong AI Competition](https://botzone.org.cn/static/gamecontest2024a_cn.html#).

## Agents

### 1. A* Agent

This agent uses `Shanten()` as a heuristic function to calculate the shortest path to win the game (HU).

### 2. Expectimax Agent

The state evaluation function for this agent is defined as:$$8 - Shanten()+tiles*0.01$$

### 3. Supervised Learning (SL) Agent

This agent uses a neural network to train an agent to mimic human actions when playing Mahjong.


## Run/Test The Whole Project

To run or test the project, use the following command:
```
python test_bot.py
```
