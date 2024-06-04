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

## Related Work

[Building a 3-Player Mahjong AI using Deep Reinforcement Learning](https://arxiv.org/abs/2202.12847?fbclid=IwZXh0bgNhZW0CMTAAAR0YjcYbLQcKdE3nHg887u7unZWGCm9znNFdwsnMOyK5wBfx9G9eQYzyujY_aem_AWoehQKIlg1YNNDJ5cfaHrxgHJvLQwlN1A7wy0_yN7aLtXAIYyYx9JDt0k0avpP25EWTTbWoLVXFSUtwgfqEUZxx)

## Baseline

The bot provided in `mahjong_env`, which choose a single wind or honor tile first, followed by a single normal tile, and if none are available, randomly choose a two duplicate tile.

## Main Approach

### 1. A* Agent

This agent uses `Shanten()` as a heuristic function to calculate the shortest path to win the game (HU).

### 2. Expectimax Agent

The state evaluation function for this agent is defined as:$$8 - Shanten()+tiles*0.01$$

### 3. Supervised Learning (SL) Agent

This agent uses a neural network to train an agent to mimic human actions when playing Mahjong.

> The detail of main approach is provided in the presentation report slide.

## Evaluation Metric

- Use the times of *HU* of 1000 rounds as winning rate of the agent.
- Use the times of *Chunk* of 1000 rounds as bad rate of the agent.

## Run/Test The Whole Project

To run or test the project, use the following command:
```
python test_bot.py
```
