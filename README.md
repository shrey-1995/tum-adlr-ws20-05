# Solving Complex Sparse Reinforcement Learning Environments
IN 2349 WS-2020 project<br>
Javier Rando<br>
Shreyash Agarwal<br>

## Introduction
In this project, we tried to extend the current state-of-the-art systems to solve complex sparse reinforcement learning tasks. In many real-world scenarios, an agent faces the challenge of sparse extrinsic reward, leading to a problematic and challenging objective to solve. We will build on top of (SAC-X)[https://arxiv.org/abs/1802.10567], a technique proposed to solve these tasks by creating internal auxiliary tasks that allow the agent to efficiently explore the environment. We built two different environments for testing and analyzed algorithm convergence. Since there was no existing implementation for SAC-X available, we  provide a public version of the algorithm in this repository.

## Code summary
In this repository, we share two sparse environments for testing (2-D and 3-D) as well as:
* Our own SAC-X implementation
* SAC-X implementation on top of Spinning Up baseline SAC algoritm.

Our own implementation yields to faster convergence (5 vs 50 episodes)

## Code usage
To reproduce our experiments, you must build the 2-D environment using (Open AI Gym specifications)[https://gym.openai.com/docs/]. Our gym is located under `environments/gym-adlr`.

Then, we provide four different scripts to run each of the considered experiments:
* SAC algorithm: `sac/train_sac.py`
* Our SAC-X implementation for 2-D environment: `sacx/run_sacx.py`
* Our SAC-X implementation for 3-D environment: `sacx/run_sacx_3d.py`
* SAC-X implementation on top of Spinning Up baseline for 2-D environment: `sacx/run_sacx_spinningup.py`
