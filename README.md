# Coordinating Multi-Agent Reinforcement Learning via Dual Collaborative Constraints

Code for the paper "Coordinating Multi-Agent Reinforcement Learning via Dual Collaborative Constraints" submitted to Neural Networks. 

This repository develops DCC algorithm on both Multi-agent Particle Environments and StarCraft Multi-Agent Challenge benchmarks, and
compares it with several baselines including LDSA, MAVEN, ROMA, QMIX, HSD and RODE_nr. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the approach in the paper, run this command:

```train
python main.py
```

You can select the training task between MPE and SMAC by setting ```--env-config='sc2' or 'mpe'```.
Also you can select the training algorithm by setting ```--config='dcc'```

## Hyper-parameters

To modify the hyper-parameters of algorithms and environments, refer to:

```
src/config/algs/dcc.yaml
src/config/default.yaml
```
```
src/config/envs/mpe.yaml
src/config/envs/sc2.yaml
```

## Note

This repository is developed based on PyMARL. And we have cited the SMAC paper in our work.
