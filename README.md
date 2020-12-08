# Meta-SGD: Learning to Learn Quickly for Few-Shot Learning

Implementation of Meta-SGD applied on Reinforcement Learning problems in Pytorch.
This repository includes environments introduced in ([Duan et al., 2016](https://arxiv.org/abs/1611.02779), [Finn et al., 2017](https://arxiv.org/abs/1703.03400)): multi-armed bandits, tabular MDPs, continuous control with MuJoCo, and 2D navigation task.

## Usage
You can use the [`main.py`](main.py) script in order to run reinforcement learning experiments with Meta-SGD. This script was tested with Python 3.7.
```
python main.py --env-name HalfCheetahDir-v1 --fast-lr 0.1 --lr-ppo 5e-5 --num-workers 20 --fast-batch-size 20 --meta-batch-size 20 --num-layers 2 --hidden-size 100 --num-batches 1000 --tau 0.99 --ppo-update-time 5 --output-folder maml-halfcheetah-dir-ppo --device cuda
```

This repository are based on the paper
> Li Z, Zhou F, Chen F, et al. Meta-SGD: Learning to learn quickly for few-shot learning[J]. arXiv preprint arXiv:1707.09835, 2017. [[ArXiv](https://arxiv.org/abs/1707.09835)]

