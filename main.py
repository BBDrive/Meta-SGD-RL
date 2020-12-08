import meta_sgd_rl.envs
import gym
import numpy as np
import torch
import json
import time

from meta_sgd_rl.metalearner import MetaLearner
from meta_sgd_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from meta_sgd_rl.baseline import LinearFeatureBaseline
from meta_sgd_rl.sampler import BatchSampler

from tensorboardX import SummaryWriter


def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
                                      for rewards in episodes_rewards], dim=0))
    return rewards.item()


def main(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
                                            'AntPos-v0', 'HalfCheetahVel-v1',
                                            'HalfCheetahDir-v1', '2DNavigation-v0'])

    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_folder = './saves/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)  # indent 缩进

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
                           num_workers=args.num_workers)
    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape)),
            sampler.envs.action_space.n,
            hidden_sizes=(args.hidden_size,) * args.num_layers)
    baseline = LinearFeatureBaseline(
        int(np.prod(sampler.envs.observation_space.shape)))

    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma,
                              tau=args.tau, clip_param=args.clip_param,
                              lr_ppo=args.lr_ppo, device=args.device)

    for batch in range(args.num_batches):  # num_batches: 大循环
        time_start = time.time()
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)  # meta_batch_size: 共有多少任务
        episodes = metalearner.sample(tasks)

        # if batch % 5 == 0:
        #     metalearner.lr_ppo *= 0.989
        #     metalearner.optimizer = torch.optim.Adam(metalearner.model_params, lr=metalearner.lr_ppo)

        metalearner.step(episodes,  ppo_update_time=args.ppo_update_time)

        # Tensorboard
        writer.add_scalar('total_rewards/before_update',
                          total_rewards([ep.rewards for ep, _ in episodes]), batch)
        writer.add_scalar('total_rewards/after_update',
                          total_rewards([ep.rewards for _, ep in episodes]), batch)

        # Save policy network
        with open(os.path.join(save_folder,
                'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(policy.state_dict(), f)

        print("Training: Batch {} of {} total batch;"
              "Time: {:.2f}s; reward: {:.2f}".format(batch, args.num_batches,
                                                     time.time()-time_start,
                                                     total_rewards([ep.rewards for _, ep in episodes])))


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Reinforcement learning with '
                                                 'Model-Agnostic Meta-Learning (MAML)')

    # General
    parser.add_argument('--env-name', type=str,
                        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='value of the discount factor gamma')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='value of the discount factor for GAE')

    # Policy network (relu activation function)
    parser.add_argument('--hidden-size', type=int, default=100,
                        help='number of hidden units per layer')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='number of hidden layers')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=20,
                        help='batch size for each individual task')  # batch_size 即为一个任务有多少trajectories

    # Optimization
    parser.add_argument('--num-batches', type=int, default=200,
                        help='number of batches')  # num_batches: 大循环
    parser.add_argument('--meta-batch-size', type=int, default=20,
                        help='number of tasks per batch')  # meta_batch_size: 共有多少任务
    parser.add_argument('--ppo-update-time', type=int, default=5,
                        help='maximum number of iterations for line search')
    parser.add_argument('--clip-param', type=float, default=0.2,
                        help='scope of ppo loss ')
    parser.add_argument('--lr-ppo', type=float, default=5e-5,
                        help='the learning rate of ppo')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
                        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                        help='number of workers for trajectories sampling')  # 几个进程收集trajectories
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda)')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
