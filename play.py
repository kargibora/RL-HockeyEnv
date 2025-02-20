import gym
import torch
import numpy as np
from hockey.hockey_env import HockeyEnv,BasicOpponent

import argparse
import yaml

from utils.replay_buffer import ReplayBuffer

from methods.td3 import TD3  # or your method
import datetime 
from trainer.td3_trainer import TD3Algorithm
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hockey-v0')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cfg', type=str, default='configs/td3_cfg.yaml')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    env = HockeyEnv(seed=args.seed)  # or gym.make('YourEnv-v0')
    # Initialize a policy with the same constructor
    # that you used before.
    state_dim = env.observation_space.shape[0]
    action_dim = env.num_actions
    max_action = float(env.action_space.high[0])

    print(f"State dim: {state_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Max action: {max_action}")
    
    # YAML to dict
    with open(args.cfg, 'r') as file:
        cfg = yaml.safe_load(file)

    algorithm_cfg = cfg['algorithm_cfg']
    optimizer_cfg = cfg['optim_cfg']
    log_cfg = cfg['log_cfg']
    reward_cfg = cfg['reward_cfg']


    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        tau=0.005,
        device='cuda'
    )

    env_cfg = dict(
        dim_states=state_dim,
        dim_actions=action_dim,
    )

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        max_size=1000000,
    )

    # Load the saved model
    # policy.load("./models/TD3_Hockey-v0_0")  # update path as needed
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    if args.exp_name:
        model_chkpt_folder = f"./models/{args.exp_name}_{args.env}_{args.seed}_{date}"
        writer_folder = f"./logs/{args.exp_name}_{args.env}_{args.seed}_{date}"
        os.makedirs(model_chkpt_folder, exist_ok=True)
        log_cfg['save_interval'] = 1000000
        log_cfg['model_chkpt_folder'] = model_chkpt_folder
        log_cfg['writer_folder'] = writer_folder
    else:
        log_cfg['writer_folder'] = None
        log_cfg['model_chkpt_folder'] = None



    algorithm = TD3Algorithm(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        algorithm_config=algorithm_cfg,
        env_config=env_cfg,
        optimizer_config=optimizer_cfg,
        log_config=log_cfg,
        reward_config=reward_cfg,
        gamma=0.95,
        device='cuda'
    )

    if args.checkpoint is not None and args.eval:
        algorithm.load(args.checkpoint)
        weak_opponent = BasicOpponent(weak=True)
        weak_opponent.agent_config = {"type": "basic", "name": "weak"}
        strong_opponent = BasicOpponent(weak=False)
        strong_opponent.agent_config = {"type": "basic", "name": "strong"}
        results = algorithm.evaluate(
            env=env,
            opponents=[weak_opponent, strong_opponent],
            n_eval=5
        )
        print(results)
    else:
        algorithm.train(
            10000000,
        )

        algorithm.save(
            model_chkpt_folder
        )