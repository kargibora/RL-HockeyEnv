import gym
import torch
import numpy as np

import hockey.hockey_env as h_env
from hockey.hockey_env import HockeyEnv,BasicOpponent

import argparse
import yaml

from utils.replay_buffer import ReplayBuffer
from utils.per import PrioritizedReplayBuffer

from methods.td3 import TD3 ,load_td3_agent_from_chkpt
import datetime 
from trainer.td3_trainer import TD3Trainer
import os
import uuid

import logging


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = setup_logger()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hockey-v0')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cfg', type=str, default='configs/td3_cfg.yaml')
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--eval_config', type=str, default='configs/eval_cfg.yaml')
    parser.add_argument('--model_name', type=str, default='_model')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    env = HockeyEnv(seed=args.seed)  # or gym.make('YourEnv-v0')
    # Initialize a policy with the same constructor
    # that you used before.
    state_dim = env.observation_space.shape[0]
    action_dim = env.num_actions
    max_action = float(env.action_space.high[0])

    logging.info(f"State dim: {state_dim}")
    logging.info(f"Action dim: {action_dim}")
    logging.info(f"Max action: {max_action}")
    
    # YAML to dict
    if args.checkpoint:
        with open(args.checkpoint + '/config.yaml', 'r') as file:
            cfg = yaml.safe_load(file)
    else:
        with open(args.cfg, 'r') as file:
            cfg = yaml.safe_load(file)

    algorithm_cfg = cfg['algorithm_cfg']
    optimizer_cfg = cfg['optim_cfg']
    log_cfg = cfg['log_cfg']
    reward_cfg = cfg['reward_cfg']
    self_play_cfg = cfg['self_play_cfg']
    if args.eval and args.checkpoint and args.eval_config:
        logging.info("Loading eval config")
        with open(args.eval_config, 'r') as file:
            eval_cfg = yaml.safe_load(file)
            self_play_cfg = eval_cfg['self_play_cfg']
        logging.info("Loaded eval config")
        logging.info(self_play_cfg)


    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        tau=0.005,
        device='cuda',
        use_layer_norm=algorithm_cfg.get('use_layer_norm', False),
        layer_norm_eps=algorithm_cfg.get('layer_norm_eps', 1e-5),
    )

    env_cfg = dict(
        dim_states=state_dim,
        dim_actions=action_dim,
    )

    replay_buffer_cfg = cfg['replay_buffer_cfg']
    if replay_buffer_cfg.get('type', 'simple') == 'simple':
        replay_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=replay_buffer_cfg.get('max_size', int(1e6))
        )
    elif replay_buffer_cfg.get('type', 'simple') == 'per':
        replay_buffer = PrioritizedReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            max_size=replay_buffer_cfg.get('max_size', int(1e6)),
            alpha=replay_buffer_cfg.get('alpha', 0.6),
        )
    else:
        raise ValueError("Replay buffer type not recognized")
    
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

    algorithm = TD3Trainer(
        env=env,
        agent=agent,
        replay_buffer=replay_buffer,
        algorithm_config=algorithm_cfg,
        env_config=env_cfg,
        optimizer_config=optimizer_cfg,
        log_config=log_cfg,
        reward_config=reward_cfg,
        self_play_config=self_play_cfg,
        gamma=0.95,
        device='cuda'
    )

    if args.checkpoint is not None and args.eval:
        agent.load(args.checkpoint,args.model_name)

        opponents = []
        # weak_opponent = BasicOpponent(weak=True)
        # weak_opponent.agent_config = {"type": "basic", "name": "weak"}
        # opponents.append(weak_opponent)
        # strong_opponent = BasicOpponent(weak=False)
        # strong_opponent.agent_config = {"type": "basic", "name": "strong"}
        # opponents.append(strong_opponent)
        eval_agents = self_play_cfg.get('eval_pretrained_agents', [])
        for i,agent_chkpt in enumerate(eval_agents):
            eval_agent = load_td3_agent_from_chkpt(
                agent_chkpt,
                model_name='_model',
                state_dim=state_dim,
                action_dim=action_dim,
                max_action=max_action,
            )

            eval_agent.agent_config = {"type": "basic", "name": f'EvalAgent_{i}'}
            opponents.append(eval_agent)
        
        results = algorithm.evaluate(
            env=env,
            opponents=opponents,
            n_eval=400,
            record_video=False
        )
        
        logging.info(f"Results: {results}")
    else:
        # Save yaml config
        with open(f"{model_chkpt_folder}/config.yaml", 'w') as file:
            yaml.dump(cfg, file)

        algorithm.train(
            5000000,
        )

        algorithm.save(
            model_chkpt_folder, '_model'
        )

