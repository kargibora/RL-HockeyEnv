import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from methods.td3 import TD3, load_td3_agent_from_chkpt
from utils.replay_buffer import ReplayBuffer
from utils.per import PrioritizedReplayBuffer
from utils.noise import sample_pink_noise, sample_gaussian_noise, sample_golden_noise, sample_brown_noise
from utils.rewards import calculate_dense_rewards_with_weights

from typing import Any
from hockey.hockey_env import HockeyEnv,BasicOpponent
import tqdm
import datetime
import os

from torch.utils.tensorboard import SummaryWriter
from typing import Dict
import copy
import cv2
import logging

class TD3Trainer():
    def __init__(
        self,
        env,
        agent : TD3,
        replay_buffer: ReplayBuffer,
        env_config: dict[str, Any],
        algorithm_config: dict[str, Any],
        optimizer_config: tuple[dict[str, Any]],
        reward_config: dict[str, Any],
        log_config: dict[str, Any],
        self_play_config: dict[str, Any],
        gamma: float,
        device,
    ) -> None:
        
        self.env = env
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.algorithm_config = algorithm_config
        self.optimizer_config = optimizer_config
        self.reward_config = reward_config
        self.self_play_config = self_play_config
        self.env_config = env_config
        self.log_config = log_config
        self.gamma = gamma

        self.starting_beta = self.algorithm_config["beta"]
        self.ending_beta = 1.0
        self.current_beta = self.starting_beta

        self.critic_criterion = nn.MSELoss()
        self.critic_optimizer = torch.optim.Adam(
            self.agent.critic.parameters(),
            **self.optimizer_config['critic_config'],
        )

        self.actor_optimizer = torch.optim.Adam(
            self.agent.actor.parameters(),
            **self.optimizer_config['actor_config'],
        )
        self.device = device
        self.learning_starts = self.algorithm_config["learning_starts"]
        self.n_eval = self.algorithm_config["n_eval_episodes"]
        self.max_steps = self.algorithm_config["max_steps"]

        self.num_timesteps = 0
        self.num_eps = 0

        self.num_actions = self.env_config["dim_actions"]
        if self.algorithm_config["noise"] == "gaussian":
            self.sample_noise_f = sample_gaussian_noise
        elif self.algorithm_config["noise"] == "pink":
            self.sample_noise_f = sample_pink_noise
        elif self.algorithm_config["noise"] == "golden":
            self.sample_noise_f = sample_golden_noise
        elif self.algorithm_config["noise"] == "brown":
            self.sample_noise_f = sample_brown_noise
        else:
            raise NotImplementedError
        
        if log_config["writer_folder"] is None:
            self.writer = None
        else:
            self.writer = SummaryWriter(
                log_dir=self.log_config["writer_folder"],
                filename_suffix=self.log_config["log_filename_suffix"],
            )

        self.best_win_percentage = 0
        self.best_win_percentage_strong = 0
        self.best_win_percentage_weak = 0
        self.pretrained_countdown = 0
        self.best_agent = None
        
        self.reward_betas = [
            self.reward_config['winner'],
            self.reward_config['reward_touch_puck'],
            self.reward_config['reward_closeness_to_puck'],
            self.reward_config['reward_puck_direction'],
        ]

        self.self_train = self.self_play_config.get('self_train', False)
        self.self_playing_threshold = self.self_play_config.get('self_playing_threshold', 1.0)
        self.self_training_max_agents = self.self_play_config.get('self_training_max_agents', 5)

        self.start_pretrained = self.self_play_config.get('start_pretrained', None)
        self.pretrained_added = False   


    def setup_default_opponents(self):
        # Create basic opponents for evaluation and training
        self.weak_opponent = BasicOpponent(weak=True)
        self.strong_opponent = BasicOpponent(weak=False)

        self.weak_opponent.name = "weak"
        self.strong_opponent.name = "strong"

        # Set up opponent lists
        self.opponents = [self.weak_opponent]
        self.self_training_opponents = []
        self.eval_opponents = [self.weak_opponent, self.strong_opponent]
        self.eval_pretrained_opponents = [self.weak_opponent, self.strong_opponent]
        self.train_opponents = []

    def get_current_progress(self):
        return self.num_timesteps / self.total_timesteps
    
    def train(self, n_steps: int) -> None:
        num_collected_observations = 0

        # Initialize training parameters
        self.total_timesteps = n_steps
        self.pretrained_countdown = 0
        self.pretrained_added = False

        # Reset the environment and store initial observations
        last_obs, _ = self.env.reset()
        self.last_observation_agent = last_obs
        self.last_observation_opponent = self.env.obs_agent_two()

        self.setup_default_opponents()
        # Load pretrained evaluation agents
        eval_agents = self.self_play_config.get('eval_pretrained_agents', [])
        for i, agent_chkpt in enumerate(eval_agents):
            eval_agent = load_td3_agent_from_chkpt(
                agent_chkpt,
                model_name='_model',
                state_dim=self.agent.state_dim,
                action_dim=self.agent.action_dim,
                max_action=self.agent.max_action,
            )
            eval_agent.name = f"EvalAgent_{i}"
            logging.info(f"[EVAL] Loaded pretrained agent {eval_agent.name}")
            self.eval_pretrained_opponents.append(eval_agent)

        # Load pretrained training agents
        train_agents = self.self_play_config.get('train_pretrained_agents', [])
        for i, agent_chkpt in enumerate(train_agents):
            train_agent = load_td3_agent_from_chkpt(
                agent_chkpt,
                model_name='_model',
                state_dim=self.agent.state_dim,
                action_dim=self.agent.action_dim,
                max_action=self.agent.max_action,
            )
            train_agent.name = f"TrainAgent_{i}"
            logging.info(f"[TRAIN] Loaded pretrained agent {train_agent.name}")
            self.train_opponents.append(train_agent)

        # Main training loop with progress tracking
        pbar = tqdm.tqdm(total=n_steps)
        while self.num_timesteps < self.total_timesteps:
            # Collect rollout samples and update observation count
            n_gradient_steps = self.rollout_collect()
            num_collected_observations += n_gradient_steps

            # Begin training once sufficient observations have been gathered
            if num_collected_observations > self.learning_starts:
                self.train_agent_steps(num_steps=n_gradient_steps)
                pbar.update(n_gradient_steps)

                # Add the strong opponent once exploration is underway
                if self.strong_opponent not in self.opponents:
                    self.opponents.append(self.strong_opponent)

                # Introduce pretrained agents when performance criteria are met
                if not self.pretrained_added:
                    if self.best_win_percentage_strong > self.self_playing_threshold:
                        self.pretrained_countdown += n_gradient_steps

                    if self.start_pretrained and self.pretrained_countdown >= self.start_pretrained:
                        logging.info("Adding pretrained agents to the pool")
                        for agent in self.train_opponents:
                            self.opponents.append(agent)

                        for agent in self.eval_pretrained_opponents:
                            self.eval_opponents.append(agent)

                        self.pretrained_added = True


    def choose_opponent(self, opponents) -> object:
        """Randomly select an opponent from the available pool."""
        return np.random.choice(opponents)

    def rollout_collect(self) -> int:
        # Set agent to evaluation mode.
        self.agent.eval()

        # Basic parameters and noise generation for the episode.
        max_episode_steps = self.max_steps
        action_dimension = self.num_actions
        noise_sampler = self.sample_noise_f
        base_action_noise = self.algorithm_config["action_noise"]
        noise_sequence = noise_sampler(shape=(max_episode_steps, action_dimension))

        # Choose an opponent and determine the noise parameters.
        current_opponent = self.choose_opponent(self.opponents + self.self_training_opponents)

        step_idx = 0
        with torch.no_grad():
            while step_idx <= max_episode_steps:
                # Compute the agent's action with added noise.
                current_noise = base_action_noise * noise_sequence[step_idx]
                agent_action = self.agent.act(self.last_observation_agent, current_noise)

                # Compute the opponent's action and add Gaussian noise.
                opp_action = current_opponent.act(self.last_observation_opponent)
                opp_action = np.clip(opp_action, -1.0, 1.0)

                # Merge both actions.
                combined_action = np.hstack((agent_action, opp_action))

                # Step the environment.
                new_state, reward, done, _, info = self.env.step(combined_action)
                info["done"] = done

                # Optionally modify the reward.
                dense_reward = calculate_dense_rewards_with_weights(
                    info,
                    betas=self.reward_betas
                    )
                
                # Convert observations and actions to torch tensors.
                new_state_tensor = torch.from_numpy(new_state.astype(np.float32))
                agent_action_tensor = torch.from_numpy(agent_action.astype(np.float32))
                last_state_tensor = torch.from_numpy(self.last_observation_agent)

                if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                    # Beta should start from low value and gradually increase to 1
                    run_ratio = self.get_current_progress()
                    # Slowly scale beta to 1
                    self.beta = self.starting_beta + run_ratio * (self.ending_beta - self.starting_beta)

                # Store the transition in the replay buffer.
                self.replay_buffer.add(
                    last_state_tensor.cpu(),
                    agent_action_tensor.cpu(),
                    new_state_tensor.cpu(),
                    dense_reward,
                    done,
                )

                step_idx += 1

                # Check if the episode has ended or maximum steps reached.
                if done or step_idx == max_episode_steps:
                    self.last_observation_agent, _ = self.env.reset()
                    self.last_observation_opponent = self.env.obs_agent_two()
                    break
                else:
                    self.last_observation_agent = new_state
                    self.last_observation_opponent = self.env.obs_agent_two()

        return step_idx
            

    def train_agent_steps(self, num_steps: int):
        # Set the agent to training mode
        self.agent.train()

        # Extract configuration parameters
        cfg = self.algorithm_config
        bs = cfg["batch_size"]
        delay_steps = cfg["policy_delay"]
        tgt_noise_std = cfg["target_policy_noise"]
        tgt_noise_bound = cfg["target_noise_clip"]

        # Track losses for logging
        actor_loss_history = []
        critic_loss_history = []

        for _ in range(num_steps):
            # Sample a mini-batch from the replay buffer and move tensors to the current device
            if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                batch, weights, indices = self.replay_buffer.sample(bs, self.beta)

            else:
                batch = self.replay_buffer.sample(bs)
                weights = np.ones(bs) # 1/m for all samples

            weights_t = torch.from_numpy(weights).to(self.device)
            obs_batch = batch[0].to(self.device)
            action_batch = batch[1].to(self.device)
            next_obs_batch = batch[2].to(self.device)
            reward_batch = batch[3].to(self.device)
            done_batch = batch[4].to(self.device)
            
            # Compute the TD target using the target networks
            with torch.no_grad():
                # Generate target noise and constrain it
                noise = torch.normal(mean=0.0, std=tgt_noise_std, size=action_batch.shape, device=self.device)
                noise = torch.clamp(noise, -tgt_noise_bound, tgt_noise_bound)

                # Compute the next actions from the target actor network, adding the clipped noise
                next_actions = self.agent.actor_target(next_obs_batch) + noise
                next_actions = torch.clamp(next_actions, -1, 1)

                # Evaluate both target critics and use the minimum Q-value
                target_q1, target_q2 = self.agent.critic_target(next_obs_batch, next_actions)
                next_q_value = torch.min(target_q1, target_q2)
                # Bellman backup to get the TD target
                td_target = reward_batch + (1 - done_batch) * self.gamma * next_q_value

            # Get current Q estimates from both critics
            current_q1, current_q2 = self.agent.critic(obs_batch, action_batch)
            loss_q1 = F.mse_loss(current_q1, td_target, reduction='none')
            loss_q2 = F.mse_loss(current_q2, td_target, reduction='none')
            critic_loss_ew = loss_q1 + loss_q2
            critic_loss = (critic_loss_ew * weights_t).mean()

            if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
                # Also update the priorities in the replay buffer
                new_priorities = critic_loss_ew.detach().cpu().numpy() + 1e-5 # Loss shouldnt be 0 therefore add small value
                self.replay_buffer.update_priorities(indices, new_priorities)

            # Log critic loss and update the critic network
            critic_loss_history.append(critic_loss.item())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Update the actor network only at specified intervals
            if self.num_eps % delay_steps == 0:
                # The actor loss is defined as the negative mean Q-value from one of the critics
                predicted_actions = self.agent.actor(obs_batch)
                actor_loss = -self.agent.critic.Q1(obs_batch, predicted_actions).mean()

                actor_loss_history.append(actor_loss.item())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Soft update target networks using an exponential moving average scheme
                self.agent.soft_update()

            if self.num_timesteps % 100000 == 0:
                self.agent.eval()
                results = evaluate_multiple_opponents(
                    self.env,
                    self.agent,
                    self.eval_opponents,
                    self.n_eval,
                    base_tag="",
                )

                logging.info(f"Step {self.num_timesteps}: {results}")

                for k,v in results.items():
                    self.writer.add_scalar(k, v, self.num_timesteps)

                if results["avg_win_percentage"] >= self.best_win_percentage:
                    self.best_win_percentage = results["avg_win_percentage"]
                    self.best_agent = copy.deepcopy(self.agent)

                if results["win_percentage_strong"] >= self.best_win_percentage_strong:
                    self.best_win_percentage_strong = results["win_percentage_strong"]

                if self.self_train and self.best_win_percentage_strong > self.self_playing_threshold:
                    logging.info(f"[Time step: {self.num_timesteps}] Adding the best agent to self-training pool with win percentage {results['win_percentage_strong']}")
                    self.self_training_opponents.append(copy.deepcopy(self.agent))
                    if len(self.self_training_opponents) > self.self_training_max_agents:
                        self.self_training_opponents.pop(0)
                self.agent.train()               

            if self.num_timesteps % self.log_config["save_interval"] == 0 and self.log_config["model_chkpt_folder"] is not None:
                logging.info(f"Saving model at timestep {self.num_timesteps}")
                self.save(self.log_config["model_chkpt_folder"], self.log_config["model_name"])
                self.save(self.log_config["model_chkpt_folder"], self.log_config["model_name"], self.best_agent, f"best_{self.num_timesteps}")
            self.num_timesteps += 1
        self.num_eps += 1

        # Log the average loss values for the current training step
        if actor_loss_history:
            avg_actor_loss = np.mean(actor_loss_history)
            self.writer.add_scalar("losses/actor_loss", avg_actor_loss, self.num_eps)
        if critic_loss_history:
            avg_critic_loss = np.mean(critic_loss_history)
            self.writer.add_scalar("losses/critic_loss", avg_critic_loss, self.num_eps)
        
    def save(self, checkpoint_folder, model_name, agent = None, appendix = ""):
        if agent is None:
            agent = self.agent
        filename = os.path.join(checkpoint_folder, model_name)
        self.agent.save(checkpoint_folder, model_name, appendix)
        torch.save(self.critic_optimizer.state_dict(), filename + f"_{appendix}_critic_optimizer")
        torch.save(self.actor_optimizer.state_dict(), filename + f"_{appendix}_actor_optimizer")

    def load(self, checkpoint_folder, model_name, agent = None, appendix = ""):
        if agent is None:
            agent = self.agent
        filename = os.path.join(checkpoint_folder, model_name)
        self.agent.load(checkpoint_folder, model_name, appendix)
        self.critic_optimizer.load_state_dict(torch.load(filename + f"_{appendix}_critic_optimizer"))
        self.actor_optimizer.load_state_dict(torch.load(filename + f"_{appendix}_actor_optimizer"))
        
    def evaluate(self,env, opponents, n_eval, record_video = False):
        results = evaluate_multiple_opponents(
            env,
            self.agent,
            opponents,
            n_eval,
            base_tag="",
            record_video=record_video,
        )
        return results


def run_episode(env, player, rival, dense_reward_func, video_storage):

    # Reset environment and optionally reset agent memory
    cum_reward = 0
    cum_dense = 0
    obs, _ = env.reset()
    rival_obs = env.obs_agent_two()
    
    # Run one episode until termination or max timesteps reached
    for _ in range(env.max_timesteps):
        act_player = player.act(obs)
        act_rival = rival.act(rival_obs)
        combined_actions = np.hstack([act_player, act_rival])
        obs, step_reward, done, _, step_info = env.step(combined_actions)
        
        # Capture frame for video recording
        frame = env.render(mode='rgb_array')
        video_storage.append(frame)
        step_info["done"] = done
        rival_obs = env.obs_agent_two()
        
        cum_reward += step_reward
        cum_dense += dense_reward_func(step_info, betas=[10.0, 1.0, 0.05, 3.0])
        
        if done:
            break
    
    # Determine win status and compute sparse reward
    win_flag = 1 if env.winner == 1 else 0
    sparse_val = 10 * env.winner
    return sparse_val, cum_dense, win_flag

def evaluate_agent_against_opponent(env, player, opponent, num_episodes, tag, record_video=False):
    sparse_rewards = []
    dense_rewards = []
    win_flags = []
    video_frames = []
    
    for _ in range(num_episodes):
        s_reward, d_reward, win_status = run_episode(env, player, opponent, calculate_dense_rewards_with_weights, video_frames)
        sparse_rewards.append(s_reward)
        dense_rewards.append(d_reward)
        win_flags.append(win_status)
    
    # Optionally record video
    if record_video and video_frames:
        frame_h, frame_w, _ = video_frames[0].shape
        writer = cv2.VideoWriter(
            f'./{tag}_video.mp4', 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            30, 
            (frame_w, frame_h)
        )
        for img in video_frames:
            writer.write(img)
        writer.release()
    
    # Compute averages
    avg_sparse = np.mean(sparse_rewards).item()
    avg_dense = np.mean(dense_rewards).item()
    win_rate = np.mean(win_flags).item()
    
    # Build result keys with tag if provided
    key_sparse = "avg_sparse_reward" + ("_" + tag if tag else "")
    key_dense = "avg_dense_reward" + ("_" + tag if tag else "")
    key_win = "win_percentage" + ("_" + tag if tag else "")
    
    return {
        key_sparse: avg_sparse,
        key_dense: avg_dense,
        key_win: win_rate,
    }

def evaluate_multiple_opponents(env, player, opponents_list, num_episodes, base_tag, record_video=False):
    overall_results = {}
    win_rates = []
    
    if len(opponents_list) == 0:
        return overall_results
    
    for foe in opponents_list:
        foe_tag = foe.agent_config["name"]
        if base_tag:
            foe_tag += "_" + base_tag
        outcome = evaluate_agent_against_opponent(env, player, foe, num_episodes, foe_tag, record_video)
        overall_results.update(outcome)
        # Extract win rate for overall average calculation
        for key, value in outcome.items():
            if "win_percentage" in key:
                win_rates.append(value)
    
    overall_win = np.mean(win_rates)
    overall_key = "avg_win_percentage" + ("_" + base_tag if base_tag else "")
    overall_results[overall_key] = overall_win
    return overall_results

