from __future__ import annotations

import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np
import yaml
from methods.td3 import TD3

from comprl.client import Agent, launch_client

class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that can be weak or strong."""

    def __init__(self, weak: bool) -> None:
        super().__init__()

        self.hockey_agent = h_env.BasicOpponent(weak=weak)

    def get_step(self, observation: list[float]) -> list[float]:
        # NOTE: If your agent is using discrete actions (0-7), you can use
        # HockeyEnv.discrete_to_continous_action to convert the action:
        #
        # from hockey.hockey_env import HockeyEnv
        # env = HockeyEnv()
        # continuous_action = env.discrete_to_continous_action(discrete_action)

        action = self.hockey_agent.act(observation).tolist()
        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

class TD3CompAgent(Agent):
    def __init__(self, agent: TD3) -> None:
        super().__init__()
        self.agent = agent

    def get_step(self, observation: list[float]) -> list[float]:
        action =  self.agent.act(np.array(observation))
        # Convert it to list of float
        return action.tolist()

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id, byteorder='big'))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        if stats[0] == stats[1]:
            text_result = "tied"
        print(
            f"[{text_result}] with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )

def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["weak", "strong", "random", "td3"],
        default="td3",
        help="Which agent to use.",
    )

    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='_model')
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    if args.agent == "weak":
        agent = HockeyAgent(weak=True)
    elif args.agent == "strong":
        agent = HockeyAgent(weak=False)
    elif args.agent == "random":
        agent = RandomAgent()
    elif args.agent == "td3":
        if args.checkpoint:
            with open(args.checkpoint + '/config.yaml', 'r') as file:
                cfg = yaml.safe_load(file)
        else:
            raise ValueError("Checkpoint is required for TD3 agent")

        algorithm_cfg = cfg['algorithm_cfg']
        td3_agent = TD3(
            state_dim=18,
            action_dim=4,
            max_action=1,
            tau=0.005,
            device='cuda',
            use_layer_norm=algorithm_cfg.get('use_layer_norm', False),
            layer_norm_eps=algorithm_cfg.get('layer_norm_eps', 1e-5),
        )
        td3_agent.load(args.checkpoint, args.model_name)
        agent = TD3CompAgent(td3_agent)

    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()