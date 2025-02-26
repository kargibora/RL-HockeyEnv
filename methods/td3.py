import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import yaml
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a)) # -1 and 1

class LayerNormActor(Actor):
    def __init__(self, state_dim, action_dim, max_action, ln_eps=1e-5):
        super(LayerNormActor, self).__init__(state_dim=state_dim, action_dim=action_dim, max_action=max_action)
        self.ln1 = nn.LayerNorm(256, eps=ln_eps)
        self.ln2 = nn.LayerNorm(256, eps=ln_eps)

    def forward(self, state):
        a = F.relu(self.ln1(self.l1(state)))
        a = F.relu(self.ln2(self.l2(a)))
        return torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1

class LayerNormCritic(Critic):
    def __init__(self, state_dim, action_dim, ln_eps=1e-5):
        super(LayerNormCritic, self).__init__(state_dim=state_dim, action_dim=action_dim)
        self.ln1 = nn.LayerNorm(256, eps=ln_eps)
        self.ln2 = nn.LayerNorm(256, eps=ln_eps)
        self.ln4 = nn.LayerNorm(256, eps=ln_eps)
        self.ln5 = nn.LayerNorm(256, eps=ln_eps)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.ln1(self.l1(sa)))
        q1 = F.relu(self.ln2(self.l2(q1)))
        q1 = self.l3(q1)

        q2 = F.relu(self.ln4(self.l4(sa)))
        q2 = F.relu(self.ln5(self.l5(q2)))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.ln1(self.l1(sa)))
        q1 = F.relu(self.ln2(self.l2(q1)))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        tau=0.005,
        use_layer_norm=False,
        layer_norm_eps=1e-5,
    ):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        # Initialize two Q networks and actor networks
        if use_layer_norm:
            self.actor = LayerNormActor(state_dim, action_dim, max_action, ln_eps=layer_norm_eps).to(self.device)
        else:
            self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        if use_layer_norm:
            self.critic = LayerNormCritic(state_dim, action_dim, ln_eps = layer_norm_eps).to(self.device)
        else:
            self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.tau = tau
        
    def act(self, observation: np.ndarray, noise: np.ndarray = None) -> np.ndarray:
        # Augment input data
        observation_t = torch.from_numpy(observation.astype(np.float32)).to(self.device)

        # Handle stacked observations
        with torch.no_grad():
            action = self.actor(observation_t)
            
        action = action.detach().cpu()

        if noise is None:
            noise = np.zeros_like(action)

        # Prepare final output
        action += noise
        action = action.clamp(-1, 1).squeeze().numpy()

        return action

    def train(self):
        self.actor.train()
        self.actor_target.train()
        self.critic.train()
        self.critic_target.train()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()

    def save(self, checkpoint_folder, model_name, appendix=""):
        filename = os.path.join(checkpoint_folder, model_name)
        torch.save(self.critic.state_dict(), filename + f"_{appendix}_critic")
        torch.save(self.actor.state_dict(), filename + f"_{appendix}_actor")

    def load(self, checkpoint_folder, model_name, appendix=""):
        filename = os.path.join(checkpoint_folder, model_name)
        self.critic.load_state_dict(torch.load(filename + f"{appendix}_critic"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + f"{appendix}_actor"))
        self.actor_target = copy.deepcopy(self.actor)

    def soft_update(self):
        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
            
    def hard_update(self):
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
    
    
def load_td3_agent_from_chkpt(checkpoint_folder,
                            appendix = "",
                            model_name = "model",
                            state_dim = 18,
                            action_dim = 4,
                            max_action = 1,
                            ) -> TD3:
    
    yaml_file = os.path.join(checkpoint_folder, "config.yaml")
    with open(yaml_file, 'r') as file:
        cfg = yaml.safe_load(file)

    algorithm_cfg = cfg['algorithm_cfg']

    agent = TD3(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device='cuda',
        tau=0.005,
        use_layer_norm=algorithm_cfg.get('use_layer_norm', False),
        layer_norm_eps=algorithm_cfg.get('layer_norm_eps', 1e-5),
    )
    agent.load(checkpoint_folder, model_name, appendix)
    
    agent.eval()
    return agent