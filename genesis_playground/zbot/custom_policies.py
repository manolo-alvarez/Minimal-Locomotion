from rsl_rl.modules import ActorCritic
import torch
import torch.nn as nn
from torch.distributions import Normal

class VAEConditionedActorCritic(nn.Module):
    is_recurrent = False
    
    def __init__(self, num_obs, num_privileged_obs, num_actions, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation='elu', init_noise_std=1.0, **kwargs):
        super().__init__()
        
        # Store parameters
        self.num_obs = num_obs
        self.num_privileged_obs = num_privileged_obs
        self.num_actions = num_actions
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims = critic_hidden_dims
        self.conditional_dim = kwargs.get('conditional_dim', 0)
        
        # Create networks
        self.actor = self._build_network(num_obs, actor_hidden_dims, num_actions)
        self.critic = self._build_network(num_privileged_obs, critic_hidden_dims, 1)
        
        # Initialize noise parameter
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        
        # Initialize distribution
        self.distribution = None
        Normal.set_default_validate_args(False)
        
    def _build_network(self, input_dim, hidden_dims, output_dim):
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.ELU())
            prev_dim = dim
            
        # Add output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        return nn.Sequential(*layers)
        
    def reset(self, dones=None):
        pass
        
    def forward(self):
        raise NotImplementedError
        
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        std = torch.exp(self.std).expand_as(mean)  # Apply exp to ensure positive std
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value