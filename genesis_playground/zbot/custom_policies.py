from rsl_rl.modules import ActorCritic
import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.utils import resolve_nn_activation

class VAEConditionedActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs, num_privileged_obs, num_actions,
                 actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
                 activation='elu', init_noise_std=1.0, noise_std_type="scalar", **kwargs):
        super().__init__()
        
        # Store parameters
        self.num_obs = num_obs
        self.num_privileged_obs = num_privileged_obs
        self.num_actions = num_actions
        self.actor_hidden_dims = actor_hidden_dims
        self.critic_hidden_dims = critic_hidden_dims
        self.conditional_dim = kwargs.get('conditional_dim', 0)
        self.noise_std_type = noise_std_type
        self.activation_fn = resolve_nn_activation(activation)

        # Create networks
        self.actor = self._build_network(num_obs, actor_hidden_dims, num_actions)
        self.critic = self._build_network(num_privileged_obs, critic_hidden_dims, 1)
        
        # Initialize noise parameter based on type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown noise_std_type: {self.noise_std_type}. Use 'scalar' or 'log'.")

        # Initialize distribution
        self.distribution = None
        Normal.set_default_validate_args(False)

        
    def _build_network(self, input_dim, hidden_dims, output_dim):
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LayerNorm(dim))  # Added layer norm
            layers.append(self.activation_fn)  # Use the resolved activation
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)

        
    def reset(self, dones=None):
        pass
        
    def forward(self, observations):
        return self.act_inference(observations)
        
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
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
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
