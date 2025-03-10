from rsl_rl.modules import ActorCritic
import torch

class VAEConditionedActorCritic(ActorCritic):
    def __init__(self, num_obs, num_privileged_obs, num_actions, conditional_dim, activation='elu', **kwargs):
        super().__init__(num_obs, num_privileged_obs, num_actions, activation)
        
        # Properly handle the conditional dimensions (VAE latent + motion features)
        self.conditional_dim = conditional_dim
        
        # Build networks with proper dimensions
        actor_dims = [512, 256, 128]
        critic_dims = [512, 256, 128]
        
        self.actor = self._build_network(num_obs, actor_dims, num_actions)
        self.critic = self._build_network(num_obs, critic_dims, 1)
        
    def _build_network(self, input_dim, hidden_dims, output_dim):
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, dim))
            layers.append(torch.nn.ELU())
            prev_dim = dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        return torch.nn.Sequential(*layers)