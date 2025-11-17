import numpy as np
import torch

from models.ppo import PPOAgent, PPOConfig


def test_actor_and_critic_shapes():
    config = PPOConfig(rollout_horizon=2, minibatch_size=1)
    agent = PPOAgent(obs_dim=6, action_dim=3, config=config)
    pop = np.random.randn(10, 6).astype(np.float32)
    features = agent._encode(pop)
    dist = agent._distribution(features)
    alpha = dist.concentration
    assert torch.all(alpha > 0)
    action = dist.sample()
    logp = dist.log_prob(action)
    assert logp.shape == torch.Size([1])
    value = agent.critic(features)
    assert value.shape == torch.Size([1])
