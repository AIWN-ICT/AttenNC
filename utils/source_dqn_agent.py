"""Source-side DQN agent wrapper and training utilities."""

import random
import math
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from pathlib import Path
import torch.nn.functional as F
# from utils.replay_buffer import PrioritizedReplayBuffer
from utils.replay_buffer import ReplayBuffer
from utils.source_agent_pytorch import Network_S
from typing import *
import os

UPDATE_TARGET_STEPS = 100
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
LR = 0.0001
t = 0.95


class DQNAgent_S(nn.Module):
    """Deep Q-learning agent responsible for source-node actions."""

    def __init__(self, state_size, arg_dict, device):
        """Initialize buffers, hyperparameters, and online/target networks."""
        super(DQNAgent_S, self).__init__()
        self.batch_size = 32
        self.state_size = state_size
        self.arg_dict = arg_dict
        self.device = device
        self.memory_size = 10000
        self.alpha = 0.6
        self.beta = 0.4
        self.prior_eps = 1e-6
        # self.memory = PrioritizedReplayBuffer(self.state_size, self.memory_size, self.batch_size,
        #                                       self.alpha)
        self.memory = ReplayBuffer(self.state_size, self.memory_size, self.batch_size,)
        self.gamma = 0.99  # Discount factor used in Bellman targets.
        self.train_step = 0
        self.train_start = 0
        self.learning_rate = 0.0001
        self.K = arg_dict['K']
        self.R = arg_dict['R']
        self.parallel_path = arg_dict['parallel_path']

        # Build the online network and its delayed target copy.
        self.dqn = self.create_q_network().to(device)
        self.target_net = self.create_q_network().to(device)
        self.target_net.load_state_dict(self.dqn.state_dict())

        self.target_net.eval()
        # self.dqn.reset_noise()
        # self.target_net.reset_noise()

        self.optimizer = torch.optim.Adam(self.dqn.parameters(), lr=self.learning_rate)
        self.is_test = False

        # Initialize transition storage used before replay insertion.
        # self.load_network()
        self.transition = list()
        self.epsilon = 1

    def create_q_network(self):
        """Create the source-side Q-network instance."""
        return Network_S(self.state_size, self.arg_dict, self.device).to(self.device)

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        """Compute the element-wise Smooth L1 TD loss for one sampled batch."""
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)

        # Compute the one-step TD target from the target network.
        # G_t   = r + gamma * v(s_{t+1})  if state != Terminal
        #       = r                       otherwise
        curr_q_value = self.dqn(state).gather(1, action)
        next_q_value = self.target_net(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = reward + self.gamma * next_q_value * mask

        # Compute an element-wise loss so prioritized replay can be restored later.
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")
        return elementwise_loss

    def update_model(self) -> torch.Tensor:
        """Sample replay memory and apply one optimizer step."""
        # PER needs beta to calculate weights.
        # samples = self.memory.sample_batch(self.beta)
        samples = self.memory.sample_batch()
        # weights = torch.FloatTensor(
        #     samples["weights"].reshape(-1, 1)
        # ).to(self.device)
        # indices = samples["indices"]

        # Average the per-sample losses for the current minibatch.
        elementwise_loss = self._compute_dqn_loss(samples)
        # loss = torch.mean(elementwise_loss * weights)
        loss = torch.mean(elementwise_loss)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # PER: update priorities.
        # loss_for_prior = elementwise_loss.detach().cpu().numpy()
        # new_priorities = loss_for_prior + self.prior_eps
        # self.memory.update_priorities(indices, new_priorities)

        # self.dqn.reset_noise()
        # self.target_net.reset_noise()
        self.train_step = self.train_step + 1
        return self.train_step

    def remember(self, state, action, reward, next_state, done):
        """Store one transition and trigger learning when enough samples exist."""
        self.transition = [state, action, reward, next_state, done]
        self.memory.store(*self.transition)
        if len(self.memory) > self.batch_size:
            self.update_model()

            if self.train_step % UPDATE_TARGET_STEPS == 0:
                self.update_target_soft()

    def update_target_soft(self):
        """Soft-update the target network using factor ``t``."""
        if self.train_step % UPDATE_TARGET_STEPS == 0:
            for target_param, source_param in zip(self.target_net.parameters(), self.dqn.parameters()):
                target_param.data.copy_((1 - t) * target_param.data + t * source_param.data)

    def get_q_val(self, state_batch):
        """Return Q-values for a batch of source states."""
        q_val = self.dqn(torch.FloatTensor(state_batch).to(self.device)).detach().cpu().numpy()
        return q_val

    def get_q_val_with_cache(self, state_batch):
        """Run inference with path-feature caching for repeated identical inputs."""
        # Preserve the original training mode of the online network.
        training_state = self.dqn.training

        # Switch to evaluation mode for deterministic inference behavior.
        self.dqn.eval()

        # Normalize the input so the forward pass always sees a batch dimension.
        if isinstance(state_batch, np.ndarray):
            state_batch = torch.FloatTensor(state_batch)
        if len(state_batch.shape) == 1:
            state_batch = state_batch.unsqueeze(0)

        # Compute Q-values without tracking gradients.
        with torch.no_grad():
            q_val = self.dqn(state_batch.to(self.device)).detach().cpu().numpy()

        # Restore the original training/evaluation mode.
        self.dqn.train(training_state)

        return q_val

    def clear_cache(self):
        """Clear cached path embeddings after a repeated inference sequence."""
        self.dqn.clear_cache()

    def load_network(self, path):
        """Load model parameters from a state dict or serialized model object."""
        loaded = torch.load(path, map_location=self.device)

        if isinstance(loaded, dict):
            self.dqn.load_state_dict(loaded)
        else:
            self.dqn.load_state_dict(loaded.state_dict())

        # Keep the target network synchronized with the loaded online weights.
        self.target_net.load_state_dict(self.dqn.state_dict())

    def save_network(self, path):
        """Save the online-network parameters to disk, creating parent folders."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        try:
            torch.save(self.dqn.state_dict(), path)
            print(f"Successfully saved model to: {path}")
        except Exception as e:
            print(f"Failed to save model: {str(e)}")
