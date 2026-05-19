"""GNN-based DQN components used by the AttenNC relay controller."""

import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:
    """Experience replay buffer for GNN-based DQN training samples."""

    def __init__(self, capacity):
        """Initialize an in-memory circular replay buffer."""
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """Store one transition in the circular replay buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a mini-batch of transitions from the buffer."""
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        """Return the number of stored transitions."""
        return len(self.buffer)


class Network(nn.Module):
    """Feed-forward attention network that scores relay actions from local features."""

    def __init__(self, arg_dict):
        """Initialize the self/path attention network for relay decisions."""
        super(Network, self).__init__()
        self.action_dim = arg_dict['action_size']
        self.K = 4
        self.parallel_path = arg_dict['max_nb']

        # Configure hidden widths for the self branch and path branch.
        num_units = arg_dict['fc2_features']
        num_path_units = arg_dict['fc2_path_features']
        num_test = num_units // 2
        self.num_test = num_test

        self.self_fc1 = nn.Linear(self.K, num_units)
        self.self_fc2 = nn.Linear(num_units, num_test)

        in_feat = self.K
        self.shared_path_fc1 = nn.Linear(in_feat, num_path_units)
        self.shared_path_fc2 = nn.Linear(num_path_units, num_test)
        self.path_norm = nn.LayerNorm(num_test)

        merge_in = num_test * 2
        self.merge_fc1 = nn.Linear(merge_in, num_units)
        self.merge_fc2 = nn.Linear(num_units, num_units)
        self.merge_fc3 = nn.Linear(num_units, self.action_dim)

    def forward(self, x):
        """Run the attention-based forward pass for relay feature batches."""
        # x: [B, (parallel_path + 1), K]
        x = x.reshape(-1, self.parallel_path + 1, self.K)

        # Normalize each feature dimension before the attention blocks.
        mean = x.mean(dim=(0, 1), keepdim=True)
        std = x.std(dim=(0, 1), keepdim=True) + 1e-8
        x = (x - mean) / std

        B = x.size(0)
        self_part = x[:, :1, :].reshape(B, -1)
        path_part = x[:, 1:, :]

        h1 = F.relu(self.self_fc1(self_part))
        self_out = F.relu(self.self_fc2(h1))

        path_outs = []
        path_segments = path_part.size(1)
        for i in range(path_segments):
            seg = path_part[:, i:(i + 1), :].reshape(B, -1)
            h_seg = F.relu(self.shared_path_fc1(seg))
            seg_out = F.relu(self.shared_path_fc2(h_seg))
            path_outs.append(seg_out)
        stack = torch.stack(path_outs, dim=2)

        attn = F.softmax(torch.matmul(self_out.unsqueeze(1), stack) / math.sqrt(self.num_test), dim=2)
        path_agg = torch.matmul(attn, stack.transpose(1, 2)).squeeze(1)
        path_agg = F.relu(self.path_norm(path_agg))

        merged = torch.cat([self_out, path_agg], dim=1)
        h = F.relu(self.merge_fc1(merged))
        h = F.relu(self.merge_fc2(h))
        out = self.merge_fc3(h)
        return out


class GNNMARL(nn.Module):
    """Single-node DQN wrapper that uses graph-style relay features."""

    def __init__(self, arg_dict, learning_rate=1e-3, gamma=0.99, buffer_size=10000, batch_size=32, target_update_freq=10):
        """Initialize online/target networks, optimizer, and replay memory."""
        super(GNNMARL, self).__init__()
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_count = 0
        self.arg_dict = arg_dict
        self.q_network = Network(self.arg_dict)
        self.device = arg_dict['device']

        self.target_q_network = Network(self.arg_dict)
        self.update_target_network()

        for param in self.target_q_network.parameters():
            param.requires_grad = False

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def forward(self, X):
        """Return relay-action Q-values for the current feature tensor."""
        q_values = self.q_network(X)
        return q_values

    def get_target_q_values(self, X):
        """Return Q-values computed by the frozen target network."""
        with torch.no_grad():
            q_values = self.target_q_network(X)
        return q_values

    def update_target_network(self):
        """Copy online-network parameters into the target network."""
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, X, epsilon=0.1):
        """Select a relay action using an epsilon-greedy policy."""
        q_values = self.forward(X)
        if random.random() < epsilon:
            return torch.tensor(random.randint(0, 1), dtype=torch.long)
        return torch.argmax(q_values)

    def remember(self, state, action, reward, next_state, done):
        """Store one transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self, device):
        """Run one DQN optimization step using a sampled replay minibatch."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(np.array(states), dtype=torch.float, device=device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        dones = torch.tensor(np.array(dones), dtype=torch.float, device=device)

        current_q_values = self.forward(states)
        with torch.no_grad():
            next_q_values = self.get_target_q_values(next_states)

        current_q_values = current_q_values.view(self.batch_size, -1)
        next_q_values = next_q_values.view(self.batch_size, -1)
        current_action_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_max_q = next_q_values.max(1)[0]
        target_q = rewards + self.gamma * (1 - dones) * next_max_q

        loss = F.mse_loss(current_action_q, target_q.detach())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def save_model(self, path):
        """Save the online and target networks to the given path."""
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)

        save_dict = {
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
        }
        torch.save(save_dict, path)
        print(f"Model saved to: {path}")

    def load_model(self, path):
        """Load the online and target networks from the given path."""
        if not os.path.exists(path):
            print(f"Model file does not exist: {path}")
            return False

        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network'])
            if 'target_q_network' in checkpoint:
                self.target_q_network.load_state_dict(checkpoint['target_q_network'])
            else:
                self.update_target_network()

            print(f"Successfully loaded model from {path}")
            return True
        except Exception as e:
            print(f"Failed to load model: {str(e)}")
            return False

    def load_network(self, path):
        """Alias of ``load_model`` for compatibility with the rest of the codebase."""
        return self.load_model(path)

    def save_network(self, path):
        """Alias of ``save_model`` for compatibility with the rest of the codebase."""
        self.save_model(path)

    def get_buffer_size(self):
        """Return the number of stored replay experiences."""
        return len(self.replay_buffer)


def normalize_features_single(features, mean=None, std=None):
    """Normalize one feature matrix and return the normalized data plus statistics."""
    arr = features.cpu().numpy() if torch.is_tensor(features) else np.asarray(features)
    if mean is None:
        mean = arr.mean(axis=0)
    if std is None:
        std = arr.std(axis=0) + 1e-8
    normed = (arr - mean) / std
    return normed, mean, std
