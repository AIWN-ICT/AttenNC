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
    """Two-layer GAT-based network that scores relay actions from local features."""

    def __init__(self, arg_dict):
        """Initialize a 2-layer, 2-head GAT for relay decisions."""
        super(Network, self).__init__()
        self.action_dim = arg_dict['action_size']
        self.K = int(arg_dict.get('K', 4))
        self.parallel_path = arg_dict['max_nb']
        self.num_heads = int(arg_dict.get('num_heads', 1))
        if self.K % self.num_heads != 0:
            raise ValueError(f"K ({self.K}) must be divisible by num_heads ({self.num_heads}).")
        self.head_dim = self.K // self.num_heads

        num_units = arg_dict['fc2_features']
        num_test = num_units // 2

        # Layer 1: K -> K (per head K/num_heads, concat heads keeps K unchanged)
        self.gat1_fc = nn.Linear(self.K, self.K, bias=False)
        self.gat1_attn = nn.Parameter(torch.empty(self.num_heads, 2 * self.head_dim))

        # Layer 2: K -> K (same output dimensionality as input)
        self.gat2_fc = nn.Linear(self.K, self.K, bias=False)
        self.gat2_attn = nn.Parameter(torch.empty(self.num_heads, 2 * self.head_dim))

        self.self_proj = nn.Linear(self.K, num_test)
        self.path_proj = nn.Linear(self.K, num_test)
        self.path_norm = nn.LayerNorm(num_test)

        merge_in = num_test * 2
        self.merge_fc1 = nn.Linear(merge_in, num_units)
        self.merge_fc2 = nn.Linear(num_units, num_units)
        self.merge_fc3 = nn.Linear(num_units, self.action_dim)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize trainable parameters for the GAT layers."""
        nn.init.xavier_uniform_(self.gat1_fc.weight)
        nn.init.xavier_uniform_(self.gat2_fc.weight)
        nn.init.xavier_uniform_(self.gat1_attn)
        nn.init.xavier_uniform_(self.gat2_attn)

    def _gat_layer(self, x, proj, attn_vec, adj_mask=None):
        """Apply one multi-head graph attention layer.

        Args:
            x: [B, N, K]
            adj_mask: Optional adjacency mask, shape [B, N, N] or [N, N].
                      Non-zero entries indicate valid edges.
        """
        B, N, _ = x.shape
        h = proj(x)  # [B, N, K]
        head_dim = self.head_dim
        h = h.view(B, N, self.num_heads, head_dim).permute(0, 2, 1, 3)  # [B, H, N, D]

        src = h.unsqueeze(3).expand(B, self.num_heads, N, N, head_dim)
        dst = h.unsqueeze(2).expand(B, self.num_heads, N, N, head_dim)
        pair = torch.cat([src, dst], dim=-1)  # [B, H, N, N, 2D]

        e = F.leaky_relu(
            (pair * attn_vec.view(1, self.num_heads, 1, 1, 2 * head_dim)).sum(dim=-1),
            negative_slope=0.2,
        )

        if adj_mask is not None:
            if adj_mask.dim() == 2:
                adj_mask = adj_mask.unsqueeze(0).expand(B, -1, -1)
            elif adj_mask.dim() != 3:
                raise ValueError(f"adj_mask must have shape [N, N] or [B, N, N], got {tuple(adj_mask.shape)}")

            if adj_mask.shape[0] != B or adj_mask.shape[1] != N or adj_mask.shape[2] != N:
                raise ValueError(f"adj_mask shape {tuple(adj_mask.shape)} incompatible with x shape {tuple(x.shape)}")

            eye = torch.eye(N, device=x.device, dtype=adj_mask.dtype).unsqueeze(0)
            adj_mask = (adj_mask + eye) > 0
            e = e.masked_fill(~adj_mask.unsqueeze(1), float('-inf'))

        alpha = F.softmax(e, dim=-1)  # normalize over neighbor j
        out = torch.matmul(alpha, h)  # [B, H, N, D]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, self.K)  # concat heads -> K
        return out

    def _format_input(self, x):
        """Support both [B, N, K] and legacy flattened [B, N*K] inputs."""
        if x.dim() == 3:
            if x.shape[-1] != self.K:
                raise ValueError(f"Expected feature dim K={self.K}, got {x.shape[-1]}.")
            return x

        if x.dim() == 2:
            if x.shape[1] % self.K != 0:
                raise ValueError(
                    f"Flattened input width ({x.shape[1]}) must be divisible by K ({self.K})."
                )
            return x.reshape(x.shape[0], x.shape[1] // self.K, self.K)

        raise ValueError(f"Unsupported input shape {tuple(x.shape)}. Expected [B,N,K] or [B,N*K].")

    def forward(self, x, adj_mask=None):
        """Run a two-layer GAT and produce relay-action Q-values.

        Args:
            x: [B, N, K] preferred; legacy [B, N*K] is also supported.
            adj_mask: Optional adjacency mask [N,N] or [B,N,N].
        """
        x = self._format_input(x)

        mean = x.mean(dim=(0, 1), keepdim=True)
        std = x.std(dim=(0, 1), keepdim=True) + 1e-8
        x = (x - mean) / std

        h = F.elu(self._gat_layer(x, self.gat1_fc, self.gat1_attn, adj_mask=adj_mask))
        h = self._gat_layer(h, self.gat2_fc, self.gat2_attn, adj_mask=adj_mask)

        self_node = h[:, 0, :]  # coding node
        nbr_nodes = h[:, 1:, :]

        self_out = F.relu(self.self_proj(self_node))
        if nbr_nodes.size(1) > 0:
            nbr_mean = nbr_nodes.mean(dim=1)
        else:
            nbr_mean = torch.zeros_like(self_node)
        path_agg = F.relu(self.path_norm(self.path_proj(nbr_mean)))

        merged = torch.cat([self_out, path_agg], dim=1)
        h_merge = F.relu(self.merge_fc1(merged))
        h_merge = F.relu(self.merge_fc2(h_merge))
        out = self.merge_fc3(h_merge)
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

    def forward(self, X, adj_mask=None):
        """Return relay-action Q-values for the current feature tensor."""
        q_values = self.q_network(X, adj_mask=adj_mask)
        return q_values

    def get_target_q_values(self, X, adj_mask=None):
        """Return Q-values computed by the frozen target network."""
        with torch.no_grad():
            q_values = self.target_q_network(X, adj_mask=adj_mask)
        return q_values

    def update_target_network(self):
        """Copy online-network parameters into the target network."""
        self.target_q_network.load_state_dict(self.q_network.state_dict())

    def select_action(self, X, epsilon=0.1, adj_mask=None):
        """Select a relay action using an epsilon-greedy policy."""
        q_values = self.forward(X, adj_mask=adj_mask)
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
