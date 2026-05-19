"""Relay-side neural network used by the DQN agent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time


class Network_R(nn.Module):
    """Attention-based Q-network for relay coding decisions."""

    def __init__(self, state_dim, arg_dict, device):
        """Initialize attention blocks and the path-feature cache."""
        super(Network_R, self).__init__()
        self.action_dim = arg_dict['action_size']
        self.K = arg_dict['K']
        self.M = arg_dict['M']
        self.device = device
        self.parallel_path = arg_dict['max_nb']

        # MLP sizes
        num_units = arg_dict['fc1_features']
        num_path_units = arg_dict['fc1_path_features']
        num_test = num_units // 2
        self.num_test = num_test

        # Self MLP: operate on the first 2 rows (sliced_x)
        self.self_fc1 = nn.Linear(2 * self.K, num_units)
        self.self_fc2 = nn.Linear(num_units, num_test)

        # Shared path MLP (to support dynamic parallel_path)
        in_feat = self.M * self.K
        self.shared_path_fc1 = nn.Linear(in_feat, num_path_units)
        self.shared_path_fc2 = nn.Linear(num_path_units, num_test)
        self.path_norm = nn.LayerNorm(num_test)

        # Merge
        merge_in = num_test * 2
        self.merge_fc1 = nn.Linear(merge_in, num_units)
        self.merge_fc2 = nn.Linear(num_units, num_units)
        self.merge_fc3 = nn.Linear(num_units, self.action_dim)

        # Cache path features during repeated single-sample inference.
        self.PFC = {
            'path_part': None,  # Cached path feature tensor.
            'stack': None,      # Cached final stacked path embedding.
        }

    def forward(self, x):
        """Run the attention-based forward pass for a batch of relay states."""
        # x: [B, M*nb + 2, K]
        x = x.reshape(-1, self.M * self.parallel_path + 2, self.K)
        B = x.size(0)

        self_part = x[:, :2, :].reshape(B, -1)         # [B, 2*K]
        path_part = x[:, 2:, :]                        # [B, R*nb, K]

        # Self part
        h1 = F.relu(self.self_fc1(self_part))          # [B, num_units]
        self_out = F.relu(self.self_fc2(h1))           # [B, num_test]

        # Path segments MLPs with optional cache reuse.
        # Reuse the cache only for single-sample inference when path features match.
        if (B == 1 and
            not self.training and
            self.PFC['path_part'] is not None and
            torch.equal(path_part, self.PFC['path_part'])):
            stack = self.PFC['stack']
        else:
            path_outs = []
            nb = path_part.size(1) // self.M
            for i in range(nb):
                seg = path_part[:, i*self.M:(i+1)*self.M, :].reshape(B, -1)  # [B, M*K]
                h_seg = F.relu(self.shared_path_fc1(seg))                    # [B, num_units]
                seg_out = F.relu(self.shared_path_fc2(h_seg))                # [B, num_test]
                path_outs.append(seg_out)
            stack = torch.stack(path_outs, dim=2)          # [B, num_test, nb]

            # Refresh the cache only in inference mode with batch size 1.
            if B == 1 and not self.training:
                self.PFC['path_part'] = path_part.detach().clone()
                self.PFC['stack'] = stack.detach().clone()

        # Attention
        attn = F.softmax(torch.matmul(self_out.unsqueeze(1), stack) / math.sqrt(self.num_test), dim=2)
        path_agg = torch.matmul(attn, stack.transpose(1, 2)).squeeze(1)  # [B, num_test]
        path_agg = F.relu(self.path_norm(path_agg))

        # Merge
        merged = torch.cat([self_out, path_agg], dim=1)  # [B, 2*num_test]
        h = F.relu(self.merge_fc1(merged))
        h = F.relu(self.merge_fc2(h))
        out = self.merge_fc3(h)
        return out

    def clear_cache(self):
        """Clear cached path embeddings after a repeated inference sequence."""
        self.PFC = {
            'path_part': None,
            'stack': None,
        }

if __name__ == "__main__":
    arg_dict = {
        'action_size': 2,
        'K': 5,
        'R': 3,
        'fc1_out_features': 256,
    }
    R = arg_dict['R']
    K = arg_dict['K']

    # test with dynamic max_nb
    for nb in [1, 2, 3]:
        arg_dict['max_nb'] = nb
        print(f"\n=== Testing with nb = {nb} ===")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = Network_R(state_dim=None, arg_dict=arg_dict, device=device).to(device)
        test_input = torch.randint(0, 2, (1, R * nb + 2, K)).float().to(device)
        print("Input shape:", test_input.shape)
        with torch.no_grad():
            start = time.time()
            out = model(test_input)
            end = time.time()
        print("Output shape:", out.shape)
        print(f"Forward time: {end-start:.6f}s")
