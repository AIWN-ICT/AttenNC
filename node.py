import copy
from dataclasses import dataclass, field

import numpy as np
import torch

try:
    from .coding_buffers import get_local_neighbor_receivememory
    from .topology_utils import getneighborNum
except ImportError:
    from coding_buffers import get_local_neighbor_receivememory
    from topology_utils import getneighborNum


@dataclass
class LinkFeedbackPacket:
    """Explicit link-layer ACK/nACK control packet.

    The simulation uses this object to make link feedback explicit while keeping
    data-packet forwarding behavior unchanged. ACK packets refresh the sender's
    local next-hop buffer view; NACK packets record loss and intentionally leave
    cached views unchanged.
    """

    feedback_type: str
    packet_id: str
    source_id: int
    destination_id: int
    generation_id: int
    sender_id: int
    receiver_id: int
    previous_hop: int
    current_hop: int
    success: bool
    reason: str = ""
    create_time: float = 0.0
    reward_bonus: float = 0.0
    reverse_path: list = field(default_factory=list)

    @property
    def is_ack(self):
        """Return whether the control packet is an ACK."""
        return self.feedback_type.upper() == "ACK"

    @property
    def is_nack(self):
        """Return whether the control packet is a NACK."""
        return self.feedback_type.upper() == "NACK"

    @property
    def is_decode_ack(self):
        """Return whether the control packet confirms generation decoding."""
        return self.feedback_type.upper() == "DECODE_ACK"

    @property
    def is_reward_ack(self):
        """Return whether the control packet carries backward reward credit."""
        return self.feedback_type.upper() == "REWARD_ACK"


@dataclass
class NetworkCodedPacket:
    """A simulation packet that keeps network-coding metadata with the payload.

    coefficients stores the coding vector over GF(2). It is the object that is
    transmitted, buffered, re-coded by relays, and finally used by the receiver
    for rank/decode decisions.
    """

    packet_id: str
    source_id: int
    destination_id: int
    generation_id: int
    coefficients: np.ndarray
    payload: np.ndarray = None
    previous_hop: int = None
    current_hop: int = None
    hop_count: int = 0
    path: list = field(default_factory=list)
    create_time: float = 0.0

    def __post_init__(self):
        """Normalize array fields and initialize the traversed path."""
        self.coefficients = np.array(self.coefficients, dtype=np.float32).copy()
        if self.payload is None:
            self.payload = self.coefficients.copy()
        else:
            self.payload = np.array(self.payload, dtype=np.float32).copy()
        if self.current_hop is not None and not self.path:
            self.path.append(self.current_hop)

    @property
    def data(self):
        """Return the coding coefficients as the packet's effective data payload."""
        return self.coefficients

    def clone_for_forward(self, from_node, to_node):
        """Clone the packet while updating hop metadata for forwarding."""
        packet = copy.deepcopy(self)
        packet.previous_hop = from_node
        packet.current_hop = to_node
        packet.hop_count += 1
        packet.path.append(to_node)
        return packet

    def recode_with(self, coefficients, node_id):
        """Clone the packet and replace its coding coefficients at one relay."""
        packet = copy.deepcopy(self)
        packet.coefficients = np.array(coefficients, dtype=np.float32).copy()
        packet.payload = packet.coefficients.copy()
        packet.current_hop = node_id
        return packet


class NODE:
    """Node-local state container plus small protocol helpers.

    The class still behaves like the original simulation node, but now groups
    the state and operations that naturally belong to an individual distributed
    node. This makes it easier to later replace the central simulator with a
    real per-node runtime without changing the numerical behavior of the current
    experiment.
    """

    def __init__(self, K, R, id):
        """Initialize per-node buffers, replay state, and local statistics."""
        self.id = id
        self.K = K
        self.R = R
        self.receivememory = np.zeros((R, K))
        self.neighbor_receivememory_views = {}
        self.neighbor_receivelen_views = {}
        self.codelen = 0
        self.receivelen = 0
        self.datamemory = []  # Legacy aggregate buffer; destination rank should use generation_datamemory.
        self.generation_datamemory = {}
        self.sendmemory = []
        self.codememory = np.zeros((R, K))
        self.receive_flag = False
        self.list_action = np.zeros(K)
        self.list_state = []
        self.list_n_state = []
        self.list_rewards = np.zeros(K)
        self.list_len = 0
        self.packet = []
        self.feedback_packet = []
        self.sent_feedback_history = []
        self.received_feedback_history = []
        self.list_data_graph = []
        self.list_action_obj = []
        self.list_next_data_graph = []
        self.list_reward_obj = []
        self.gnn_list_len = 0  # Number of GNN-related experience records.
        self.local_stats = {
            'code_packets': 0,
            'code_rank': 0,
            'edge_loss_count': {},
            'edge_rank_increase_count': {},
            'edge_history': [],
        }

    def build_source_state(self, nodelist, neighbor_matrix, source_id, S_state_size, K, R, *args):
        """Build the source-node DQN state and return its current neighbors."""
        state = np.zeros(S_state_size)
        k_data = np.zeros(K)
        pre_data = np.zeros(K)
        k_data[0] = 1
        state[0:K] = k_data
        state[K:2 * K] = pre_data
        neighbors = getneighborNum(nodelist, neighbor_matrix, source_id)
        for l in range(len(neighbors)):
            for i in range(R):
                state[2 * K + l * R * K + i * K:2 * K + l * R * K + i * K + K] = \
                    get_local_neighbor_receivememory(nodelist, neighbor_matrix, source_id, neighbors[l], R, K)[i]
        return state, neighbors

    def build_relay_state(self, nodelist, neighbor_matrix, node_id, last_data, R_state_size, K, R, *args):
        """Build the relay-node DQN state and return its current neighbors."""
        state = np.zeros(R_state_size)
        state[0:K] = last_data
        state[K:2 * K] = nodelist[node_id].codememory[0]
        neighbors = getneighborNum(nodelist, neighbor_matrix, node_id)
        for m in range(len(neighbors)):
            for j in range(R):
                state[2 * K + m * R * K + j * K:2 * K + m * R * K + j * K + K] = \
                    get_local_neighbor_receivememory(nodelist, neighbor_matrix, node_id, neighbors[m], R, K)[j]
        return state, neighbors

    def build_relay_features(self, nodelist, neighbor_matrix, node_id, max_nb, K):
        """Build compact relay features and local adjacency for the GNN policy.

        Returns:
            combined_features: np.ndarray, shape [1, N, K], N = 1 + len(active_neighbors)
            adj_mask: np.ndarray, shape [1, N, N], local adjacency over [self, neighbors...]
            active_neighbors: np.ndarray, selected next-hop node ids
        """
        next_hop_nodes = np.where(neighbor_matrix[node_id] == 1)[0]
        max_neighbor_slots = max_nb
        active_neighbors = next_hop_nodes[:max_neighbor_slots]
        num_nodes = 1 + len(active_neighbors)

        if (
            not hasattr(self, 'feature_buffer')
            or self.feature_buffer.shape[1] != num_nodes
            or self.feature_buffer.shape[2] != K
        ):
            self.feature_buffer = np.zeros((1, num_nodes, K), dtype=np.float32)

        combined_features = self.feature_buffer
        combined_features.fill(0.0)

        combined_features[0, 0, 0] = float(self.local_stats.get('code_packets', 0))
        combined_features[0, 0, 1] = float(self.local_stats.get('code_rank', 0))

        for idx, next_hop in enumerate(active_neighbors):
            row = idx + 1
            neighbor_stats = nodelist[next_hop].local_stats

            combined_features[0, row, 0] = float(neighbor_stats.get('code_packets', 0))
            combined_features[0, row, 1] = float(neighbor_stats.get('code_rank', 0))

            if K > 2:
                combined_features[0, row, 2] = float(self.local_stats['edge_loss_count'].get(next_hop, 0))
            if K > 3:
                combined_features[0, row, 3] = float(self.local_stats['edge_rank_increase_count'].get(next_hop, 0))

        if not hasattr(self, 'adj_buffer') or self.adj_buffer.shape[1] != num_nodes:
            self.adj_buffer = np.zeros((1, num_nodes, num_nodes), dtype=np.float32)
        adj_mask = self.adj_buffer
        adj_mask.fill(0.0)

        index_to_node = np.concatenate(([node_id], active_neighbors))
        for i, src in enumerate(index_to_node):
            for j, dst in enumerate(index_to_node):
                if src == dst or neighbor_matrix[src, dst] == 1:
                    adj_mask[0, i, j] = 1.0

        return combined_features, adj_mask, active_neighbors

    def prepare_relay_tensor(self, combined_features, device):
        """Move relay features into a reusable tensor on the target device."""
        return torch.from_numpy(combined_features).to(device)

    def prepare_adj_tensor(self, adj_mask, device):
        """Move adjacency mask into a tensor on the target device."""
        return torch.from_numpy(adj_mask).to(device)

    def append_local_transition(self, state, next_state, action):
        """Store one local transition snapshot for later learning updates."""
        self.list_state.append(copy.deepcopy(state))
        self.list_n_state.append(copy.deepcopy(next_state))
        self.list_action_obj.append(action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.float))

    def step_source(self, nodelist, neighbor_matrix, source_id, S_state_size, K, R, agent_s, act, getneighborNum, get_local_neighbor_receivememory):
        """Roll out one source-node decision sequence for the current generation."""
        test_node = nodelist[source_id]
        test_node.list_action = np.zeros(K)
        state, _ = self.build_source_state(nodelist, neighbor_matrix, source_id, S_state_size, K, R, getneighborNum, get_local_neighbor_receivememory)
        for t in range(K):
            xstate = np.resize(state, [1, S_state_size])
            with torch.no_grad():
                q = agent_s.get_q_val_with_cache(xstate)
            action = act(q)
            test_node.list_action[t] = action
            next_state = np.zeros(S_state_size)
            if t < K - 1:
                k_data = np.zeros(K)
                pre_data = np.zeros(K)
                k_data[t + 1] = 1
                pre_data[t] = action
                next_state[0:K] = k_data
                next_state[K:2 * K] = pre_data
                neighbors = getneighborNum(nodelist, neighbor_matrix, source_id)
                for l in range(len(neighbors)):
                    for i in range(R):
                        next_state[2 * K + l * R * K + i * K:2 * K + l * R * K + i * K + K] = \
                            get_local_neighbor_receivememory(nodelist, neighbor_matrix, source_id, neighbors[l], R, K)[i]
            self.append_local_transition(state, next_state, action)
            state = next_state
        return test_node.list_action

    def step_relay(self, nodelist, node_id, neighbor_matrix, last_data, R_state_size, K, R, model, device, act, p_xor, statistics_node, statistics_edge, edge_statistics_history, max_nb, node_efficiency_stats, action_stats, gnn_decision_stats):
        """Roll out one relay-node coding sequence and optional GNN gating decision."""
        test_node = nodelist[node_id]
        test_node.list_action = np.zeros(K)
        combined_features, _ = self.build_relay_features(nodelist, neighbor_matrix, node_id, max_nb)
        l = test_node.codelen
        L = np.minimum(l, R)
        if not node_efficiency_stats[node_id].should_use_gnn(L):
            action = 1
            action_val = 1
        else:
            input_features = self.prepare_relay_tensor(combined_features, device)
            action = model.select_action(input_features)
            action_val = action.item() if isinstance(action, torch.Tensor) else action
            action_stats[node_id]['gnn_count'] += 1
            if action_val == 1:
                gnn_decision_stats['output_1_count'] += 1
            else:
                gnn_decision_stats['output_0_count'] += 1
        state, _ = self.build_relay_state(nodelist, neighbor_matrix, node_id, last_data, R_state_size, K, R, getneighborNum, get_local_neighbor_receivememory)
        r_data = np.array(last_data, dtype=np.float32).copy()
        actual_encoded_count = 0
        for t in range(L):
            xstate = np.resize(state, [1, R_state_size])
            with torch.no_grad():
                q = model.get_q_val(xstate)
            action = act(q)
            test_node.list_action[t] = action
            actual_encoded_count += 1
            if action == 1:
                r_data = p_xor(r_data, test_node.codememory[t])
            next_state = np.zeros(R_state_size)
            if t < R - 1:
                next_state[0:K] = r_data
                next_state[K:2 * K] = test_node.codememory[t + 1]
            self.append_local_transition(state, next_state, action)
            state = next_state
        return action_val if 'action_val' in locals() else 1, actual_encoded_count, r_data, combined_features

    def reset_episode_state(self):
        """Reset per-episode buffers and statistics for a fresh simulation run."""
        self.receivememory.fill(0)
        self.neighbor_receivememory_views.clear()
        self.neighbor_receivelen_views.clear()
        self.codelen = 0
        self.receivelen = 0
        self.datamemory = []
        self.generation_datamemory = {}
        self.sendmemory = []
        self.codememory.fill(0)
        self.receive_flag = False
        self.list_action = np.zeros(self.K)
        self.list_state = []
        self.list_n_state = []
        self.list_rewards = np.zeros(self.K)
        self.list_len = 0
        self.packet = []
        self.feedback_packet = []
        self.sent_feedback_history = []
        self.received_feedback_history = []
        self.list_data_graph = []
        self.list_action_obj = []
        self.list_next_data_graph = []
        self.list_reward_obj = []
        self.gnn_list_len = 0
        self.local_stats = {
            'code_packets': 0,
            'code_rank': 0,
            'edge_loss_count': {},
            'edge_rank_increase_count': {},
        }

    def enqueue_packet(self, packet):
        """Append an incoming packet to the local receive queue."""
        self.packet.append(packet)
        self.receive_flag = True

    def has_pending_packet(self):
        """Return whether the node still has queued packets to process."""
        return len(self.packet) > 0

    def add_feedback_packet(self, feedback_packet):
        """Store one received feedback packet and append it to history."""
        self.feedback_packet.append(feedback_packet)
        self.received_feedback_history.append(feedback_packet)

    def send_feedback_packet(self, feedback_packet):
        """Record that one feedback packet has been emitted by this node."""
        self.sent_feedback_history.append(feedback_packet)

    def get_feedback_packet(self):
        """Pop the oldest queued feedback packet, if any."""
        if not self.feedback_packet:
            return None
        return self.feedback_packet.pop(0)

    def get_generation_buffer(self, generation_id):
        """Return the per-generation local coding buffer."""
        return self.generation_datamemory.setdefault(generation_id, [])

    def update_neighbor_view(self, neighbor_id, receivememory, receivelen):
        """Update the cached local estimate of a neighbor's receive state."""
        self.neighbor_receivememory_views[neighbor_id] = np.array(receivememory, dtype=np.float32).copy()
        self.neighbor_receivelen_views[neighbor_id] = int(receivelen)

    def clear_neighbor_view(self, neighbor_id):
        """Remove the cached receive-state estimate for one neighbor."""
        self.neighbor_receivememory_views.pop(neighbor_id, None)
        self.neighbor_receivelen_views.pop(neighbor_id, None)

    def getpacket(self):
        """Pop one queued packet and normalize it to the coded-packet format."""
        packet = self.packet.pop(0)
        if isinstance(packet, NetworkCodedPacket):
            return packet.data.copy(), packet
        if isinstance(packet, dict) and 'data' in packet:
            packet_id = packet.get('packet_id')
            source_id = packet.get('source_id', -1)
            destination_id = packet.get('destination_id', -1)
            generation_id = packet.get('generation_id', 0)
            coded_packet = NetworkCodedPacket(
                packet_id=packet_id,
                source_id=source_id,
                destination_id=destination_id,
                generation_id=generation_id,
                coefficients=packet['data'],
                payload=packet.get('payload'),
                previous_hop=packet.get('previous_hop'),
                current_hop=packet.get('current_hop'),
                hop_count=packet.get('hop_count', 0),
                path=packet.get('path', []),
                create_time=packet.get('create_time', 0.0),
            )
            return coded_packet.data.copy(), coded_packet
        return np.array(packet, dtype=np.float32).copy(), None
