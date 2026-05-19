from collections import deque


class NodeEfficiencyStats:
    """Track coding efficiency and GNN decision savings for one relay node."""

    def __init__(self):
        """Initialize rolling counters used by relay-side efficiency heuristics."""
        self.coding_counts = {i: 0 for i in range(20)}
        self.coding_success = {i: 0 for i in range(20)}
        self.gnn_decision_counts = {i: 0 for i in range(20)}
        self.gnn_skip_encoding = {i: 0 for i in range(20)}
        self.optimal_L = 3
        self.update_window = deque(maxlen=50)
        self.gnn_window = deque(maxlen=50)

    def update_stats(self, L, success):
        """Update coding statistics for code length L."""
        self.coding_counts[L] = self.coding_counts.get(L, 0) + 1
        if success:
            self.coding_success[L] = self.coding_success.get(L, 0) + 1
        self.update_window.append((L, success))

    def update_gnn_stats(self, L, decision, actual_encoded_count):
        """Update GNN decision statistics.

        L: current number of coded packets.
        decision: GNN decision, 0 means skip encoding, 1 means encode.
        actual_encoded_count is kept for API compatibility with the original code.
        """
        self.gnn_decision_counts[L] = self.gnn_decision_counts.get(L, 0) + 1
        if decision == 0:
            self.gnn_skip_encoding[L] = self.gnn_skip_encoding.get(L, 0) + L
            self.gnn_window.append((L, L))
        else:
            self.gnn_window.append((L, 0))

    def get_efficiency_ratio(self, L):
        """Return coding success ratio for a specific L."""
        count = self.coding_counts.get(L, 0)
        if count == 0:
            return 0
        return self.coding_success.get(L, 0) / count

    def update_optimal_L(self):
        """Update the optimal GNN-use threshold based on overhead balance."""
        if len(self.update_window) < 50:
            return

        cost_efficiency = {}
        for L in range(2, 6):
            L_gnn_records = [saved for l, saved in self.gnn_window if l == L]
            if not L_gnn_records:
                continue

            avg_saved_encodings = sum(L_gnn_records) / len(L_gnn_records)
            gnn_cost = 1
            potential_savings = avg_saved_encodings
            net_benefit = potential_savings - gnn_cost
            if net_benefit > 0:
                cost_efficiency[L] = net_benefit

        if cost_efficiency:
            self.optimal_L = min(cost_efficiency.keys())

    def should_use_gnn(self, L):
        """Return whether GNN should be used at code length L."""
        return L >= self.optimal_L
