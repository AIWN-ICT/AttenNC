"""Reward helpers for destination-rank gains and GNN-assisted decisions."""


def calculate_reward(K, rank_add_des, rank_des):
    """Calculate reward for a coded packet based on destination rank increase."""
    previous_rank = rank_des - rank_add_des
    term2 = rank_add_des * (K - previous_rank) / K
    return term2


def calculate_reward_GNN(K, rank_add_des, rank_des, GNN_action):
    """Calculate reward for a GNN-assisted coding decision."""
    previous_rank = rank_des - rank_add_des
    term2 = (rank_add_des - 0.5 * GNN_action) * (K - previous_rank) / K
    return term2
