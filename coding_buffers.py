"""Helpers for per-generation coding buffers and neighbor-state caches."""

import time

import numpy as np

try:
    from .topology_utils import getneighborNum
except ImportError:
    from topology_utils import getneighborNum


def get_generation_buffer(node, generation_id):
    """Return the per-generation coding buffer for a node.

    The destination must evaluate decodability only within one network-coding
    generation. Keeping a global aggregate datamemory for compatibility is fine,
    but rank/decode decisions should use this per-generation buffer.
    """
    if not hasattr(node, 'generation_datamemory'):
        node.generation_datamemory = {}
    return node.generation_datamemory.setdefault(generation_id, [])


def generation_rank(node, generation_id):
    """Return the matrix rank of the packets stored for one generation."""
    return np.linalg.matrix_rank(get_generation_buffer(node, generation_id))


def aggregate_generation_packet_count(node):
    """Count packets across all generation-specific buffers for a node."""
    if hasattr(node, 'generation_datamemory') and node.generation_datamemory:
        return sum(len(buffer) for buffer in node.generation_datamemory.values())
    return len(node.datamemory)


def calculate_decode_compute_delay_us(K, decode_start_wall_time, use_measured_decode_delay, delay_coefficient_us, time_unit):
    """Return destination decoding computation delay in simulation microseconds."""
    if use_measured_decode_delay:
        return (time.time() - decode_start_wall_time) * time_unit
    return delay_coefficient_us * (K ** 3)


def ensure_neighbor_receivememory_views(nodelist, nb_M, R, K):
    """Ensure every node keeps a local ACK-based view of each next-hop neighbor buffer.

    neighbor_receivememory_views[u][v] is node u's local cached estimate of next-hop
    neighbor v's receivememory. The cache is updated only when v ACKs a successful
    reception from u; on NACK/loss the cache remains unchanged.
    """
    for node_id, node in enumerate(nodelist):
        if not hasattr(node, 'neighbor_receivememory_views'):
            node.neighbor_receivememory_views = {}
        if not hasattr(node, 'neighbor_receivelen_views'):
            node.neighbor_receivelen_views = {}
        for next_hop in getneighborNum(nodelist, nb_M, node_id):
            if next_hop not in node.neighbor_receivememory_views:
                node.neighbor_receivememory_views[next_hop] = np.zeros((R, K))
            if next_hop not in node.neighbor_receivelen_views:
                node.neighbor_receivelen_views[next_hop] = 0


def get_local_neighbor_receivememory(nodelist, nb_M, node_id, next_hop, R, K):
    """Return node_id's cached receive-memory view for the selected next hop."""
    ensure_neighbor_receivememory_views(nodelist, nb_M, R, K)
    return nodelist[node_id].neighbor_receivememory_views[next_hop]


def sync_local_neighbor_receivememory_on_ack(nodelist, nb_M, ack_node, R, K):
    """Refresh local neighbor-buffer views when ack_node ACKs a successful reception."""
    ensure_neighbor_receivememory_views(nodelist, nb_M, R, K)
    for observer_id, observer in enumerate(nodelist):
        if ack_node in getneighborNum(nodelist, nb_M, observer_id):
            observer.neighbor_receivememory_views[ack_node] = nodelist[ack_node].receivememory.copy()
            observer.neighbor_receivelen_views[ack_node] = nodelist[ack_node].receivelen
