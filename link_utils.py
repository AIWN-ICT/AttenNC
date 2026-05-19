"""Low-level propagation and transmission-delay helpers."""

import numpy as np


def calculate_link_delay_us(
    src,
    dst,
    nbits,
    node_positions,
    bit_transmission_time,
    bit_transport_time,
    propagation_guard_time,
    mac_header_length=0,
    phy_header_length=0,
):
    """Return one-hop data/control frame delay in simulation microseconds."""
    if node_positions is None:
        raise ValueError("node_positions is required for propagation-delay calculation.")
    src_pos = np.asarray(node_positions[src], dtype=float)
    dst_pos = np.asarray(node_positions[dst], dtype=float)
    distance = float(np.linalg.norm(src_pos - dst_pos))
    tx_time = (nbits + mac_header_length + phy_header_length) * bit_transmission_time
    return distance * bit_transport_time + propagation_guard_time + tx_time
