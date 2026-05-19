"""Compatibility exports for the AttenNC_PFC slot-based MAC/PHY simulation.

The implementation is split into focused modules:
- ``pdu.py``: PDU data structure
- ``mac_layer.py``: slot-based MAC queueing and receive handoff
- ``phy_layer.py``: PHY timing, propagation delay and packet loss
- ``layer_node.py``: node container binding MAC and PHY layers
- ``slot_simulator.py``: slot scheduler and simulation environment

Keep this module as the public entry point for existing code that imports from
``network_layers``.
"""

try:
    from .layer_node import LayerNode
    from .mac_layer import AttenMacLayer
    from .pdu import AttenPDU
    from .phy_layer import AttenPhyLayer
    from .slot_simulator import TDMASimulator, set_tdma_seed
except ImportError:
    from layer_node import LayerNode
    from mac_layer import AttenMacLayer
    from pdu import AttenPDU
    from phy_layer import AttenPhyLayer
    from slot_simulator import TDMASimulator, set_tdma_seed


def transmit_tdma(
    node_count,
    neighbor_matrix,
    links,
    src,
    payload,
    slot_duration=3000.0,
    frame_slot=None,
    seed=None,
    node_positions=None,
    packet_nbits=None,
    bit_transmission_time=1.0 / (4e6),
    bit_transport_time=1e6 / 3e8,
    phy_header_length=32,
    mac_header_length=32,
    propagation_guard_time=1.0,
    start_time=0.0,
):
    """Broadcast a payload from ``src`` to its next-hop neighbors via TDMA.

    ``frame_slot`` manually maps node_id -> TDMA slot index, matching FQF's
    FRAME_SLOT style. Slot index 0 means the node has no sending slot.
    Returns delivered destinations, lost destinations and the SimPy time spent
    by this hop. Packet loss is decided exclusively by ``links[src][dst]``.
    """

    sim = TDMASimulator(
        node_count=node_count,
        neighbor_matrix=neighbor_matrix,
        links=links,
        slot_duration=slot_duration,
        frame_slot=frame_slot,
        seed=seed,
        node_positions=node_positions,
        bit_transmission_time=bit_transmission_time,
        bit_transport_time=bit_transport_time,
        phy_header_length=phy_header_length,
        mac_header_length=mac_header_length,
        propagation_guard_time=propagation_guard_time,
        start_time=start_time,
    )
    for dst in range(src + 1, node_count):
        if neighbor_matrix[src][dst] == 1:
            sim.queue_pdu(src, dst, payload, packet_nbits)
    sim.run_until_idle()
    delivered = {dst for _, dst, _, _ in sim.deliveries}
    lost = {dst for _, dst, _, _ in sim.losses}
    hop_delay = sim.env.now - sim.start_time
    delivery_times = {dst: rx_end_time - sim.start_time for _, dst, _, rx_end_time in sim.deliveries}
    loss_times = {dst: rx_end_time - sim.start_time for _, dst, _, rx_end_time in sim.losses}
    return delivered, lost, hop_delay, delivery_times, loss_times


__all__ = [
    "AttenPDU",
    "set_tdma_seed",
    "TDMASimulator",
    "LayerNode",
    "AttenMacLayer",
    "AttenPhyLayer",
    "transmit_tdma",
]
