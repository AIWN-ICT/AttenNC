import random

import numpy as np
import simpy

try:
    from .layer_node import LayerNode
    from .pdu import AttenPDU
except ImportError:
    from layer_node import LayerNode
    from pdu import AttenPDU

_tdma_random = random.Random()


def set_tdma_seed(seed):
    """Set the random seed used by TDMA packet-loss decisions."""
    _tdma_random.seed(seed)


class TDMASimulator:
    """A lightweight SimPy simulator with fixed TDMA slots.

    The simulator is intentionally small so it can be embedded in the existing
    synchronous coding loop. Each node owns one fixed slot in a frame. Packets
    queued at a MAC are transmitted only when the environment reaches that
    node's slot. The physical layer still uses the AttenNC_PFC ``links`` matrix
    as the packet-delivery probability.

    Timing defaults are aligned with FQF (microsecond-scale):
    - ``slot_duration`` in us (default 3000)
    - ``bit_transmission_time`` in us/bit
    - ``bit_transport_time`` in us/m
    - ``propagation_guard_time`` in us (default 1)
    """

    def __init__(
        self,
        node_count,
        neighbor_matrix,
        links,
        slot_duration=3000.0,
        frame_slot=None,
        seed=None,
        node_positions=None,
        bit_transmission_time=1.0 / (4e6),
        bit_transport_time=1e6 / 3e8,
        phy_header_length=32,
        mac_header_length=32,
        propagation_guard_time=1.0,
        start_time=0.0,
    ):
        """Initialize simulator timing, topology, and per-node MAC/PHY objects."""
        self.env = simpy.Environment(initial_time=start_time)
        self.start_time = start_time
        self.node_count = node_count
        self.neighbor_matrix = neighbor_matrix
        self.links = links
        self.slot_duration = slot_duration
        self.frame_slot = self._build_frame_slot(node_count, frame_slot)
        self.active_slots = sorted({slot for slots in self.frame_slot.values() for slot in slots if slot > 0})
        self.frame_duration = max(self.active_slots, default=max(1, node_count)) * slot_duration
        self.random = random.Random(seed) if seed is not None else _tdma_random
        self.node_positions = node_positions or {}
        if not self.node_positions:
            raise ValueError("node_positions must be provided for distance-based propagation delay calculation.")
        self.bit_transmission_time = bit_transmission_time
        self.bit_transport_time = bit_transport_time
        self.phy_header_length = phy_header_length
        self.mac_header_length = mac_header_length
        self.propagation_guard_time = propagation_guard_time
        self.nodes = [LayerNode(self, node_id) for node_id in range(node_count)]
        self.deliveries = []
        self.losses = []

    @staticmethod
    def _build_frame_slot(node_count, frame_slot):
        """Normalize the frame-slot mapping to lists of positive slot indices."""
        if frame_slot is None:
            return {node_id: [node_id + 1] for node_id in range(node_count)}

        normalized = {}
        for node_id in range(node_count):
            slots = frame_slot.get(node_id, 0)
            if isinstance(slots, (list, tuple, set)):
                normalized[node_id] = sorted(int(slot) for slot in slots)
            else:
                normalized[node_id] = [int(slots)]
        return normalized

    def slot_start_delay(self, node_id):
        """Return the time until the next active TDMA slot for one node."""
        slot_indices = [slot for slot in self.frame_slot.get(node_id, []) if slot > 0]
        if not slot_indices:
            return None

        now_in_frame = self.env.now % self.frame_duration
        delays = []
        for slot_index in slot_indices:
            slot_offset = (slot_index - 1) * self.slot_duration
            delay = slot_offset - now_in_frame
            if delay < 0:
                delay += self.frame_duration
            delays.append(delay)
        return min(delays)

    def queue_pdu(self, src, dst, payload, packet_nbits=None):
        """Create a PDU and enqueue it on the source node MAC."""
        pdu = AttenPDU(src, dst, payload, packet_nbits)
        self.nodes[src].mac.enqueue(pdu)

    def distance(self, src, dst):
        """Return Euclidean distance between two nodes for propagation delay."""
        src_pos = self.node_positions.get(src)
        dst_pos = self.node_positions.get(dst)
        if src_pos is None or dst_pos is None:
            raise ValueError(
                f"Missing node_positions for propagation delay: src={src} pos={src_pos}, dst={dst_pos}"
            )
        src_pos = np.asarray(src_pos, dtype=float)
        dst_pos = np.asarray(dst_pos, dtype=float)
        return float(np.linalg.norm(src_pos - dst_pos))

    def run_until_idle(self):
        """Advance the simulation until all node MAC queues become empty."""
        while any(node.mac.has_pending() for node in self.nodes):
            next_times = []
            for node in self.nodes:
                if node.mac.has_pending():
                    delay = self.slot_start_delay(node.id)
                    if delay is not None:
                        next_times.append(self.env.now + delay)
            if not next_times:
                break
            next_time = min(next_times)
            if next_time > self.env.now:
                self.env.run(until=next_time)

            # FQF-style TDMA: every node whose active slot is the current slot may
            # transmit one queued packet in the same slot. This avoids converting
            # TDMA into a node-id serial loop when multiple nodes share a slot.
            current_slot_nodes = []
            for node in self.nodes:
                delay = self.slot_start_delay(node.id)
                if node.mac.has_pending() and delay is not None and abs(delay) < 1e-12:
                    current_slot_nodes.append(node)
            max_tx_finish_time = self.env.now
            for node in current_slot_nodes:
                tx_finish_time = node.mac.transmit_one()
                if tx_finish_time is not None:
                    max_tx_finish_time = max(max_tx_finish_time, tx_finish_time)
            self.env.run(until=max(self.env.now + self.slot_duration, max_tx_finish_time))
