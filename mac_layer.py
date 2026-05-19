from collections import deque

import numpy as np


class AttenMacLayer:
    """TDMA MAC layer with a FIFO queue per node."""

    def __init__(self, node):
        """Initialize queue state and MAC-side traffic counters."""
        self.node = node
        self.queue = deque()
        self.total_tx = 0
        self.total_rx = 0

    def enqueue(self, pdu):
        """Append one outbound PDU to the node's transmit queue."""
        self.queue.append(pdu)

    def has_pending(self):
        """Return whether at least one PDU is waiting for transmission."""
        return len(self.queue) > 0

    def transmit_one(self):
        """Transmit the next queued PDU when the current TDMA slot is active."""
        if not self.queue:
            return None
        pdu = self.queue.popleft()
        self.total_tx += 1
        return self.node.phy.send_pdu(pdu)

    def on_receive_pdu(self, pdu):
        """Record one received PDU and hand it to the simulator delivery log."""
        self.total_rx += 1
        self.node.sim.deliveries.append((pdu.src, pdu.dst, np.array(pdu.payload, copy=True), pdu.rx_end_time))
