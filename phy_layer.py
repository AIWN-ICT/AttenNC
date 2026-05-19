import numpy as np


class AttenPhyLayer:
    """Physical layer using AttenNC_PFC link quality for packet loss."""

    def __init__(self, node):
        """Initialize PHY-side counters for transmission, reception, and loss."""
        self.node = node
        self.total_tx = 0
        self.total_rx = 0
        self.total_error = 0

    def send_pdu(self, pdu):
        """Schedule one PDU transmission and return its expected receive-end time."""
        self.total_tx += 1
        tx_time = (
            pdu.nbits + self.node.sim.mac_header_length + self.node.sim.phy_header_length
        ) * self.node.sim.bit_transmission_time
        # Keep FQF-consistent propagation shape: dist * BIT_TRANSPORT_TIME + 1(unit).
        # Here the constant term is configurable by propagation_guard_time.
        prop_time = (
            self.node.sim.distance(pdu.src, pdu.dst) * self.node.sim.bit_transport_time
            + self.node.sim.propagation_guard_time
        )
        pdu.tx_start_time = self.node.sim.env.now
        pdu.rx_start_time = pdu.tx_start_time + prop_time
        pdu.rx_end_time = pdu.rx_start_time + tx_time
        dst_node = self.node.sim.nodes[pdu.dst]
        delivery_probability = float(self.node.sim.links[pdu.src][pdu.dst])
        if self.node.sim.random.random() <= delivery_probability:
            self.total_rx += 1
            self.node.sim.env.process(dst_node.phy.delayed_rx_end(pdu, prop_time + tx_time))
        else:
            self.total_error += 1
            self.node.sim.env.process(self.delayed_loss(pdu, prop_time + tx_time))
        return pdu.rx_end_time

    def delayed_rx_end(self, pdu, delay):
        """Deliver a received PDU after its propagation and transmission delay."""
        yield self.node.sim.env.timeout(delay)
        self.on_rx_end(pdu)

    def delayed_loss(self, pdu, delay):
        """Record a lost PDU after the same end-to-end delay budget expires."""
        yield self.node.sim.env.timeout(delay)
        self.node.sim.losses.append((pdu.src, pdu.dst, np.array(pdu.payload, copy=True), pdu.rx_end_time))

    def on_rx_end(self, pdu):
        """Notify the MAC layer that a PDU has been received successfully."""
        self.node.mac.on_receive_pdu(pdu)
