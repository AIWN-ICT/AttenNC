import numpy as np


class AttenPDU:
    """Minimal PDU used by the AttenNC_PFC MAC/PHY simulation."""

    def __init__(self, src, dst, payload, nbits=None):
        """Store addressing, payload, and timing fields for one transmission."""
        self.src = src
        self.dst = dst
        self.payload = np.array(payload, copy=True)
        self.nbits = int(nbits) if nbits is not None else int(self.payload.size)
        self.tx_start_time = None
        self.rx_start_time = None
        self.rx_end_time = None
