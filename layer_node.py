try:
    from .mac_layer import AttenMacLayer
    from .phy_layer import AttenPhyLayer
except ImportError:
    from mac_layer import AttenMacLayer
    from phy_layer import AttenPhyLayer


class LayerNode:
    """Node container that binds one MAC layer and one PHY layer."""

    def __init__(self, sim, node_id):
        """Attach MAC/PHY helpers to one simulator-owned node wrapper."""
        self.sim = sim
        self.id = node_id
        self.mac = AttenMacLayer(self)
        self.phy = AttenPhyLayer(self)
