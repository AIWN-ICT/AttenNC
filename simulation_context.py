from __future__ import annotations

from dataclasses import dataclass, field

from link_events import LinkEventManager


@dataclass
class SimulationContext:
    """Mutable runtime state and pending-event queues for one simulation test."""

    simulated_network_time: float = 0.0
    packet_sequence: int = 0
    event_manager: LinkEventManager = field(default_factory=LinkEventManager)

    def __post_init__(self) -> None:
        """Initialize queue aliases after the event manager is created."""

        self.sync_aliases()

    def sync_aliases(self) -> None:
        """Expose event-manager queues through context-level convenience aliases."""

        self.pending_data_arrivals = self.event_manager.pending_data_arrivals
        self.pending_feedback_arrivals = self.event_manager.pending_feedback_arrivals
        self.pending_decode_ack_arrivals = self.event_manager.pending_decode_ack_arrivals
        self.pending_reward_ack_arrivals = self.event_manager.pending_reward_ack_arrivals

    def reset(self) -> None:
        """Reset timing state and clear all pending events for a new test run."""

        self.simulated_network_time = 0.0
        self.packet_sequence = 0
        self.event_manager.reset()
        self.sync_aliases()

    def flush_pending_events(self, until_time=None) -> None:
        """Advance the event manager and synchronize queue aliases afterward."""

        if until_time is not None:
            self.simulated_network_time = until_time
        self.event_manager.flush_pending_events(self.simulated_network_time, None, until_time)
        self.sync_aliases()
