import numpy as np

try:
    from .coding_buffers import get_generation_buffer, sync_local_neighbor_receivememory_on_ack
    from .node import LinkFeedbackPacket
    from .topology_utils import getneighborNum
except ImportError:
    from coding_buffers import get_generation_buffer, sync_local_neighbor_receivememory_on_ack
    from node import LinkFeedbackPacket
    from topology_utils import getneighborNum


class LinkEventManager:
    """Manage delayed data, ACK/NACK, and decode-ACK events.

    This class replaces several module-level pending-event lists while keeping
    the original event payloads and application order unchanged.
    """

    def __init__(self):
        """Initialize per-event-type queues for delayed network events."""
        self.pending_data_arrivals = []
        self.pending_feedback_arrivals = []
        self.pending_decode_ack_arrivals = []
        self.pending_reward_ack_arrivals = []

    def reset(self):
        """Clear all pending event queues before a new simulation run."""
        self.pending_data_arrivals.clear()
        self.pending_feedback_arrivals.clear()
        self.pending_decode_ack_arrivals.clear()
        self.pending_reward_ack_arrivals.clear()

    def record_link_loss(self, nodelist, src, dst):
        """Record a failed transmission on the source-local link statistics."""
        src_stats = nodelist[src].local_stats
        if dst not in src_stats['edge_loss_count']:
            src_stats['edge_loss_count'][dst] = 0
        src_stats['edge_loss_count'][dst] += 1

    def apply_data_arrival(self, nodelist, neighbor_M, src, dst, tx_packet, tx_data, K, R):
        """Apply one delivered data packet to the receiver state and statistics."""
        nodelist[dst].receive_flag = True
        generation_buffer = get_generation_buffer(nodelist[dst], tx_packet.generation_id)
        rank1 = np.linalg.matrix_rank(generation_buffer)
        rece_data = np.zeros(K)
        for j in range(K):
            rece_data[j] = tx_data[j]
        generation_buffer.append(rece_data)
        nodelist[dst].datamemory.append(rece_data)
        rank2 = np.linalg.matrix_rank(generation_buffer)

        # Update decentralized local coding statistics on the receiving node.
        dst_stats = nodelist[dst].local_stats
        dst_stats['code_rank'] = int(rank2)
        dst_stats['code_packets'] = len(generation_buffer)

        rank_increased = rank2 > rank1
        src_stats = nodelist[src].local_stats
        if dst not in src_stats['edge_rank_increase_count']:
            src_stats['edge_rank_increase_count'][dst] = 0
        if rank_increased:
            src_stats['edge_rank_increase_count'][dst] += 1

        rx_packet = tx_packet.clone_for_forward(src, dst)
        nodelist[dst].enqueue_packet(rx_packet)

        l = nodelist[dst].receivelen
        nodelist[dst].receivememory[int(l % R)] = rece_data
        nodelist[dst].receivelen = nodelist[dst].receivelen + 1
        return rank_increased
    def apply_feedback_arrival(self, nodelist, nb_M, sender_id, feedback_packet, R, K):
        """Apply an ACK/NACK after its control frame reaches the original data sender."""
        nodelist[feedback_packet.sender_id].send_feedback_packet(feedback_packet)
        nodelist[sender_id].add_feedback_packet(feedback_packet)
        if feedback_packet.is_ack:
            sync_local_neighbor_receivememory_on_ack(nodelist, nb_M, feedback_packet.sender_id, R, K)

    def schedule_link_feedback(
        self,
        nodelist,
        nb_M,
        sender_id,
        feedback_packet,
        R,
        K,
        arrival_time,
        pending_feedback_arrivals=None,
    ):
        """Schedule explicit ACK/NACK control-frame delivery at arrival_time."""
        if pending_feedback_arrivals is None:
            self.apply_feedback_arrival(nodelist, nb_M, sender_id, feedback_packet, R, K)
            return
        pending_feedback_arrivals.append({
            'arrival_time': arrival_time,
            'nodelist': nodelist,
            'nb_M': nb_M,
            'sender_id': sender_id,
            'feedback_packet': feedback_packet,
            'R': R,
            'K': K,
        })

    def schedule_destination_decode_ack(
        self,
        nodelist,
        source_id,
        destination_id,
        decode_ack_packet,
        arrival_time,
        pending_decode_ack_arrivals=None,
    ):
        """Schedule an end-to-end generation decode ACK from destination to source."""
        if pending_decode_ack_arrivals is None:
            nodelist[destination_id].send_feedback_packet(decode_ack_packet)
            nodelist[source_id].add_feedback_packet(decode_ack_packet)
            return
        pending_decode_ack_arrivals.append({
            'arrival_time': arrival_time,
            'nodelist': nodelist,
            'source_id': source_id,
            'destination_id': destination_id,
            'decode_ack_packet': decode_ack_packet,
        })

    def schedule_reward_ack(
        self,
        nodelist,
        source_id,
        destination_id,
        reward_ack_packet,
        arrival_time,
        pending_reward_ack_arrivals=None,
    ):
        """Schedule a reward ACK that walks backward without consuming slots."""
        if pending_reward_ack_arrivals is None:
            self.apply_reward_ack_arrival(nodelist, source_id, destination_id, reward_ack_packet)
            return
        pending_reward_ack_arrivals.append({
            'arrival_time': arrival_time,
            'nodelist': nodelist,
            'source_id': source_id,
            'destination_id': destination_id,
            'reward_ack_packet': reward_ack_packet,
        })

    def apply_reward_ack_arrival(self, nodelist, source_id, destination_id, reward_ack_packet):
        """Propagate a reward ACK backward and credit every relay on the path."""
        path = list(reward_ack_packet.reverse_path)
        if not path:
            return
        if destination_id in path:
            path = path[path.index(destination_id):]
        if source_id not in path:
            path.append(source_id)

        reward_value = float(reward_ack_packet.reward_bonus)
        for node_id in path:
            if node_id == destination_id:
                continue
            node = nodelist[node_id]
            if node.list_len > 0:
                node.list_rewards[:node.list_len] += reward_value
            node.add_feedback_packet(reward_ack_packet)
        nodelist[destination_id].send_feedback_packet(reward_ack_packet)

    def flush_pending_events(self, simulated_network_time, statistics_node, until_time=None):
        """Commit data/control packets whose simulated arrival time has passed."""
        if until_time is None:
            until_time = simulated_network_time

        progressed = True
        while progressed:
            progressed = False
            ready_data = [
                event for event in self.pending_data_arrivals
                if event['arrival_time'] <= until_time + 1e-12
            ]
            if ready_data:
                progressed = True
                for event in sorted(ready_data, key=lambda item: item['arrival_time']):
                    self.apply_data_arrival(
                        event['nodelist'],
                        event['neighbor_M'],
                        event['src'],
                        event['dst'],
                        event['tx_packet'],
                        event['tx_data'],
                        event['K'],
                        event['R'],
                    )
                    event['ack_packet'].create_time = round(event['arrival_time'], 2)
                    self.schedule_link_feedback(
                        event['nodelist'],
                        event['neighbor_M'],
                        event['src'],
                        event['ack_packet'],
                        event['R'],
                        event['K'],
                        event['feedback_arrival_time'],
                        self.pending_feedback_arrivals,
                    )
                ready_ids = {id(event) for event in ready_data}
                self.pending_data_arrivals[:] = [
                    event for event in self.pending_data_arrivals if id(event) not in ready_ids
                ]

            ready_feedback = [
                event for event in self.pending_feedback_arrivals
                if event['arrival_time'] <= until_time + 1e-12
            ]
            if ready_feedback:
                progressed = True
                for event in sorted(ready_feedback, key=lambda item: item['arrival_time']):
                    self.apply_feedback_arrival(
                        event['nodelist'],
                        event['nb_M'],
                        event['sender_id'],
                        event['feedback_packet'],
                        event['R'],
                        event['K'],
                    )
                ready_ids = {id(event) for event in ready_feedback}
                self.pending_feedback_arrivals[:] = [
                    event for event in self.pending_feedback_arrivals if id(event) not in ready_ids
                ]

            ready_decode_ack = [
                event for event in self.pending_decode_ack_arrivals
                if event['arrival_time'] <= until_time + 1e-12
            ]
            if ready_decode_ack:
                progressed = True
                for event in sorted(ready_decode_ack, key=lambda item: item['arrival_time']):
                    event['decode_ack_packet'].create_time = round(event['arrival_time'], 2)
                    event['nodelist'][event['destination_id']].send_feedback_packet(event['decode_ack_packet'])
                    event['nodelist'][event['source_id']].add_feedback_packet(event['decode_ack_packet'])
                ready_ids = {id(event) for event in ready_decode_ack}
                self.pending_decode_ack_arrivals[:] = [
                    event for event in self.pending_decode_ack_arrivals if id(event) not in ready_ids
                ]

            ready_reward_ack = [
                event for event in self.pending_reward_ack_arrivals
                if event['arrival_time'] <= until_time + 1e-12
            ]
            if ready_reward_ack:
                progressed = True
                for event in sorted(ready_reward_ack, key=lambda item: item['arrival_time']):
                    event['reward_ack_packet'].create_time = round(event['arrival_time'], 2)
                    self.apply_reward_ack_arrival(
                        event['nodelist'],
                        event['source_id'],
                        event['destination_id'],
                        event['reward_ack_packet'],
                    )
                ready_ids = {id(event) for event in ready_reward_ack}
                self.pending_reward_ack_arrivals[:] = [
                    event for event in self.pending_reward_ack_arrivals if id(event) not in ready_ids
                ]


def make_link_feedback_packet(tx_packet, sender_id, receiver_id, success, simulated_network_time, reason=""):
    """Create an ACK or NACK packet for a completed data-link transmission."""
    feedback_type = "ACK" if success else "NACK"
    return LinkFeedbackPacket(
        feedback_type=feedback_type,
        packet_id=tx_packet.packet_id,
        source_id=tx_packet.source_id,
        destination_id=tx_packet.destination_id,
        generation_id=tx_packet.generation_id,
        sender_id=receiver_id,
        receiver_id=sender_id,
        previous_hop=receiver_id,
        current_hop=sender_id,
        success=success,
        reason=reason,
        create_time=round(simulated_network_time, 2),
    )


def make_destination_decode_ack(packet_id, source_id, destination_id, generation_id, simulated_network_time, reason="generation_decoded"):
    """Create a control packet that reports successful generation decoding."""
    return LinkFeedbackPacket(
        feedback_type="DECODE_ACK",
        packet_id=packet_id,
        source_id=source_id,
        destination_id=destination_id,
        generation_id=generation_id,
        sender_id=destination_id,
        receiver_id=source_id,
        previous_hop=destination_id,
        current_hop=source_id,
        success=True,
        reason=reason,
        create_time=round(simulated_network_time, 2),
    )


def make_reward_ack(packet_id, source_id, destination_id, generation_id, simulated_network_time, reward_bonus, reverse_path, reason="round_reward_ack"):
    """Create a reward ACK that propagates round-level credit back to the source."""
    return LinkFeedbackPacket(
        feedback_type="REWARD_ACK",
        packet_id=packet_id,
        source_id=source_id,
        destination_id=destination_id,
        generation_id=generation_id,
        sender_id=destination_id,
        receiver_id=source_id,
        previous_hop=destination_id,
        current_hop=destination_id,
        success=True,
        reason=reason,
        create_time=round(simulated_network_time, 2),
        reward_bonus=reward_bonus,
        reverse_path=list(reverse_path),
    )


def source_has_decode_ack(nodelist, source_id, generation_id):
    """Return whether the source has already received a decode ACK for the generation."""
    return any(
        feedback.is_decode_ack and feedback.generation_id == generation_id
        for feedback in nodelist[source_id].received_feedback_history
    )


def source_has_reward_ack(nodelist, source_id, generation_id=None, packet_id=None):
    """Return whether the source has already received a matching reward ACK.

    Matching priority:
    1) packet_id (when provided)
    2) generation_id (fallback for backward compatibility)
    """
    if packet_id is not None:
        return any(
            feedback.is_reward_ack and feedback.packet_id == packet_id
            for feedback in nodelist[source_id].received_feedback_history
        )

    if generation_id is not None:
        return any(
            feedback.is_reward_ack and feedback.generation_id == generation_id
            for feedback in nodelist[source_id].received_feedback_history
        )

    return any(feedback.is_reward_ack for feedback in nodelist[source_id].received_feedback_history)
