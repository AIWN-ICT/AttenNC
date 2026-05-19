"""Packet-forwarding helpers for TDMA network-coded transmissions."""

import random

import numpy as np

try:
    from .coding_buffers import get_generation_buffer
    from .link_events import make_link_feedback_packet
    from .link_utils import calculate_link_delay_us
    from .network_layers import transmit_tdma
    from .node import NetworkCodedPacket
    from .topology_utils import getneighborNum
except ImportError:
    from coding_buffers import get_generation_buffer
    from link_events import make_link_feedback_packet
    from link_utils import calculate_link_delay_us
    from network_layers import transmit_tdma
    from node import NetworkCodedPacket
    from topology_utils import getneighborNum


def forward_data(
    nodelist,
    links,
    neighbor_M,
    node_id,
    data,
    event_manager,
    K,
    R,
    extrinsic_reward,
    simulated_network_time,
    node_efficiency_stats=None,
    packet_id=None,
    slot_duration=3000.0,
    frame_slot=None,
    node_positions=None,
    packet_length=None,
    bit_transmission_time=1.0 / (4e6),
    bit_transport_time=1e6 / 3e8,
    phy_header_length=32,
    mac_header_length=32,
    propagation_guard_time=1.0,
    control_packet_length=128,
    control_mac_header_length=32,
    control_phy_header_length=32,
    pending_data_arrivals=None,
    pending_feedback_arrivals=None,
    already_in_slot=False,
):
    """Forward one coded packet over next-hop links and return reward plus updated time.

    This is an equivalent extraction of the original ``coding_pytorch.forward_data``
    logic. The caller owns the global simulation clock, so the updated
    ``simulated_network_time`` is returned explicitly.
    """
    if isinstance(data, NetworkCodedPacket):
        tx_packet = data
        tx_data = tx_packet.data.copy()
        packet_id = tx_packet.packet_id
    else:
        tx_packet = NetworkCodedPacket(
            packet_id=packet_id,
            source_id=node_id,
            destination_id=len(neighbor_M) - 1,
            generation_id=0,
            coefficients=data,
            payload=data,
            current_hop=node_id,
            create_time=round(simulated_network_time, 2),
        )
        tx_data = tx_packet.data.copy()

    # Reject all-zero coded packets because they cannot increase any downstream rank.
    zero_prefix_len = 0
    for i in range(len(tx_data)):
        if tx_data[i] > 0:
            break
        zero_prefix_len += 1
    if zero_prefix_len == len(tx_data):
        nb_set = getneighborNum(nodelist, neighbor_M, node_id)
        reward = -extrinsic_reward * len(nb_set)
        return reward, simulated_network_time

    sum_reward = 0
    rank_increased = False
    tx_nbits = packet_length if packet_length is not None else len(tx_data)

    if already_in_slot:
        # The outer TDMA loop has already advanced the global clock to this
        # node's active slot. Do not wait for another TDMA slot here; charge
        # only the over-the-air transmission plus propagation time.
        delivered_nodes = set()
        lost_nodes = set()
        delivery_times = {}
        loss_times = {}
        max_link_delay = 0.0
        tx_start_time = simulated_network_time
        for dst in range(node_id + 1, len(neighbor_M)):
            if neighbor_M[node_id][dst] == 1:
                link_delay = calculate_link_delay_us(
                    node_id,
                    dst,
                    tx_nbits,
                    node_positions,
                    bit_transmission_time,
                    bit_transport_time,
                    propagation_guard_time,
                    mac_header_length,
                    phy_header_length,
                )
                max_link_delay = max(max_link_delay, link_delay)
                if random.random() <= float(links[node_id][dst]):
                    delivered_nodes.add(dst)
                    delivery_times[dst] = tx_start_time + link_delay
                else:
                    lost_nodes.add(dst)
                    loss_times[dst] = tx_start_time + link_delay
        simulated_network_time += max_link_delay
    else:
        # Standalone MAC/PHY time model: wait for this node's TDMA slot, then
        # deliver after propagation delay plus packet transmission time.
        delivered_nodes, lost_nodes, hop_delay, delivery_times, loss_times = transmit_tdma(
            node_count=len(neighbor_M),
            neighbor_matrix=neighbor_M,
            links=links,
            src=node_id,
            payload=tx_data,
            slot_duration=slot_duration,
            frame_slot=frame_slot,
            node_positions=node_positions,
            packet_nbits=tx_nbits,
            bit_transmission_time=bit_transmission_time,
            bit_transport_time=bit_transport_time,
            phy_header_length=phy_header_length,
            mac_header_length=mac_header_length,
            propagation_guard_time=propagation_guard_time,
            start_time=simulated_network_time,
        )
        simulated_network_time += hop_delay
        delivery_times = {
            dst: simulated_network_time - hop_delay + delay
            for dst, delay in delivery_times.items()
        }
        loss_times = {
            dst: simulated_network_time - hop_delay + delay
            for dst, delay in loss_times.items()
        }

    for i in range(node_id + 1, len(neighbor_M)):
        if neighbor_M[node_id][i] == 1:
            if i in lost_nodes:
                loss_time = loss_times.get(i, simulated_network_time)
                feedback_delay = calculate_link_delay_us(
                    i,
                    node_id,
                    control_packet_length,
                    node_positions,
                    bit_transmission_time,
                    bit_transport_time,
                    propagation_guard_time,
                    control_mac_header_length,
                    control_phy_header_length,
                )
                nack_packet = make_link_feedback_packet(
                    tx_packet,
                    node_id,
                    i,
                    success=False,
                    simulated_network_time=simulated_network_time,
                    reason="link_loss",
                )
                nack_packet.create_time = round(loss_time, 2)
                event_manager.schedule_link_feedback(
                    nodelist,
                    neighbor_M,
                    node_id,
                    nack_packet,
                    R,
                    K,
                    loss_time + feedback_delay,
                    pending_feedback_arrivals,
                )
                event_manager.record_link_loss(nodelist, node_id, i)
                temp_memory = []
                generation_buffer = get_generation_buffer(nodelist[i], tx_packet.generation_id)
                l_m = len(generation_buffer)
                for m in range(l_m):
                    temp_memory.append(generation_buffer[m])
                rank1 = np.linalg.matrix_rank(temp_memory)
                rece_data = np.zeros(K)
                for j in range(K):
                    rece_data[j] = tx_data[j]
                temp_memory.append(rece_data)
                rank2 = np.linalg.matrix_rank(temp_memory)
                if rank2 > rank1:
                    reward = extrinsic_reward
                    rank_increased = True
                else:
                    reward = -extrinsic_reward
                sum_reward += reward

            elif i in delivered_nodes:
                arrival_time = delivery_times.get(i, simulated_network_time)
                feedback_delay = calculate_link_delay_us(
                    i,
                    node_id,
                    control_packet_length,
                    node_positions,
                    bit_transmission_time,
                    bit_transport_time,
                    propagation_guard_time,
                    control_mac_header_length,
                    control_phy_header_length,
                )
                ack_packet = make_link_feedback_packet(
                    tx_packet,
                    node_id,
                    i,
                    success=True,
                    simulated_network_time=simulated_network_time,
                    reason="received",
                )
                if pending_data_arrivals is None:
                    # Apply the arrival immediately when the caller is not batching events.
                    rank_increased_now = event_manager.apply_data_arrival(
                        nodelist,
                        neighbor_M,
                        node_id,
                        i,
                        tx_packet,
                        tx_data,
                        K,
                        R,
                    )
                    if rank_increased_now:
                        reward = extrinsic_reward
                        rank_increased = True
                    else:
                        reward = -extrinsic_reward
                    sum_reward += reward
                    ack_packet.create_time = round(arrival_time, 2)
                    event_manager.schedule_link_feedback(
                        nodelist,
                        neighbor_M,
                        node_id,
                        ack_packet,
                        R,
                        K,
                        arrival_time + feedback_delay,
                        pending_feedback_arrivals,
                    )
                else:
                    # Estimate rank impact first, then defer the actual arrival event.
                    generation_buffer = get_generation_buffer(nodelist[i], tx_packet.generation_id)
                    temp_memory = [row for row in generation_buffer]
                    rank1 = np.linalg.matrix_rank(temp_memory)
                    rece_data = np.zeros(K)
                    for j in range(K):
                        rece_data[j] = tx_data[j]
                    temp_memory.append(rece_data)
                    rank2 = np.linalg.matrix_rank(temp_memory)
                    if rank2 > rank1:
                        reward = extrinsic_reward
                        rank_increased = True
                    else:
                        reward = -extrinsic_reward
                    sum_reward += reward
                    pending_data_arrivals.append({
                        'arrival_time': arrival_time,
                        'nodelist': nodelist,
                        'neighbor_M': neighbor_M,
                        'src': node_id,
                        'dst': i,
                        'tx_packet': tx_packet,
                        'tx_data': tx_data.copy(),
                        'K': K,
                        'R': R,
                        'ack_packet': ack_packet,
                        'feedback_arrival_time': arrival_time + feedback_delay,
                    })

    # Update node-level coding-efficiency statistics after the forwarding attempt.
    if node_efficiency_stats is not None and 0 < node_id < len(neighbor_M) - 1:
        current_L = min(nodelist[node_id].codelen, R)
        node_efficiency_stats[node_id].update_stats(current_L, rank_increased)
        node_efficiency_stats[node_id].update_optimal_L()

    return sum_reward, simulated_network_time
