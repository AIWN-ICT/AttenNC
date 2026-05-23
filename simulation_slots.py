from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from coding_actions import act, p_xor
from coding_buffers import calculate_decode_compute_delay_us, generation_rank, get_generation_buffer
from forwarding import forward_data
from link_events import make_destination_decode_ack, make_reward_ack
from link_utils import calculate_link_delay_us
from node import NetworkCodedPacket, get_local_neighbor_receivememory
from topology_utils import getneighborNum


@dataclass(frozen=True)
class TransmissionParams:
    """Shared transmission and propagation parameters for one simulation run."""

    links: Any
    event_manager: Any
    extrinsic_reward: float
    slot_duration: float
    frame_slot: Any
    node_positions: Any
    packet_length: int
    bit_transmission_time: float
    bit_transport_time: float
    phy_header_length: int
    mac_header_length: int
    propagation_guard_time: float
    control_packet_length: int
    control_mac_header_length: int
    control_phy_header_length: int
    pending_data_arrivals: list[Any]
    pending_feedback_arrivals: list[Any]


@dataclass(frozen=True)
class TimingParams:
    """Controls whether measured compute time is injected into simulation time."""

    include_inference_delay_in_sim_time: bool
    inference_delay_time_unit: float


@dataclass(frozen=True)
class SourceSlotParams:
    """Inputs required to execute a source TDMA slot."""

    source_id: int
    node_num: int
    neighbor_matrix: Any
    state_size: int
    K: int
    R: int
    agent_s: Any
    generation_start_times: dict[int, float]
    active_generation_id: int
    packet_sequence: int
    simulated_network_time: float
    episode_idx: int
    test_idx: int
    time_stats: dict[str, Any]
    action_stats: dict[str, Any]
    node_efficiency_stats: dict[int, Any]
    transmission: TransmissionParams
    timing: TimingParams


@dataclass(frozen=True)
class DecodeTimingParams:
    """Parameters used when modeling destination decode delay."""

    pending_decode_ack_arrivals: list[Any]
    source_id: int
    generation_start_times: dict[int, float]
    generation_decode_recorded: set[int]
    generation_decode_compute_delays: list[float]
    decode_ack_scheduled_generations: set[int]
    use_measured_decode_delay: bool
    decode_compute_delay_coefficient_us: float
    include_decode_compute_delay_in_sim_time: bool
    decode_delay_stats: list[float]


@dataclass(frozen=True)
class RelaySlotParams:
    """Inputs required to execute a relay TDMA slot."""

    relay_id: int
    node_num: int
    neighbor_matrix: Any
    state_size: int
    K: int
    R: int
    max_nb: int
    model: Any
    agent_r: Any
    device: Any
    node_efficiency_stats: dict[int, Any]
    node_coding_stats: dict[int, dict[str, int]]
    action_stats: dict[str, Any]
    gnn_decision_stats: dict[str, int]
    time_stats: dict[str, Any]
    transmission: TransmissionParams
    timing: TimingParams
    simulated_network_time: float
    active_generation_id: int
    decode_timing: DecodeTimingParams
    enable_relay_coding_selection: bool = True


def reset_dqn_episode_buffers(node) -> None:
    """Reset per-slot replay buffers stored on a node."""

    node.list_action = np.zeros(node.K)
    node.list_state = []
    node.list_n_state = []
    node.list_rewards = np.zeros(node.K)
    node.list_len = 0


def flush_dqn_transitions(node, agent, enable_training_updates: bool = False) -> None:
    """Push buffered DQN transitions from one node into the agent replay buffer."""

    if not enable_training_updates:
        return
    for j in range(node.list_len):
        done = 1 if j == node.list_len - 1 else 0
        agent.remember(node.list_state[j], node.list_action[j], node.list_rewards[j], node.list_n_state[j], done)


def flush_round_transitions(nodelist, agent_s, agent_r, enable_training_updates: bool = False) -> None:
    """Flush buffered transitions for all nodes at the end of a round."""

    if not enable_training_updates:
        return
    for node_id, node in enumerate(nodelist):
        if node.list_len == 0:
            continue
        agent = agent_s if node_id == 0 else agent_r
        flush_dqn_transitions(node, agent, enable_training_updates=True)


def schedule_reward_ack(
    test_nodelist,
    event_manager,
    source_id: int,
    destination_id: int,
    packet,
    reward_bonus: float,
    simulated_network_time: float,
    pending_reward_ack_arrivals,
) -> None:
    """Schedule a reward ACK to travel back along the reverse packet path."""

    reverse_path = list(reversed(packet.path)) if getattr(packet, "path", None) else [destination_id, source_id]
    reward_ack = make_reward_ack(
        packet.packet_id,
        source_id,
        destination_id,
        packet.generation_id,
        simulated_network_time,
        reward_bonus,
        reverse_path,
    )
    event_manager.schedule_reward_ack(
        test_nodelist,
        source_id,
        destination_id,
        reward_ack,
        simulated_network_time,
        pending_reward_ack_arrivals,
    )


def execute_source_slot(test_nodelist, params: SourceSlotParams):
    """Execute one source-node transmission slot and forward the encoded packet."""

    source_rank_before = generation_rank(test_nodelist[params.node_num - 1], params.active_generation_id)
    reset_dqn_episode_buffers(test_nodelist[params.source_id])

    k_data = np.zeros(params.K)
    pre_data = np.zeros(params.K)
    k_data[0] = 1
    state, _ = test_nodelist[params.source_id].build_source_state(
        test_nodelist,
        params.neighbor_matrix,
        params.source_id,
        params.state_size,
        params.K,
        params.R,
    )

    start_time_source = time.time()
    for t in range(params.K):
        test_nodelist[params.source_id].list_len += 1
        xstate = np.resize(state, [1, params.state_size])
        with torch.no_grad():
            q = params.agent_s.get_q_val_with_cache(xstate)
        action = act(q)
        params.action_stats["source"]["encoding_count"] += 1
        test_nodelist[params.source_id].list_action[t] = action
        next_state = np.zeros(params.state_size)
        if t < params.K - 1:
            k_data = np.zeros(params.K)
            k_data[t + 1] = 1
            pre_data[t] = action
            next_state[0 : params.K] = k_data
            next_state[params.K : 2 * params.K] = pre_data
            neighbors = getneighborNum(test_nodelist, params.neighbor_matrix, params.source_id)
            for neighbor_idx, neighbor in enumerate(neighbors):
                neighbor_memory = get_local_neighbor_receivememory(
                    test_nodelist,
                    params.neighbor_matrix,
                    params.source_id,
                    neighbor,
                    params.R,
                    params.K,
                )
                for i in range(params.R):
                    start = 2 * params.K + neighbor_idx * params.R * params.K + i * params.K
                    next_state[start : start + params.K] = neighbor_memory[i]
        test_nodelist[params.source_id].list_state.append(copy.deepcopy(state))
        test_nodelist[params.source_id].list_n_state.append(copy.deepcopy(next_state))
        state = next_state

    params.agent_s.clear_cache()
    source_inference_delay = time.time() - start_time_source
    params.time_stats["source"].append(source_inference_delay)
    slot_extra_delay = 0.0
    if params.timing.include_inference_delay_in_sim_time:
        slot_extra_delay += source_inference_delay * params.timing.inference_delay_time_unit

    data = np.array(test_nodelist[params.source_id].list_action, dtype=np.float32).copy()
    packet_sequence = params.packet_sequence + 1
    packet_id = f"{params.source_id}_{packet_sequence}_{params.node_num - 1}_{params.episode_idx}_{params.test_idx}"
    source_packet = NetworkCodedPacket(
        packet_id=packet_id,
        source_id=params.source_id,
        destination_id=params.node_num - 1,
        generation_id=params.active_generation_id,
        coefficients=data,
        payload=data,
        current_hop=params.source_id,
        path=[params.source_id],
        create_time=round(params.simulated_network_time, 2),
    )
    params.generation_start_times.setdefault(source_packet.generation_id, source_packet.create_time)
    rewards, simulated_network_time = forward_data(
        test_nodelist,
        params.transmission.links,
        params.neighbor_matrix,
        params.source_id,
        source_packet,
        params.transmission.event_manager,
        params.K,
        params.R,
        params.transmission.extrinsic_reward,
        params.simulated_network_time,
        node_efficiency_stats=params.node_efficiency_stats,
        packet_id=packet_id,
        slot_duration=params.transmission.slot_duration,
        frame_slot=params.transmission.frame_slot,
        node_positions=params.transmission.node_positions,
        packet_length=params.transmission.packet_length,
        bit_transmission_time=params.transmission.bit_transmission_time,
        bit_transport_time=params.transmission.bit_transport_time,
        phy_header_length=params.transmission.phy_header_length,
        mac_header_length=params.transmission.mac_header_length,
        propagation_guard_time=params.transmission.propagation_guard_time,
        control_packet_length=params.transmission.control_packet_length,
        control_mac_header_length=params.transmission.control_mac_header_length,
        control_phy_header_length=params.transmission.control_phy_header_length,
        pending_data_arrivals=params.transmission.pending_data_arrivals,
        pending_feedback_arrivals=params.transmission.pending_feedback_arrivals,
        already_in_slot=True,
    )
    test_nodelist[params.source_id].list_reward_obj.append(torch.tensor(rewards, dtype=torch.float32).clone().detach())
    test_nodelist[params.source_id].list_rewards[: test_nodelist[params.source_id].list_len] = rewards
    return rewards, simulated_network_time, packet_sequence, slot_extra_delay, source_rank_before, source_packet


def execute_relay_slot(test_nodelist, params: RelaySlotParams):
    """Execute one relay slot, including GNN gating, recoding, and forwarding."""

    relay_id = params.relay_id
    queue_before = len(test_nodelist[relay_id].packet)
    last_data, rx_packet = test_nodelist[relay_id].getpacket()
    packet_generation_id = rx_packet.generation_id if rx_packet is not None else params.active_generation_id
    dest_rank_before = generation_rank(test_nodelist[params.node_num - 1], packet_generation_id)
    packet_id = rx_packet.packet_id if rx_packet is not None else None

    reset_dqn_episode_buffers(test_nodelist[relay_id])
    r_data = np.array(last_data, dtype=np.float32).copy()

    combined_features, adj_mask, _next_hop_nodes = test_nodelist[relay_id].build_relay_features(
        test_nodelist,
        params.neighbor_matrix,
        relay_id,
        params.max_nb,
        params.K,
    )

    l = test_nodelist[relay_id].codelen
    L = np.minimum(l, params.R)

    if relay_id not in params.time_stats["relay"]:
        params.time_stats["relay"][relay_id] = []

    start_time_relay = time.time()
    if not getattr(params, "enable_relay_coding_selection", True):
        action = 1
        action_val = 1
        params.node_coding_stats[relay_id]["Coding"] += 1
    elif not params.node_efficiency_stats[relay_id].should_use_gnn(L):
        action = 1
        action_val = 1
        params.node_coding_stats[relay_id]["Coding"] += 1
    else:
        input_features = test_nodelist[relay_id].prepare_relay_tensor(combined_features, params.device)
        input_adj = test_nodelist[relay_id].prepare_adj_tensor(adj_mask, params.device)
        action = params.model.select_action(input_features, adj_mask=input_adj)
        params.action_stats[relay_id]["gnn_count"] += 1
        action_val = action.item() if isinstance(action, torch.Tensor) else action
        if action_val == 1:
            params.gnn_decision_stats["output_1_count"] += 1
            params.node_coding_stats[relay_id]["Coding"] += 1
        else:
            params.gnn_decision_stats["output_0_count"] += 1
            params.node_coding_stats[relay_id]["Non-Coding"] += 1

    test_nodelist[relay_id].list_data_graph.append(combined_features.reshape(-1))
    test_nodelist[relay_id].list_action_obj.append(
        action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.float)
    )

    actual_encoded_count = 0
    slot_extra_delay = 0.0
    if action_val == 0:
        relay_inference_delay = time.time() - start_time_relay
        params.time_stats["relay"][relay_id].append(relay_inference_delay)
        if params.timing.include_inference_delay_in_sim_time:
            slot_extra_delay += relay_inference_delay * params.timing.inference_delay_time_unit

    if action == 1:
        state, _ = test_nodelist[relay_id].build_relay_state(
            test_nodelist,
            params.neighbor_matrix,
            relay_id,
            last_data,
            params.state_size,
            params.K,
            params.R,
        )
        for t in range(L):
            test_nodelist[relay_id].list_len += 1
            test_nodelist[relay_id].list_state.append(copy.deepcopy(state))
            xstate = np.resize(state, [1, params.state_size])
            with torch.no_grad():
                q = params.agent_r.get_q_val(xstate)
            action = act(q)
            params.action_stats[relay_id]["encoding_count"] += 1
            actual_encoded_count += 1
            test_nodelist[relay_id].list_action[t] = action
            next_state = np.zeros(params.state_size)
            if action == 1:
                r_data = p_xor(r_data, test_nodelist[relay_id].codememory[t])
            if t < params.R - 1:
                next_state[0 : params.K] = r_data
                next_state[params.K : 2 * params.K] = test_nodelist[relay_id].codememory[t + 1]
                neighbors = getneighborNum(test_nodelist, params.neighbor_matrix, relay_id)
                for neighbor_idx, neighbor in enumerate(neighbors):
                    neighbor_memory = get_local_neighbor_receivememory(
                        test_nodelist,
                        params.neighbor_matrix,
                        relay_id,
                        neighbor,
                        params.R,
                        params.K,
                    )
                    for j in range(params.R):
                        start = 2 * params.K + neighbor_idx * params.R * params.K + j * params.K
                        next_state[start : start + params.K] = neighbor_memory[j]
            test_nodelist[relay_id].list_n_state.append(copy.deepcopy(next_state))
            state = next_state
        relay_inference_delay = time.time() - start_time_relay
        params.time_stats["relay"][relay_id].append(relay_inference_delay)
        if params.timing.include_inference_delay_in_sim_time:
            slot_extra_delay += relay_inference_delay * params.timing.inference_delay_time_unit

    if relay_id in params.node_efficiency_stats:
        l = test_nodelist[relay_id].codelen
        L = np.minimum(l, params.R)
        action_val = action.item() if isinstance(action, torch.Tensor) else action
        params.node_efficiency_stats[relay_id].update_gnn_stats(L, action_val, actual_encoded_count)

    l = test_nodelist[relay_id].codelen
    test_nodelist[relay_id].codememory[int(l % params.R)] = last_data
    test_nodelist[relay_id].codelen = test_nodelist[relay_id].codelen + 1

    relay_packet = rx_packet.recode_with(r_data, relay_id) if rx_packet is not None else NetworkCodedPacket(
        packet_id=packet_id,
        source_id=relay_id,
        destination_id=params.node_num - 1,
        generation_id=packet_generation_id,
        coefficients=r_data,
        payload=r_data,
        current_hop=relay_id,
        create_time=round(params.simulated_network_time, 2),
    )
    des_generation_buffer = get_generation_buffer(test_nodelist[params.node_num - 1], relay_packet.generation_id)
    des_packet_count_before = len(des_generation_buffer)
    rewards, simulated_network_time = forward_data(
        test_nodelist,
        params.transmission.links,
        params.neighbor_matrix,
        relay_id,
        relay_packet,
        params.transmission.event_manager,
        params.K,
        params.R,
        params.transmission.extrinsic_reward,
        params.simulated_network_time,
        node_efficiency_stats=params.node_efficiency_stats,
        packet_id=packet_id,
        slot_duration=params.transmission.slot_duration,
        frame_slot=params.transmission.frame_slot,
        node_positions=params.transmission.node_positions,
        packet_length=params.transmission.packet_length,
        bit_transmission_time=params.transmission.bit_transmission_time,
        bit_transport_time=params.transmission.bit_transport_time,
        phy_header_length=params.transmission.phy_header_length,
        mac_header_length=params.transmission.mac_header_length,
        propagation_guard_time=params.transmission.propagation_guard_time,
        control_packet_length=params.transmission.control_packet_length,
        control_mac_header_length=params.transmission.control_mac_header_length,
        control_phy_header_length=params.transmission.control_phy_header_length,
        pending_data_arrivals=params.transmission.pending_data_arrivals,
        pending_feedback_arrivals=params.transmission.pending_feedback_arrivals,
        already_in_slot=True,
    )
    params.transmission.event_manager.flush_pending_events(simulated_network_time, None, simulated_network_time)
    des_packet_count_after = len(des_generation_buffer)
    test_nodelist[relay_id].list_reward_obj.append(torch.tensor(rewards, dtype=torch.float32).clone().detach())
    test_nodelist[relay_id].list_rewards[: test_nodelist[relay_id].list_len] = rewards

    if des_packet_count_after > des_packet_count_before:
        generation_id = relay_packet.generation_id
        decode_start_wall_time = time.time()
        current_des_rank = generation_rank(test_nodelist[params.node_num - 1], generation_id)
        if current_des_rank == params.K:
            decode_compute_delay_us = calculate_decode_compute_delay_us(
                params.K,
                decode_start_wall_time,
                params.decode_timing.use_measured_decode_delay,
                params.decode_timing.decode_compute_delay_coefficient_us,
                params.timing.inference_delay_time_unit,
            )
            if params.decode_timing.include_decode_compute_delay_in_sim_time:
                simulated_network_time += decode_compute_delay_us
            params.decode_timing.generation_decode_compute_delays.append(decode_compute_delay_us / 1e6)
            if generation_id in params.decode_timing.generation_start_times and generation_id not in params.decode_timing.generation_decode_recorded:
                params.decode_timing.decode_delay_stats.append(
                    (simulated_network_time - params.decode_timing.generation_start_times[generation_id]) / 1e6
                )
                params.decode_timing.generation_decode_recorded.add(generation_id)
            if generation_id not in params.decode_timing.decode_ack_scheduled_generations:
                decode_ack_packet = make_destination_decode_ack(
                    relay_packet.packet_id,
                    params.decode_timing.source_id,
                    params.node_num - 1,
                    generation_id,
                    simulated_network_time,
                )
                decode_ack_delay = calculate_link_delay_us(
                    params.node_num - 1,
                    params.decode_timing.source_id,
                    params.transmission.control_packet_length,
                    params.transmission.node_positions,
                    params.transmission.bit_transmission_time,
                    params.transmission.bit_transport_time,
                    params.transmission.propagation_guard_time,
                    params.transmission.control_mac_header_length,
                    params.transmission.control_phy_header_length,
                )
                params.transmission.event_manager.schedule_destination_decode_ack(
                    test_nodelist,
                    params.decode_timing.source_id,
                    params.node_num - 1,
                    decode_ack_packet,
                    simulated_network_time + decode_ack_delay,
                    params.decode_timing.pending_decode_ack_arrivals,
                )
                params.decode_timing.decode_ack_scheduled_generations.add(generation_id)

    if test_nodelist[relay_id].has_pending_packet():
        test_nodelist[relay_id].receive_flag = True

    dest_rank_after = generation_rank(test_nodelist[params.node_num - 1], packet_generation_id)
    return simulated_network_time, slot_extra_delay, action_val, actual_encoded_count, queue_before, dest_rank_before, dest_rank_after
