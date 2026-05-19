from __future__ import annotations

import os
import pickle
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import simulation_slots as sim_slots
from coding_buffers import generation_rank, get_generation_buffer
from efficiency_stats import NodeEfficiencyStats
from link_events import source_has_decode_ack, source_has_reward_ack
from network_layers import set_tdma_seed
from node import NODE
from results_writer import CsvOutputPayload, build_result_paths, write_csv_outputs
from rewards import calculate_reward, calculate_reward_GNN
from simulation_context import SimulationContext
from topology_utils import get_tdma_slot_schedule


torch.backends.cudnn.enabled = False
np.set_printoptions(threshold=np.inf)

seed = 555
BASE_DIR = Path(__file__).resolve().parent

_time_stats_template = {
    "source": [],
    "relay": {},
}


def run_evaluate(
    *,
    e: int,
    config: dict[str, Any],
    agent_s,
    agent_r,
    model,
    result_dir,
    min_f,
    load_best_state: bool = False,
    best_state_path=None,
    save_csv: bool = True,
    enable_training_updates: bool = False,
):
    """Run one evaluation episode over multiple TDMA tests and persist summary outputs."""

    node_num = config["node_num"]
    neighbor_matrix = config["neighbor_matrix"]
    links = config["links"]
    device = config["device"]
    Max_s_f = config["Max_s_f"]
    Max_test = config["Max_test"]
    extrinsic_reward = config["extrinsic_reward"]
    parallel_path = config["parallel_path"]
    max_nb = config["max_nb"]
    K = config["K"]
    M = config["M"]
    slot_duration = config.get("slot_duration", 0.003)
    frame_slot = config.get("frame_slot", None)
    node_positions = config.get("node_positions", None)
    if not node_positions:
        raise ValueError("node_positions is required and should be defined in config_topology.py for propagation-delay calculation.")
    packet_length = config.get("packet_length", K)
    mac_header_length = config.get("mac_header_length", 0)
    phy_header_length = config.get("phy_header_length", 0)
    control_packet_length = config.get("control_packet_length", 128)
    control_mac_header_length = config.get("control_mac_header_length", mac_header_length)
    control_phy_header_length = config.get("control_phy_header_length", phy_header_length)
    bit_transmission_time = config.get("bit_transmission_time", 1.0)
    bit_transport_time = config.get("bit_transport_time", 0.0)
    propagation_guard_time = config.get("propagation_guard_time", 0.0)
    include_inference_delay_in_sim_time = config.get("include_inference_delay_in_sim_time", False)
    inference_delay_time_unit = config.get("inference_delay_time_unit", 1e6)
    include_decode_compute_delay_in_sim_time = config.get("include_decode_compute_delay_in_sim_time", True)
    decode_compute_delay_coefficient_us = config.get("decode_compute_delay_coefficient_us", 1.0)
    use_measured_decode_delay = config.get("use_measured_decode_delay", False)
    enable_relay_coding_selection = config.get("enable_relay_coding_selection", True)
    tdma_slot_schedule = get_tdma_slot_schedule(frame_slot, node_num)

    S_state_size = (M * parallel_path + 2) * K
    R_state_size = (M * max_nb + 2) * K
    source_id = 0

    os.makedirs(result_dir, exist_ok=True)
    result_paths = build_result_paths(result_dir, e)

    if load_best_state and best_state_path and Path(best_state_path).exists():
        with open(best_state_path, "rb") as f:
            data = pickle.load(f)
            best_state = data["best_state"]
        np_random_state, random_state = best_state
        np.random.set_state(np_random_state)
        random.setstate(random_state)
        set_tdma_seed(seed + e)

    avg_reward = 0.0
    avg_s_f = 0.0
    avg_overhead = 0
    start_time = time.time()
    np_random_state = np.random.get_state()
    random_state = random.getstate()
    source_send_count_list = []
    decode_delay_stats = []
    set_tdma_seed(seed + e)

    node_coding_stats = {i: {"Coding": 0, "Non-Coding": 0} for i in range(1, node_num - 1)}
    action_stats = {"source": {"encoding_count": 0}}
    for i in range(1, node_num - 1):
        action_stats[i] = {"gnn_count": 0, "encoding_count": 0}

    gnn_decision_stats = {"output_1_count": 0, "output_0_count": 0}
    node_efficiency_stats = {i: NodeEfficiencyStats() for i in range(1, node_num - 1)}
    slot_action_records = []
    time_stats = {"source": [], "relay": {}}
    ctx = SimulationContext()

    for test in range(Max_test):
        ctx.reset()
        total_reward = 0
        test_nodelist = []
        for i in range(node_num):
            node = NODE(K, M, i)
            node.reset_episode_state()
            test_nodelist.append(node)

        for i in range(node_num):
            node = test_nodelist[i]
            node.local_stats["code_packets"] = 0
            node.local_stats["code_rank"] = 0
            node.local_stats["edge_loss_count"].clear()
            node.local_stats["edge_rank_increase_count"].clear()

        ss_f_num = 0
        rank_des_buffer = 0
        max_rank = 0
        active_generation_id = 1
        source_packet = None
        reward_ack_sent_generations = set()
        generation_start_times = {}
        generation_decode_recorded = set()
        generation_decode_compute_delays = []
        decode_ack_scheduled_generations = set()

        transmission_params = sim_slots.TransmissionParams(
            links=links,
            event_manager=ctx.event_manager,
            extrinsic_reward=extrinsic_reward,
            slot_duration=slot_duration,
            frame_slot=frame_slot,
            node_positions=node_positions,
            packet_length=packet_length,
            bit_transmission_time=bit_transmission_time,
            bit_transport_time=bit_transport_time,
            phy_header_length=phy_header_length,
            mac_header_length=mac_header_length,
            propagation_guard_time=propagation_guard_time,
            control_packet_length=control_packet_length,
            control_mac_header_length=control_mac_header_length,
            control_phy_header_length=control_phy_header_length,
            pending_data_arrivals=ctx.pending_data_arrivals,
            pending_feedback_arrivals=ctx.pending_feedback_arrivals,
        )
        timing_params = sim_slots.TimingParams(
            include_inference_delay_in_sim_time=include_inference_delay_in_sim_time,
            inference_delay_time_unit=inference_delay_time_unit,
        )
        decode_timing_params = sim_slots.DecodeTimingParams(
            pending_decode_ack_arrivals=ctx.pending_decode_ack_arrivals,
            source_id=source_id,
            generation_start_times=generation_start_times,
            generation_decode_recorded=generation_decode_recorded,
            generation_decode_compute_delays=generation_decode_compute_delays,
            decode_ack_scheduled_generations=decode_ack_scheduled_generations,
            use_measured_decode_delay=use_measured_decode_delay,
            decode_compute_delay_coefficient_us=decode_compute_delay_coefficient_us,
            include_decode_compute_delay_in_sim_time=include_decode_compute_delay_in_sim_time,
            decode_delay_stats=decode_delay_stats,
        )

        while ss_f_num < Max_s_f:
            for current_slot, current_tdma_node in tdma_slot_schedule:
                ctx.flush_pending_events(ctx.simulated_network_time)
                slot_start_time = ctx.simulated_network_time
                slot_extra_delay = 0.0
                decode_ack_received = source_has_decode_ack(test_nodelist, source_id, active_generation_id)

                if current_tdma_node == source_id and not decode_ack_received:
                    ss_f_num += 1
                    source_params = sim_slots.SourceSlotParams(
                        source_id=source_id,
                        node_num=node_num,
                        neighbor_matrix=neighbor_matrix,
                        state_size=S_state_size,
                        K=K,
                        R=M,
                        agent_s=agent_s,
                        generation_start_times=generation_start_times,
                        active_generation_id=active_generation_id,
                        packet_sequence=ctx.packet_sequence,
                        simulated_network_time=ctx.simulated_network_time,
                        episode_idx=e,
                        test_idx=test,
                        time_stats=time_stats,
                        action_stats=action_stats,
                        node_efficiency_stats=node_efficiency_stats,
                        transmission=transmission_params,
                        timing=timing_params,
                    )
                    rewards, ctx.simulated_network_time, ctx.packet_sequence, slot_extra_delay, source_rank_before, source_packet = sim_slots.execute_source_slot(
                        test_nodelist,
                        source_params,
                    )
                    ctx.flush_pending_events(ctx.simulated_network_time)
                    source_rank_after = generation_rank(test_nodelist[node_num - 1], source_packet.generation_id)
                    slot_action_records.append(
                        {
                            "Episode": e,
                            "Test": test,
                            "Frame_Source_Packet_Index": ss_f_num,
                            "Slot": int(current_slot),
                            "Node_ID": int(source_id),
                            "Node_Type": "Source",
                            "Queue_Before": len(test_nodelist[source_id].packet),
                            "Queue_After": len(test_nodelist[source_id].packet),
                            "Decision": 1,
                            "Encoded_Count": K,
                            "Delivered_To_Destination": int(source_rank_after > source_rank_before),
                            "Dest_Rank_Before": int(source_rank_before),
                            "Dest_Rank_After": int(source_rank_after),
                            "Dest_Rank_Increase": int(source_rank_after - source_rank_before),
                        }
                    )
                    test_nodelist[source_id].list_reward_obj.append(torch.tensor(rewards, dtype=torch.float32).clone().detach())

                elif 0 < current_tdma_node < node_num - 1:
                    i = current_tdma_node
                    if len(test_nodelist[i].packet) > 0:
                        test_nodelist[i].receive_flag = False
                        relay_params = sim_slots.RelaySlotParams(
                            relay_id=i,
                            node_num=node_num,
                            neighbor_matrix=neighbor_matrix,
                            state_size=R_state_size,
                            K=K,
                            R=M,
                            max_nb=max_nb,
                            model=model,
                            agent_r=agent_r,
                            device=device,
                            node_efficiency_stats=node_efficiency_stats,
                            node_coding_stats=node_coding_stats,
                            action_stats=action_stats,
                            gnn_decision_stats=gnn_decision_stats,
                            time_stats=time_stats,
                            transmission=transmission_params,
                            timing=timing_params,
                            simulated_network_time=ctx.simulated_network_time,
                            active_generation_id=active_generation_id,
                            decode_timing=decode_timing_params,
                            enable_relay_coding_selection=enable_relay_coding_selection,
                        )
                        ctx.simulated_network_time, slot_extra_delay, action_val, actual_encoded_count, queue_before, dest_rank_before, dest_rank_after = sim_slots.execute_relay_slot(
                            test_nodelist,
                            relay_params,
                        )
                        ctx.flush_pending_events(ctx.simulated_network_time)
                        slot_action_records.append(
                            {
                                "Episode": e,
                                "Test": test,
                                "Frame_Source_Packet_Index": ss_f_num,
                                "Slot": int(current_slot),
                                "Node_ID": int(i),
                                "Node_Type": "Relay",
                                "Queue_Before": int(queue_before),
                                "Queue_After": int(len(test_nodelist[i].packet)),
                                "Decision": int(action_val),
                                "Encoded_Count": int(actual_encoded_count),
                                "Delivered_To_Destination": int(dest_rank_after > dest_rank_before),
                                "Dest_Rank_Before": int(dest_rank_before),
                                "Dest_Rank_After": int(dest_rank_after),
                                "Dest_Rank_Increase": int(dest_rank_after - dest_rank_before),
                            }
                        )

                ctx.simulated_network_time = max(
                    ctx.simulated_network_time,
                    slot_start_time + slot_duration + slot_extra_delay,
                )
                ctx.flush_pending_events(ctx.simulated_network_time)

            pending_queue = any(len(test_nodelist[i].packet) > 0 for i in range(1, node_num - 1)) or bool(ctx.pending_data_arrivals)
            decode_ack_received = source_has_decode_ack(test_nodelist, source_id, active_generation_id)

            if not pending_queue or decode_ack_received:
                rank = generation_rank(test_nodelist[node_num - 1], active_generation_id)
                S = rank - rank_des_buffer
                round_bonus = calculate_reward(K, S, rank)
                if rank > max_rank:
                    max_rank = rank

                if source_packet is not None and active_generation_id not in reward_ack_sent_generations:
                    sim_slots.schedule_reward_ack(
                        test_nodelist,
                        ctx.event_manager,
                        source_id,
                        node_num - 1,
                        source_packet,
                        round_bonus,
                        ctx.simulated_network_time,
                        ctx.pending_reward_ack_arrivals,
                    )
                    reward_ack_sent_generations.add(active_generation_id)
                    ctx.flush_pending_events(ctx.simulated_network_time)

                for i in range(1, node_num - 1):
                    for j in range(test_nodelist[i].gnn_list_len):
                        old_reward = test_nodelist[i].list_reward_obj[j]
                        action = test_nodelist[i].list_action_obj[j]
                        test_nodelist[i].list_reward_obj[j] = calculate_reward_GNN(K, S, rank, action) + old_reward
                rank_des_buffer = rank

                if source_has_reward_ack(test_nodelist, source_id, active_generation_id):
                    sim_slots.flush_round_transitions(test_nodelist, agent_s, agent_r, enable_training_updates)

                if decode_ack_received:
                    total_reward = total_reward / ss_f_num
                    break

        avg_reward += total_reward
        source_send_count_list.append(ss_f_num)
        avg_s_f += ss_f_num
        decoded_generation_packets = len(get_generation_buffer(test_nodelist[node_num - 1], active_generation_id))
        overhead = (1 / K) * (decoded_generation_packets - K) * 100
        avg_overhead += overhead
        elapsed = time.time() - start_time
        average_time = elapsed / (test + 1)
        remaining_time = average_time * (Max_test - (test + 1))
        print("\r test: %d, remaining: %d s" % (test, remaining_time), end="")

    avg_s_f = avg_s_f / Max_test
    avg_reward = avg_reward / Max_test
    avg_overhead = avg_overhead / Max_test

    with open(result_paths.test_log, "a+", encoding="utf-8") as fl:
        print(e, " ", avg_s_f, file=fl)
        print(" ", file=fl)
    with open(result_paths.reward_log, "a+", encoding="utf-8") as fe:
        print(e, " ", avg_reward, file=fe)
        print(" ", file=fe)

    is_best = avg_s_f < min_f

    if save_csv:
        write_csv_outputs(
            result_paths,
            CsvOutputPayload(
                episode_idx=e,
                node_num=node_num,
                source_id=source_id,
                action_stats=action_stats,
                node_coding_stats=node_coding_stats,
                gnn_decision_stats=gnn_decision_stats,
                slot_action_records=slot_action_records,
                time_stats=time_stats,
                decode_delay_stats=decode_delay_stats,
                generation_decode_compute_delays=generation_decode_compute_delays,
                include_inference_delay_in_sim_time=include_inference_delay_in_sim_time,
                include_decode_compute_delay_in_sim_time=include_decode_compute_delay_in_sim_time,
                use_measured_decode_delay=use_measured_decode_delay,
            ),
        )

    return {
        "min_f": min_f,
        "avg_overhead": avg_overhead,
        "avg_s_f": avg_s_f,
        "avg_reward": avg_reward,
        "is_best": is_best,
        "best_state": (np_random_state, random_state),
        "source_send_count_list": source_send_count_list,
    }
