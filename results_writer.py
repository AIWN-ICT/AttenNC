from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CsvOutputPayload:
    """Bundle CSV output inputs generated during one evaluation episode."""

    episode_idx: int
    node_num: int
    source_id: int
    action_stats: dict[str, Any]
    node_coding_stats: dict[int, dict[str, int]]
    gnn_decision_stats: dict[str, int]
    slot_action_records: list[dict[str, Any]]
    time_stats: dict[str, Any]
    decode_delay_stats: list[float]
    generation_decode_compute_delays: list[float]
    include_inference_delay_in_sim_time: bool
    include_decode_compute_delay_in_sim_time: bool
    use_measured_decode_delay: bool


@dataclass(frozen=True)
class ResultPaths:
    """Resolved output paths for one evaluation run."""

    test_log: Path
    reward_log: Path
    action_counts: Path
    ratio_result: Path
    gnn_decision: Path
    slot_action: Path
    time_stats: Path
    decode_delay: Path


def build_result_paths(result_dir: str | Path, episode_idx: int) -> ResultPaths:
    """Create the set of output file paths used by evaluation."""
    result_dir = Path(result_dir)
    episode_tag = f"episode_{episode_idx:03d}"
    return ResultPaths(
        test_log=result_dir / "evaluation_log.txt",
        reward_log=result_dir / "reward_log.txt",
        action_counts=result_dir / f"{episode_tag}_action_counts.csv",
        ratio_result=result_dir / f"{episode_tag}_coding_ratio_summary.csv",
        gnn_decision=result_dir / f"{episode_tag}_gnn_decision_summary.csv",
        slot_action=result_dir / f"{episode_tag}_slot_action_log.csv",
        time_stats=result_dir / f"{episode_tag}_inference_time_summary.csv",
        decode_delay=result_dir / f"{episode_tag}_decode_delay_summary.csv",
    )


def write_csv_outputs(result_paths: ResultPaths, payload: CsvOutputPayload) -> None:
    """Persist all per-episode CSV summaries and print timing diagnostics."""
    _write_action_counts(result_paths.action_counts, payload)
    _write_coding_ratios(result_paths.ratio_result, payload)
    _write_gnn_decisions(result_paths.gnn_decision, payload)
    pd.DataFrame(payload.slot_action_records).to_csv(result_paths.slot_action, index=False)

    time_stats_data, relay_avg_times, all_times = _build_time_stats(payload)
    _print_time_stats(payload.episode_idx, payload.time_stats, relay_avg_times, all_times)
    pd.DataFrame(time_stats_data).to_csv(result_paths.time_stats, index=False)

    _write_decode_delay_stats(result_paths.decode_delay, payload, all_times)


def _write_action_counts(output_path: Path, payload: CsvOutputPayload) -> None:
    """Write source and relay action counts."""
    action_df_data = {
        "Node_ID": ["Source"],
        "GNN_Count": [0],
        "Encoding_Count": [payload.action_stats["source"]["encoding_count"]],
    }
    for node_id in range(1, payload.node_num - 1):
        action_df_data["Node_ID"].append(f"Node_{node_id}")
        action_df_data["GNN_Count"].append(payload.action_stats[node_id]["gnn_count"])
        action_df_data["Encoding_Count"].append(payload.action_stats[node_id]["encoding_count"])
    pd.DataFrame(action_df_data).to_csv(output_path, index=False)


def _write_coding_ratios(output_path: Path, payload: CsvOutputPayload) -> None:
    """Write relay coding versus non-coding ratios."""
    df_data = {
        "Node_ID": [],
        "Coding": [],
        "Non-Coding": [],
        "Ratio": [],
    }
    for node_id in range(1, payload.node_num - 1):
        coding_count = payload.node_coding_stats[node_id]["Coding"]
        non_coding_count = payload.node_coding_stats[node_id]["Non-Coding"]
        total = coding_count + non_coding_count
        ratio = coding_count / total if total > 0 else 0
        df_data["Node_ID"].append(f"Node_{node_id}")
        df_data["Coding"].append(coding_count)
        df_data["Non-Coding"].append(non_coding_count)
        df_data["Ratio"].append(ratio)
    pd.DataFrame(df_data).to_csv(output_path, index=False)


def _write_gnn_decisions(output_path: Path, payload: CsvOutputPayload) -> None:
    """Write aggregate GNN decision distribution."""
    total_gnn_decisions = payload.gnn_decision_stats["output_1_count"] + payload.gnn_decision_stats["output_0_count"]
    output_1_percentage = 0 if total_gnn_decisions == 0 else (payload.gnn_decision_stats["output_1_count"] / total_gnn_decisions) * 100
    output_0_percentage = 0 if total_gnn_decisions == 0 else (payload.gnn_decision_stats["output_0_count"] / total_gnn_decisions) * 100
    pd.DataFrame(
        {
            "Decision": ["Output_1", "Output_0", "Total"],
            "Count": [payload.gnn_decision_stats["output_1_count"], payload.gnn_decision_stats["output_0_count"], total_gnn_decisions],
            "Percentage": [output_1_percentage, output_0_percentage, 100.0],
        }
    ).to_csv(output_path, index=False)


def _build_time_stats(payload: CsvOutputPayload) -> tuple[dict[str, list[Any]], list[float], list[float]]:
    """Build the tabular inference-time summary data."""
    source_times = payload.time_stats["source"]
    time_stats_data = {
        "Node_Type": ["Source"],
        "Node_ID": [0],
        "Total_Inference_Time_s": [np.sum(source_times) if source_times else 0],
        "Avg_Inference_Time_s": [np.mean(source_times) if source_times else 0],
        "Std_Inference_Time_s": [np.std(source_times) if source_times else 0],
    }

    relay_avg_times: list[float] = []
    relay_total_times: list[float] = []
    for relay_id in sorted(payload.time_stats["relay"].keys()):
        relay_times = payload.time_stats["relay"][relay_id]
        total_time = np.sum(relay_times)
        avg_time = np.mean(relay_times)
        std_time = np.std(relay_times)
        time_stats_data["Node_Type"].append("Relay")
        time_stats_data["Node_ID"].append(relay_id)
        time_stats_data["Total_Inference_Time_s"].append(total_time)
        time_stats_data["Avg_Inference_Time_s"].append(avg_time)
        time_stats_data["Std_Inference_Time_s"].append(std_time)
        relay_avg_times.append(avg_time)
        relay_total_times.append(total_time)

    if relay_avg_times:
        time_stats_data["Node_Type"].append("All_Relays")
        time_stats_data["Node_ID"].append(-1)
        time_stats_data["Total_Inference_Time_s"].append(np.sum(relay_total_times))
        time_stats_data["Avg_Inference_Time_s"].append(np.mean(relay_avg_times))
        time_stats_data["Std_Inference_Time_s"].append(np.std(relay_avg_times))

    all_times = payload.time_stats["source"].copy()
    for relay_id in payload.time_stats["relay"]:
        all_times.extend(payload.time_stats["relay"][relay_id])

    if all_times:
        time_stats_data["Node_Type"].append("All_Nodes")
        time_stats_data["Node_ID"].append(-2)
        time_stats_data["Total_Inference_Time_s"].append(np.sum(all_times))
        time_stats_data["Avg_Inference_Time_s"].append(np.mean(all_times))
        time_stats_data["Std_Inference_Time_s"].append(np.std(all_times))

    return time_stats_data, relay_avg_times, all_times


def _print_time_stats(episode_idx: int, time_stats: dict[str, Any], relay_avg_times: list[float], all_times: list[float]) -> None:
    """Print human-readable inference-time statistics in English."""
    print(f"\nInference time statistics (Episode {episode_idx}):")
    print(f"Average source-node inference time: {(np.mean(time_stats['source']) if time_stats['source'] else 0):.6f} s")
    if relay_avg_times:
        print(f"Average relay-node inference time: {np.mean(relay_avg_times):.6f} s")
    if all_times:
        print(f"Average inference time across all nodes: {np.mean(all_times):.6f} s")


def _write_decode_delay_stats(output_path: Path, payload: CsvOutputPayload, all_times: list[float]) -> None:
    """Write end-to-end decode-delay statistics."""
    total_inference_time = np.sum(all_times) if all_times else 0
    avg_network_decode_delay = np.mean(payload.decode_delay_stats) if payload.decode_delay_stats else 0
    avg_decode_compute_delay = np.mean(payload.generation_decode_compute_delays) if payload.generation_decode_compute_delays else 0
    pd.DataFrame(
        {
            "Source_ID": [payload.source_id],
            "Destination_ID": [payload.node_num - 1],
            "Delay_Unit": ["s"],
            "Timing_Model": ["outer_tdma_slots_with_idle_time"],
            "Feedback_Assumption": ["explicit_control_packet_delay_no_tdma_feedback_slot"],
            "Inference_Included_In_Network_Delay": [payload.include_inference_delay_in_sim_time],
            "Decode_Compute_Included_In_End_To_End_Delay": [payload.include_decode_compute_delay_in_sim_time],
            "Decode_Compute_Model": ["measured_wall_time" if payload.use_measured_decode_delay else "coefficient_times_K_cubed"],
            "Decode_Count": [len(payload.decode_delay_stats)],
            "Avg_Decode_Compute_Delay_s": [avg_decode_compute_delay],
            "Std_Decode_Compute_Delay_s": [np.std(payload.generation_decode_compute_delays) if payload.generation_decode_compute_delays else 0],
            "Avg_Network_Decode_Delay_s": [avg_network_decode_delay],
            "Std_Network_Decode_Delay_s": [np.std(payload.decode_delay_stats) if payload.decode_delay_stats else 0],
            "Min_Network_Decode_Delay_s": [np.min(payload.decode_delay_stats) if payload.decode_delay_stats else 0],
            "Max_Network_Decode_Delay_s": [np.max(payload.decode_delay_stats) if payload.decode_delay_stats else 0],
            "Total_Inference_Time_s": [total_inference_time],
            "Avg_End_To_End_Delay_With_Total_Inference_s": [avg_network_decode_delay + total_inference_time],
        }
    ).to_csv(output_path, index=False)

    print("avg_network_decode_delay(s):", f"{payload.source_id}-{payload.node_num - 1}", round(avg_network_decode_delay, 6))
