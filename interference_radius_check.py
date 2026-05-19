"""Check TDMA-slot interference relationships for the configured topology.

Run directly from the AttenNC_PFC directory or project root:
    python AttenNC_PFC/interference_radius_check.py

The script computes an interference radius from a Friis-style threshold model,
then checks whether nodes that can transmit in the same TDMA slot are close
enough to interfere with each receiver in config_topology.py.
"""

import argparse
import math
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from config import BAND_WIDTH, FRAME_SLOT, NODE_POSITION
from config_topology import links, neighbor_matrix, node_num, node_positions

# Keep these constants local because AttenNC_PFC/config.py did not define them yet.
# They match FQF/config.py.
NOISE_STRENGTH = -174.0  # dBm/Hz
TRANSMITTING_POWER = 8  # dBm
GAIN_T = 1
GAIN_R = 1
FREQUENCY = 2.4e9  # Hz


def dbm_to_mw(power_dbm):
    """Convert dBm to mW."""
    return 10 ** (power_dbm / 10.0)


def mw_to_w(power_mw):
    """Convert mW to W."""
    return power_mw / 1000.0


def calculate_interference_distance(
    transmitting_power_dbm,
    frequency_hz,
    gain_t,
    gain_r,
    interference_threshold_w,
):
    """Calculate the interference radius using a Friis-style model.

    If received interference power is greater than or equal to
    ``interference_threshold_w``, the transmitting node is considered an
    interferer. This function solves that threshold equation for distance.
    """
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive.")
    if interference_threshold_w <= 0:
        raise ValueError("interference_threshold_w must be positive.")

    speed_of_light = 3e8
    transmitting_power_w = mw_to_w(dbm_to_mw(transmitting_power_dbm))
    wavelength = speed_of_light / frequency_hz
    return (wavelength / (4 * math.pi)) * math.sqrt(
        (transmitting_power_w * gain_t * gain_r) / interference_threshold_w
    )


def distance(node_a, node_b, positions):
    """Calculate Euclidean distance between two configured nodes."""
    pos_a = np.asarray(positions[node_a], dtype=float)
    pos_b = np.asarray(positions[node_b], dtype=float)
    return float(np.linalg.norm(pos_a - pos_b))


def noise_threshold_w(noise_strength_dbm_per_hz, bandwidth_hz, multiplier=1.0):
    """Convert thermal-noise density and bandwidth into a W threshold."""
    noise_mw = dbm_to_mw(noise_strength_dbm_per_hz) * bandwidth_hz
    return mw_to_w(noise_mw) * multiplier


def normalize_frame_slot(frame_slot, total_nodes):
    """Normalize FRAME_SLOT to node_id -> list[int]."""
    normalized = {}
    for node_id in range(total_nodes):
        slots = frame_slot.get(node_id, 0)
        if isinstance(slots, (list, tuple, set)):
            normalized[node_id] = sorted(int(slot) for slot in slots)
        else:
            normalized[node_id] = [int(slots)]
    return normalized


def get_directed_links(matrix):
    """Return all directed edges enabled by neighbor_matrix."""
    return [(src, dst) for src in range(matrix.shape[0]) for dst in range(matrix.shape[1]) if matrix[src][dst] == 1]


def find_topology_interference_pairs(positions, radius):
    """Find all node pairs whose distance is within the interference radius."""
    records = []
    for node_a in range(node_num):
        for node_b in range(node_a + 1, node_num):
            pair_distance = distance(node_a, node_b, positions)
            if pair_distance <= radius:
                records.append(
                    {
                        "node_a": node_a,
                        "node_b": node_b,
                        "distance": pair_distance,
                    }
                )
    return records


def check_interference(positions, frame_slot, radius):
    """Check same-slot transmitter/receiver interference relationships."""
    normalized_slots = normalize_frame_slot(frame_slot, node_num)
    directed_links = get_directed_links(neighbor_matrix)
    slot_to_links = {}
    for src, dst in directed_links:
        for slot in normalized_slots.get(src, []):
            if slot > 0:
                slot_to_links.setdefault(slot, []).append((src, dst))

    interference_records = []
    for slot, slot_links in sorted(slot_to_links.items()):
        transmitters = sorted({src for src, _ in slot_links})
        for victim_src, victim_dst in slot_links:
            for interferer_src in transmitters:
                if interferer_src == victim_src:
                    continue
                dist_to_receiver = distance(interferer_src, victim_dst, positions)
                if dist_to_receiver <= radius:
                    interference_records.append(
                        {
                            "slot": slot,
                            "tx": interferer_src,
                            "rx": victim_dst,
                            "victim_link": (victim_src, victim_dst),
                            "distance": dist_to_receiver,
                        }
                    )
    return slot_to_links, interference_records


def print_distance_matrix(positions):
    """Print the pairwise node-distance matrix in meters."""
    print("\nNode distance matrix (m):")
    for i in range(node_num):
        row = []
        for j in range(node_num):
            row.append(f"{distance(i, j, positions):7.2f}")
        print(f"node {i}: " + " ".join(row))


def main():
    """Run the interference-radius analysis for the configured topology."""
    parser = argparse.ArgumentParser(description="Check interference radius relationships for AttenNC_PFC topology.")
    parser.add_argument("--frequency", type=float, default=FREQUENCY, help="Carrier frequency in Hz.")
    parser.add_argument("--power-dbm", type=float, default=TRANSMITTING_POWER, help="Transmit power in dBm.")
    parser.add_argument("--gain-t", type=float, default=GAIN_T, help="Transmitter antenna gain.")
    parser.add_argument("--gain-r", type=float, default=GAIN_R, help="Receiver antenna gain.")
    parser.add_argument(
        "--threshold-w",
        type=float,
        default=None,
        help="Interference threshold in W. If omitted, use noise power from NOISE_STRENGTH * BAND_WIDTH.",
    )
    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to the default noise-based threshold.",
    )
    parser.add_argument("--show-distance-matrix", action="store_true", help="Print all pairwise node distances.")
    args = parser.parse_args()

    positions = node_positions or NODE_POSITION
    threshold_w = args.threshold_w
    if threshold_w is None:
        threshold_w = noise_threshold_w(NOISE_STRENGTH, BAND_WIDTH, args.threshold_multiplier)

    radius = calculate_interference_distance(
        args.power_dbm,
        args.frequency,
        args.gain_t,
        args.gain_r,
        threshold_w,
    )

    print("Interference-radius check:")
    print(f"  Transmit power: {args.power_dbm} dBm")
    print(f"  Frequency: {args.frequency} Hz")
    print(f"  Threshold: {threshold_w:.6e} W")
    print(f"  Interference radius: {radius:.2f} m")
    print("\nTDMA slot configuration:")
    for node_id, slots in normalize_frame_slot(FRAME_SLOT, node_num).items():
        print(f"  node {node_id}: slots {slots}")

    if args.show_distance_matrix:
        print_distance_matrix(positions)

    topology_interference_pairs = find_topology_interference_pairs(positions, radius)
    print("\nTopology-level interference pairs (based only on node positions and radius):")
    if not topology_interference_pairs:
        print("  No node pairs were found within the interference radius.")
    else:
        for record in topology_interference_pairs:
            print(
                f"  node {record['node_a']} <-> node {record['node_b']}: "
                f"distance {record['distance']:.2f} m <= interference radius {radius:.2f} m, so they may interfere"
            )
        print(f"  Found {len(topology_interference_pairs)} potentially interfering node pairs.")

    slot_to_links, interference_records = check_interference(positions, FRAME_SLOT, radius)

    print("\nLinks scheduled by slot:")
    for slot, slot_links in sorted(slot_to_links.items()):
        print(f"  slot {slot}: {slot_links}")

    print("\nSame-slot link-interference results:")
    if not interference_records:
        print("  No transmitter-receiver pairs were found within the interference radius in the same slot.")
    else:
        for record in interference_records:
            victim_src, victim_dst = record["victim_link"]
            link_quality = links[victim_src][victim_dst]
            print(
                f"  slot {record['slot']}: "
                f"transmitter {record['tx']} may interfere with receiver {record['rx']}; "
                f"affected link {victim_src}->{victim_dst}, "
                f"distance {record['distance']:.2f} m <= {radius:.2f} m, "
                f"original link success rate links[{victim_src}][{victim_dst}]={link_quality:.2f}"
            )


if __name__ == "__main__":
    main()
