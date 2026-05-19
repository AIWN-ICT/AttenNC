"""Project-wide training, MAC/PHY, and timing configuration defaults."""

import torch

EPISODES = 1
Max_test = 1000
K = 5
M = 3
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
batch_size = 32
extrinsic_reward = 1
epsilon = 1
fc1_features = 256
fc1_path_features = 256

fc2_features = 32
fc2_path_features = 32
max_buffer_size = 10000
epsilon_period = 500

learning_rate = 0.0001
gamma = 0.99
hidden_dim = 64

epsilon_min = 0.01

action_size = 2
Max_s_f = 50
inital_epsilon = 1
restart_max = 30

# ---------------------------- Time ----------------------------- #
# Align with FQF: the simulation time unit is microseconds (us).
UNIT = 1e6

# Whether to charge source/relay inference latency into simulated network time.
# Keep this disabled by default so network decode delay and algorithm inference
# delay are reported separately and remain comparable across machines.
INCLUDE_INFERENCE_DELAY_IN_SIM_TIME = False
INFERENCE_DELAY_TIME_UNIT = UNIT  # Convert measured seconds to simulation microseconds.

# Whether to charge destination-side decoding computation latency into end-to-end delay.
INCLUDE_DECODE_COMPUTE_DELAY_IN_SIM_TIME = True
# Approximate GF(2) decoding cost as c * K^3 when measured decoding latency is not used.
DECODE_COMPUTE_DELAY_COEFFICIENT_US = 1.0
# If True, use wall-clock rank-check/decoding measurement converted to simulation time.
USE_MEASURED_DECODE_DELAY = False

# ---------------------- Relay mode switch --------------------- #
# Enable/disable relay coding-node selection gate (GNN decision):
# - True: enable coding-node selection (default behavior)
# - False: disable selection and always forward with coding path
ENABLE_RELAY_CODING_SELECTION = True

# -------------------------- MAC layer -------------------------- #
# Manual TDMA slot settings, following the same style as FQF.
# FRAME_SLOT maps node_id -> slot index. Nodes with slot 0 do not actively send.
FRAME_SLOT = {
    0: [1],
    1: [2],
    2: [3],
    3: [4],
    4: [5],
    5: [6, 7],
    6: [8, 9],
    7: [0],
}
SLOT_DURATION = 3000  # us
# Keep backward compatibility for legacy lowercase config key consumers.
slot_duration = SLOT_DURATION

# Explicit ACK/NACK control-frame timing. Control packets are delivered after
# their own MAC/PHY transmission plus propagation delay instead of instantly.
CONTROL_PACKET_LENGTH = 128  # bit
CONTROL_MAC_HEADER_LENGTH = 32  # bit
CONTROL_PHY_HEADER_LENGTH = 32  # bit

# -------------------------- PHY layer -------------------------- #
# Align with FQF PHY/MAC timing and unit definitions.
# A network-coded packet carries both the coded payload and its coding-vector
# coefficients. The PHY packet length therefore explicitly includes K coding
# coefficients in addition to the application payload and packet header.
PACKET_HEADER_LENGTH = 128  # bit
PACKET_PAYLOAD_LENGTH = 1000 * 8  # bit
COEFFICIENT_BITS = 8  # bit per coding coefficient, e.g. GF(2^8)
CODING_VECTOR_LENGTH = K * COEFFICIENT_BITS  # bit
PACKET_LENGTH = PACKET_PAYLOAD_LENGTH + CODING_VECTOR_LENGTH + PACKET_HEADER_LENGTH
MAC_HEADER_LENGTH = 32  # bit
PHY_HEADER_LENGTH = 32  # bit
BAND_WIDTH = 4 * UNIT  # Hz
BIT_RATE = BAND_WIDTH * 1  # BPSK: 1 bps/Hz
BIT_TRANSMISSION_TIME = 1 / BIT_RATE * 1e6  # us/bit
BIT_TRANSPORT_TIME = 1 / 3e8 * UNIT  # us/m
# Match FQF's "+1 us" propagation offset.
PROPAGATION_GUARD_TIME = 1.0

# Optional fallback positions (meters) for propagation delay calculation.
# The primary source remains config_topology.node_positions.
NODE_POSITION = {
    0: [0, 0],
    1: [100, 100],
    2: [100, 0],
    3: [100, -100],
    4: [200, 100],
    5: [200, 0],
    6: [200, -100],
    7: [300, 0],
}
