"""Load simulation and training settings from the project config modules."""

import config
import config_topology


def load_config():
    """Return a flat dictionary that combines runtime and topology settings."""
    return {
        'EPISODES': config.EPISODES,
        'train_eval_interval': getattr(config, 'TRAIN_EVAL_INTERVAL', 1),
        'Max_test': config.Max_test,
        'K': config.K,
        'M': config.M,
        'device': config.device,
        'batch_size': config.batch_size,
        'extrinsic_reward': config.extrinsic_reward,
        'epsilon': config.epsilon,
        'epsilon_min': config.epsilon_min,
        'epsilon_period': config.epsilon_period,
        'parallel_path': config_topology.parallel_path,
        'max_nb': config_topology.max_nb,
        'fc1_features': config.fc1_features,
        'fc1_path_features': config.fc1_path_features,
        'fc2_features': config.fc2_features,
        'fc2_path_features': config.fc2_path_features,
        'action_size': config.action_size,
        'max_buffer_size': config.max_buffer_size,
        'learning_rate': config.learning_rate,
        'gamma': config.gamma,
        'hidden_dim': config.hidden_dim,

        'Max_s_f': config.Max_s_f,
        'inital_epsilon': config.inital_epsilon,
        'restart_max': config.restart_max,
        'include_inference_delay_in_sim_time': getattr(config, 'INCLUDE_INFERENCE_DELAY_IN_SIM_TIME', False),
        'inference_delay_time_unit': getattr(config, 'INFERENCE_DELAY_TIME_UNIT', getattr(config, 'UNIT', 1e6)),
        'include_decode_compute_delay_in_sim_time': getattr(config, 'INCLUDE_DECODE_COMPUTE_DELAY_IN_SIM_TIME', True),
        'decode_compute_delay_coefficient_us': getattr(config, 'DECODE_COMPUTE_DELAY_COEFFICIENT_US', 1.0),
        'use_measured_decode_delay': getattr(config, 'USE_MEASURED_DECODE_DELAY', False),
        'enable_relay_coding_selection': getattr(config, 'ENABLE_RELAY_CODING_SELECTION', True),
        'relay_coding_mode': getattr(config, 'RELAY_CODING_MODE', 'auto'),
        'slot_duration': getattr(config, 'SLOT_DURATION', getattr(config, 'slot_duration', 1.0)),
        'frame_slot': getattr(config, 'FRAME_SLOT', None),
        'slot_list': getattr(config, 'SLOT_LIST', None),
        'packet_payload_length': getattr(config, 'PACKET_PAYLOAD_LENGTH', config.K),
        'coefficient_bits': getattr(config, 'COEFFICIENT_BITS', 1),
        'coding_vector_length': getattr(config, 'CODING_VECTOR_LENGTH', config.K * getattr(config, 'COEFFICIENT_BITS', 1)),
        'packet_header_length': getattr(config, 'PACKET_HEADER_LENGTH', 0),
        'packet_length': getattr(config, 'PACKET_LENGTH', config.K),
        'mac_header_length': getattr(config, 'MAC_HEADER_LENGTH', 0),
        'phy_header_length': getattr(config, 'PHY_HEADER_LENGTH', 0),
        'control_packet_length': getattr(config, 'CONTROL_PACKET_LENGTH', 128),
        'control_mac_header_length': getattr(config, 'CONTROL_MAC_HEADER_LENGTH', getattr(config, 'MAC_HEADER_LENGTH', 0)),
        'control_phy_header_length': getattr(config, 'CONTROL_PHY_HEADER_LENGTH', getattr(config, 'PHY_HEADER_LENGTH', 0)),
        'bit_transmission_time': getattr(config, 'BIT_TRANSMISSION_TIME', 1.0),
        'bit_transport_time': getattr(config, 'BIT_TRANSPORT_TIME', 0.0),
        'propagation_guard_time': getattr(config, 'PROPAGATION_GUARD_TIME', 0.0),
        # Node coordinates are sourced from topology definition (meters).
        'node_positions': getattr(config_topology, 'node_positions', getattr(config, 'NODE_POSITION', None)),
        'node_num': config_topology.node_num,
        'neighbor_matrix': config_topology.neighbor_matrix,
        'links': config_topology.links,

    }