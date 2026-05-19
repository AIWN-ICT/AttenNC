"""Static topology, link quality, and node-position configuration."""

import numpy as np

# Example topology with 8 nodes and 3 parallel source paths.
node_num = 8
parallel_path = 3
max_nb = 2
neighbor_matrix = np.zeros((node_num, node_num))
neighbor_matrix[0][1] = 1
neighbor_matrix[0][2] = 1
neighbor_matrix[0][3] = 1
neighbor_matrix[1][4] = 1
neighbor_matrix[1][5] = 1
neighbor_matrix[2][5] = 1
neighbor_matrix[2][6] = 1
neighbor_matrix[3][6] = 1
neighbor_matrix[4][7] = 1
neighbor_matrix[5][7] = 1
neighbor_matrix[6][7] = 1
links = np.zeros((node_num, node_num))
links[0][1] = 0.97
links[0][2] = 0.96
links[0][3] = 0.97
links[1][4] = 0.95
links[1][5] = 0.96
links[2][5] = 0.96
links[2][6] = 0.97
links[3][6] = 0.98
links[4][7] = 0.97
links[5][7] = 0.95
links[6][7] = 0.96

# Node coordinates for propagation-delay calculation (unit: meters).
node_positions = {
    0: [0, 0],
    1: [100, 100],
    2: [100, 0],
    3: [100, -100],
    4: [200, 100],
    5: [200, 0],
    6: [200, -100],
    7: [300, 0],
}
