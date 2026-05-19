"""CSV helpers for aggregating evaluation metrics and packet counts."""

import csv
import os

import numpy as np


def std_dev(data):
    """Return the standard deviation of a numeric sequence."""
    return np.std(data)


def ensure_folder_exists(folder_path):
    """Create the output folder when it does not already exist."""
    os.makedirs(folder_path, exist_ok=True)


def write_counts_to_csv(source_send_count_list, file_path, file_name):
    """Write per-test source transmission counts to a CSV file."""
    ensure_folder_exists(file_path)
    csv_file_path = os.path.join(file_path, file_name)
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows([[count] for count in source_send_count_list])


def calculate_decode_probability(source_send_count_list):
    """Return decode-success probability for each send-count threshold from 1 to 60."""
    decode_probability = []
    for i in range(1, 61):
        decode_success_count = sum(1 for j in source_send_count_list if j <= i)
        decode_probability.append(decode_success_count / len(source_send_count_list))
    return decode_probability


def write_results_to_csv(name, decode_probabilities, avg_overhead, std_dev_count, avg_s_f, data_folder):
    """Write aggregated decode probability and summary metrics to CSV."""
    filename = os.path.join(data_folder, f'{name}.csv')
    ensure_folder_exists(data_folder)
    overhead_row_index = 1
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sent packet', 'Probability', 'avg_overhead', 'std_dev', 'avg_s_f'])
        for index, prob in enumerate(decode_probabilities, start=1):
            if index == overhead_row_index:
                writer.writerow([index, prob, avg_overhead, std_dev_count, avg_s_f])
            else:
                writer.writerow([index, prob, ''])
