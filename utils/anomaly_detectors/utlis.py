import numpy as np


def mitigate_false_positive(y_pred, anomaly_score, change_threshold):
    """
    Pruning the False Positive values after the "Adaptive Threshold Detector".
    """
    sequences = []  # List of (start_position, idx, max_element, delete_sequence)
    delete_sequence, start_position, end_position = 0, 0, 0

    max_element = anomaly_score[0]
    for idx in range(1, len(y_pred)):
        if idx == len(y_pred) - 1:
            sequences.append([start_position, idx, max_element, delete_sequence])
        elif y_pred[idx] == 1 and y_pred[idx + 1] == 0:
            sequences.append(
                [start_position, end_position, max_element, delete_sequence]
            )
            end_position = idx
        elif y_pred[idx] == 1 and y_pred[idx - 1] == 0:
            max_element = anomaly_score[idx]
            start_position = idx
        if (
            y_pred[idx] == 1
            and y_pred[idx - 1] == 1
            and anomaly_score[idx] > max_element
        ):
            max_element = anomaly_score[idx]

    max_elements = np.sort(np.array([sequence[2] for sequence in sequences]))[::-1]
    change_percent = abs(max_elements[1:] - max_elements[:-1]) / max_elements[1:]

    # Add 0 for the 1st element which is not change percent
    delete_seq = np.append(0, change_percent < change_threshold)

    # Map maximum element and sequences
    for i, element in enumerate(max_elements):
        for j in range(0, len(sequences)):
            if sequences[j][2] == element:
                sequences[j][3] = delete_seq[i]

    # Flag the sequence as normal
    for sequence in sequences:
        if sequence[3] == 1:
            y_pred[sequence[0] : sequence[1] + 1] = [0] * (
                sequence[1] - sequence[0] + 1
            )

    return y_pred
