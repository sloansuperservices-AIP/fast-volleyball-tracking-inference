import numpy as np
from scipy.signal import find_peaks


def merge_sequences(sequences: list[tuple[int, int]], max_frame_gap: int = 10) -> list[tuple[int, int]]:
    """Merges cyclic intervals if the gap between them is <= max_frame_gap."""
    if not sequences:
        return []
    sequences = sorted(sequences, key=lambda x: x[0])
    merged = []
    current_start, current_end = sequences[0]
    for start, end in sequences[1:]:
        if start <= current_end + max_frame_gap:
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))
    return merged


def find_cyclic_sequences(
    positions: list[list],
    min_cycle_amplitude: float = 30.0,
    max_amplitude_variation: float = 50.0,
    min_num_amplitudes: int = 4,
) -> list[tuple[int, int]]:
    """Finds stable cyclic movements (e.g. ball bouncing before service)."""
    if not positions or len(positions) < 10:
        return []
    pos_array = np.array([(pos[0][0], pos[0][1], pos[1]) for pos in positions], dtype=np.float64)
    x_values, y_values, frames = pos_array[:, 0], pos_array[:, 1], pos_array[:, 2].astype(int)
    sequences = []
    i, n = 0, len(pos_array)
    while i < n - 10:
        j = i + 1
        while j < n:
            if np.max(x_values[i : j + 1]) - np.min(x_values[i : j + 1]) > 150:
                break
            j += 1
        if j - i < 100:
            i = j
            continue
        y_segment = y_values[i:j]
        if np.max(y_segment) - np.min(y_segment) < min_cycle_amplitude:
            i = j
            continue
        peaks, _ = find_peaks(y_segment, prominence=10)
        troughs, _ = find_peaks(-y_segment, prominence=10)
        if len(peaks) < 2 or len(troughs) < 2:
            i = j
            continue
        events = sorted([(p, "peak") for p in peaks] + [(t, "trough") for t in troughs], key=lambda x: x[0])
        amplitudes = [abs(y_segment[events[k][0]] - y_segment[events[k - 1][0]]) for k in range(1, len(events))]
        if len(amplitudes) < min_num_amplitudes:
            i = j
            continue
        amp_idx = 0
        while amp_idx < len(amplitudes):
            if amplitudes[amp_idx] < min_cycle_amplitude:
                amp_idx += 1
                continue
            amp_j = amp_idx
            while amp_j < len(amplitudes) and amplitudes[amp_j] >= min_cycle_amplitude:
                amp_j += 1
            if amp_j - amp_idx >= min_num_amplitudes:
                left = amp_idx
                for right in range(amp_idx, amp_j):
                    sub = amplitudes[left : right + 1]
                    while sub and (max(sub) - min(sub) > max_amplitude_variation):
                        left += 1
                        sub = amplitudes[left : right + 1]
                    if right - left + 1 >= min_num_amplitudes:
                        sequences.append((int(frames[i + events[left][0]]), int(frames[i + events[right + 1][0]])))
                amp_idx = amp_j
            else:
                amp_idx += 1
        i = j
    return merge_sequences(sequences)


def find_rolling_sequences(
    positions: list[list], max_y_range: float = 40.0, min_x_range: float = 50.0, min_length: int = 70
) -> list[tuple[int, int]]:
    """Finds sequences where the ball is rolling on the floor."""
    if not positions or len(positions) < min_length:
        return []
    pos_array = np.array([(pos[0][0], pos[0][1], pos[1]) for pos in positions], dtype=np.float64)
    x_values, y_values, frames = pos_array[:, 0], pos_array[:, 1], pos_array[:, 2].astype(int)
    sequences = []
    i, n = 0, len(pos_array)
    while i < n - min_length + 1:
        j = i + min_length - 1
        while j < n:
            if (np.max(y_values[i : j + 1]) - np.min(y_values[i : j + 1]) <= max_y_range) and (
                np.max(x_values[i : j + 1]) - np.min(x_values[i : j + 1]) >= min_x_range
            ):
                j += 1
            else:
                break
        if j - i >= min_length:
            sequences.append((int(frames[i]), int(frames[j - 1])))
        i += 1
    return merge_sequences(sequences, max_frame_gap=30)
