from scipy.signal import find_peaks
import numpy as np


def merge_sequences(
    sequences: list[tuple[int, int]], max_frame_gap: int = 10
) -> list[tuple[int, int]]:
    """Merges cyclic sections if the distance between them does not exceed max_frame_gap frames.

    Args:
        sequences: List of tuples (start_frame, end_frame) for cyclic sections.
        max_frame_gap: Maximum distance between sections (in frames) for merging them.

    Returns:
        List of merged tuples (start_frame, end_frame).
    """
    if not sequences:
        return []

    # Sorting by start
    sequences = sorted(sequences, key=lambda x: x[0])
    merged = []
    current_start, current_end = sequences[0]

    for start, end in sequences[1:]:
        if start <= current_end + max_frame_gap:
            # Sections overlap or are within max_frame_gap, updating end
            current_end = max(current_end, end)
        else:
            # New section, adding previous and starting new
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    # Adding the last merged section
    merged.append((current_start, current_end))
    return merged


def find_cyclic_sequences(
    positions: list[list],
    min_cycle_amplitude: float = 30.0,  # Minimum amplitude of one cycle (range)
    max_amplitude_variation: float = 50.0,  # Max amplitude variation between cycles
    min_num_amplitudes: int = 4,  # Min number of amplitudes for a sequence (~2 cycles)
) -> list[tuple[int, int]]:
    """Finds sections with regular cyclic ball movements (≥2 cycles),
    where oscillation amplitudes differ by no more than max_amplitude_variation.
    Improved for detection of local stable cycles (e.g., ball bouncing before serving),
    even if total amplitude variation is large - looking for subsequences.

    Args:
        positions: List of positions in format [[x, y], frame].
        min_cycle_amplitude: Minimum Y range to recognize cycles as significant.
        max_amplitude_variation: Maximum difference between cycle amplitudes.
        min_num_amplitudes: Minimum number of consecutive amplitudes for a sequence.

    Returns:
        List of tuples (start_frame, end_frame) for stable cyclic sections.
    """
    if not positions or len(positions) < 10:
        return []

    # Convert to array
    pos_array = np.array(
        [(pos[0][0], pos[0][1], pos[1]) for pos in positions], dtype=np.float64
    )
    x_values = pos_array[:, 0]
    y_values = pos_array[:, 1]
    frames = pos_array[:, 2].astype(int)

    sequences = []
    i = 0
    n = len(pos_array)

    while i < n - 10:
        start_idx = i
        j = i + 1

        # Looking for a section with small X change
        while j < n:
            x_range = np.max(x_values[i : j + 1]) - np.min(x_values[i : j + 1])
            if x_range > 150:
                break
            j += 1

        if j - i < 100:  # section too short
            i = j
            continue

        y_segment = y_values[i:j]
        total_y_range = np.max(y_segment) - np.min(y_segment)
        if total_y_range < min_cycle_amplitude:
            i = j
            continue

        # Finding peaks and valleys
        peaks, _ = find_peaks(y_segment, prominence=10)
        troughs, _ = find_peaks(-y_segment, prominence=10)

        if len(peaks) < 2 or len(troughs) < 2:
            i = j
            continue

        # Sorting events by index
        events = sorted(
            [(p, "peak") for p in peaks] + [(t, "trough") for t in troughs],
            key=lambda x: x[0],
        )

        # Extracting amplitudes (all, no filter yet)
        amplitudes = []
        for k in range(1, len(events)):
            prev_idx, _ = events[k - 1]
            curr_idx, _ = events[k]
            amplitude = abs(y_segment[curr_idx] - y_segment[prev_idx])
            amplitudes.append(amplitude)

        if len(amplitudes) < min_num_amplitudes:
            i = j
            continue

        # Step 1: Finding 'good' amplitude segments where all >= min_cycle_amplitude (no small transitions)
        good_segments = []
        amp_idx = 0
        while amp_idx < len(amplitudes):
            if amplitudes[amp_idx] < min_cycle_amplitude:
                amp_idx += 1
                continue
            amp_j = amp_idx
            while amp_j < len(amplitudes) and amplitudes[amp_j] >= min_cycle_amplitude:
                amp_j += 1
            if amp_j - amp_idx >= min_num_amplitudes:
                good_segments.append((amp_idx, amp_j))
            amp_idx = amp_j

        # Step 2: For each good segment, looking for subsequences with similar amplitudes (range <= var)
        for amp_start, amp_end in good_segments:
            left = amp_start
            for right in range(amp_start, amp_end):
                sub = amplitudes[left : right + 1]
                sub_min = min(sub)
                sub_max = max(sub)
                while (sub_max - sub_min > max_amplitude_variation) and left <= right:
                    left += 1
                    sub = amplitudes[left : right + 1]
                    if sub:
                        sub_min = min(sub)
                        sub_max = max(sub)
                if right - left + 1 >= min_num_amplitudes:
                    # Adding section (from event left to event right+1)
                    event_left = events[left][0]
                    event_right = events[right + 1][0]
                    f_start = int(frames[i + event_left])
                    f_end = int(frames[i + event_right])
                    sequences.append((f_start, f_end))
                    # Moving to the next non-overlapping
                    left = right + 1

        i = j  # moving to the next segment
    sequences = merge_sequences(sequences)

    return sequences


def find_rolling_sequences(
    positions: list[list],
    max_y_range: float = 40.0,  # Reduced for track 0005
    min_x_range: float = 50.0,
    min_length: int = 70,  # Reduced for short sections
) -> list[tuple[int, int]]:
    """Finds sections where the ball is rolling on the floor (small Y range, large X range)."""
    if not positions or len(positions) < min_length:
        return []

    pos_array = np.array(
        [(pos[0][0], pos[0][1], pos[1]) for pos in positions], dtype=np.float64
    )
    x_values = pos_array[:, 0]
    y_values = pos_array[:, 1]
    frames = pos_array[:, 2].astype(int)

    sequences = []
    i = 0
    n = len(pos_array)

    while i < n - min_length + 1:
        j = i + min_length - 1
        while j < n:
            y_range = np.max(y_values[i : j + 1]) - np.min(y_values[i : j + 1])
            x_range = np.max(x_values[i : j + 1]) - np.min(x_values[i : j + 1])
            if y_range <= max_y_range and x_range >= min_x_range:
                j += 1
            else:
                break
        if j - i >= min_length:
            sequences.append((int(frames[i]), int(frames[j - 1])))
        i += 1

    sequences = merge_sequences(sequences, max_frame_gap=30)

    return sequences
