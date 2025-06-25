# src/distractor_decoder/data_io.py

import tkinter as tk
from tkinter import filedialog
import mne
import numpy as np
from typing import List, Dict, Any, Optional
import yaml
import os

class EEGDataset:
    """
    Structured EEG dataset for pipeline integration.
    Attributes:
        path: Path to the EEG file
        data: np.ndarray, shape (n_samples, n_channels)
        info: MNE Info object
    """
    def __init__(self, path: str, data: np.ndarray, info: Any):
        self.path = path
        self.data = data
        self.info = info

    def __repr__(self):
        return f"<EEGDataset path={self.path} shape={self.data.shape}>"

def select_eeg_files() -> List[str]:
    """
    Open a file dialog for the user to pick EEG GDF files.
    """
    root = tk.Tk()
    root.withdraw()
    paths = filedialog.askopenfilenames(
        title="Select EEG GDF files",
        filetypes=[("GDF files", "*.gdf"), ("All files", "*.")]
    )
    root.destroy()
    return list(paths)


def load_selected_files(file_paths: List[str]) -> List[EEGDataset]:
    """
    Read each .gdf and return a list of EEGDataset objects.
    """
    loaded = []
    for p in file_paths:
        try:
            raw = mne.io.read_raw_gdf(p, preload=True)
            data = raw.get_data().T  # (n_samples, n_channels)
            loaded.append(EEGDataset(path=p, data=data, info=raw.info))
        except Exception as e:
            print(f"[ERROR] Could not load {p}: {e}")
    return loaded

def set_params(header, config_path="../../configs/default.yaml"):
    """
    Set parameters based on header info (sampling frequency and channel labels),
    and fill in other defaults from YAML config.
    Args:
        header: A dict-like object with 'sfreq' (float) and 'ch_names' (list of str)
        config_path: Path to the YAML config file
    Returns:
        params: dict with all parameters
    Raises:
        ValueError: if required header fields or config fields are missing/invalid
    """
    # Load defaults from YAML
    try:
        with open(config_path, "r") as f:
            defaults = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"Could not load config file '{config_path}': {e}")

    params = dict(defaults)

    # Check header fields (dict-style)
    if 'sfreq' not in header:
        raise ValueError("Header is missing 'sfreq' (sampling frequency)")
    if 'ch_names' not in header:
        raise ValueError("Header is missing 'ch_names' (channel labels)")
    params['fsamp'] = header['sfreq']
    params['chan_labels'] = list(header['ch_names'])

    # Epoching
    fsamp = params['fsamp']
    try:
        epoch_samples = np.arange(int(-0.5 * fsamp) + 1, int(1.0 * fsamp) + 1)
        params['epoch_samples'] = epoch_samples
        params['epoch_time'] = epoch_samples / fsamp
        zero_idx = np.where(np.isclose(params['epoch_time'], 0))[0]
        params['epoch_onset'] = int(zero_idx[0]) if len(zero_idx) > 0 else None
    except Exception as e:
        raise RuntimeError(f"Error computing epoching parameters: {e}")

    # temp_window from YAML (in seconds)
    temp_window_sec = params.get('temp_window', None)
    if temp_window_sec is None or not (isinstance(temp_window_sec, (list, tuple)) and len(temp_window_sec) == 2):
        raise ValueError("'temp_window' must be a list of two numbers (start, end in seconds) in the YAML config.")
    try:
        temp_window = np.arange(
            int(round(temp_window_sec[0] * fsamp)),
            int(round(temp_window_sec[1] * fsamp))
        )
        params['temp_window'] = temp_window
        if params['epoch_onset'] is not None:
            params['epoch_roi'] = temp_window + params['epoch_onset']
        else:
            params['epoch_roi'] = None
    except Exception as e:
        raise RuntimeError(f"Error computing temp_window indices: {e}")

    # Resampling ratio
    if 'resample' not in params or params['resample'] is None:
        params['resample'] = {}
    try:
        params['resample']['ratio'] = int(round(fsamp / 64))
    except Exception as e:
        raise RuntimeError(f"Error computing resample ratio: {e}")

    return params

def compute_index(trigger, trigger_codes):
    """
    Identify relevant triggers and assign types.
    Args:
        trigger: 1D numpy array of trigger values
        trigger_codes: list of trigger codes to consider
    Returns:
        pos: positions of move triggers
        typ: labels: 0=neutral, 1=pos, 2=neg
    Raises:
        ValueError: if input is not 1D or trigger_codes is not a list/array
    """
    trigger = np.asarray(trigger)
    if trigger.ndim != 1:
        raise ValueError("'trigger' must be a 1D array of trigger values.")
    if not isinstance(trigger_codes, (list, tuple, np.ndarray)):
        raise ValueError("'trigger_codes' must be a list, tuple, or numpy array.")
    relevant_codes = trigger_codes
    is_relevant = np.isin(trigger, relevant_codes)
    code_idx = np.zeros_like(trigger, dtype=int)
    for i, code in enumerate(relevant_codes, 1):
        code_idx[trigger == code] = i
    index_pos = np.where(is_relevant)[0]
    code_types = code_idx[is_relevant]

    # Initialize types: 0=start, 1=pos, 2=neg
    index_typ = np.zeros_like(code_types)
    index_typ[code_types == 2] = 1  # 104 = pos feedback
    index_typ[code_types == 3] = 2  # 108 = neg feedback

    # Post-process to assign feedback to the previous move
    for i in range(len(index_typ) - 1):
        if index_typ[i] == 0:
            if index_typ[i + 1] == 1 or index_typ[i + 1] == 2:
                index_typ[i] = index_typ[i + 1]

    # Only keep move triggers (100s), now labeled as 0=neutral, 1=positive, 2=negative
    move_mask = code_types == 1  # 100 = move
    pos = index_pos[move_mask]
    typ = index_typ[move_mask]
    return pos, typ

def preprocess_dataset(dataset, params):
    """
    Preprocess a dataset according to config.
    Extracts EEG, EOG, and trigger channels, converts types, and computes trigger indices.
    Args:
        dataset: EEGDataset object with .data (samples x channels)
        params: config dictionary with channel indices and trigger codes
        fname: filename (unused, for compatibility)
        trigtype: trigger type (unused, for compatibility)
    Raises:
        ValueError: if required config fields are missing or data shapes are invalid.
    """
    # Check required config fields
    for key in ['eeg_channels', 'eog_channels', 'trigger_channel', 'trigger_codes']:
        if key not in params:
            raise ValueError(f"Missing required config field: '{key}'")
    # Check data shape
    if not hasattr(dataset, 'data') or not isinstance(dataset.data, np.ndarray):
        raise ValueError("'dataset' must have a 'data' attribute that is a numpy array.")
    n_channels = dataset.data.shape[1]
    for ch_list, name in zip([params['eeg_channels'], params['eog_channels']], ['eeg_channels', 'eog_channels']):
        if any(ch >= n_channels or ch < 0 for ch in ch_list):
            raise ValueError(f"Some indices in '{name}' are out of bounds for data with {n_channels} channels.")
    if params['trigger_channel'] >= n_channels or params['trigger_channel'] < 0:
        raise ValueError(f"'trigger_channel' index {params['trigger_channel']} is out of bounds for data with {n_channels} channels.")

    # Extract channels
    dataset.eeg = dataset.data[:, params['eeg_channels']].astype(np.float32)
    dataset.eog = dataset.data[:, params['eog_channels']].astype(np.float32)
    dataset.trigger = dataset.data[:, params['trigger_channel']].astype(np.float32)

    # Compute trigger indices
    dataset.trig = {}
    try:
        dataset.trig['pos'], dataset.trig['typ'] = compute_index(dataset.trigger, params['trigger_codes'])
    except Exception as e:
        raise RuntimeError(f"Error in compute_index: {e}")

    return dataset


from scipy.signal import butter, filtfilt
def bandpass_filter(data, params):
    """
    Apply a Butterworth bandpass filter to EEG data.
    Args:
        data: np.ndarray, shape (samples, channels)
        params: dict, must contain 'spectral_filter' with 'order', 'freqs', and 'fsamp'
    Returns:
        Filtered data (same shape as input)
    Raises:
        ValueError: if parameters are missing or invalid
        RuntimeError: if filtering fails
    """
    # Check data
    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("'data' must be a 2D numpy array (samples x channels).")
    # Check params
    try:
        order = params['spectral_filter']['order']
        freqs = params['spectral_filter']['freqs']
        fsamp = params['fsamp']
    except KeyError as e:
        raise ValueError(f"Missing required filter parameter: {e}")
    if not (isinstance(freqs, (list, tuple, np.ndarray)) and len(freqs) == 2):
        raise ValueError("'freqs' must be a list or tuple of two values [low, high].")
    if fsamp <= 0:
        raise ValueError("'fsamp' must be positive.")

    # Normalize frequencies to Nyquist
    nyq = 0.5 * fsamp
    low = freqs[0] / nyq
    high = freqs[1] / nyq
    if not (0 < low < high < 1):
        raise ValueError(f"Filter frequencies must satisfy 0 < low < high < Nyquist. Got low={low}, high={high}")

    try:
        b, a = butter(order, [low, high], btype='band')
        params['spectral_filter']['b'] = b
        params['spectral_filter']['a'] = a
        data_filt = filtfilt(b, a, data, axis=0)
    except Exception as e:
        raise RuntimeError(f"Filtering failed: {e}")

    return data_filt

def epoch_data(data, params, trigger):
    """
    Epoch continuous data based on a trigger dictionary.

    Args:
        data (np.ndarray): Continuous data, shape (n_samples, n_channels).
        params (dict): Configuration dictionary with 'epoch_samples'.
        trigger (dict): Dictionary with 'pos' (indices) and 'typ' (labels).

    Returns:
        tuple: A tuple containing (epoched_data, epoch_labels), where
               epoched_data has shape (n_samples_per_epoch, n_channels, n_epochs).
    """
    # --- Input Validation ---
    if not isinstance(trigger, dict) or 'pos' not in trigger or 'typ' not in trigger:
        raise ValueError("'trigger' must be a dictionary with 'pos' and 'typ' keys.")
    trig_pos = trigger['pos']
    trig_typ = trigger['typ']

    if not isinstance(data, np.ndarray) or data.ndim != 2:
        raise ValueError("'data' must be a 2D numpy array (samples x channels).")
    if 'epoch_samples' not in params:
        raise ValueError("Missing 'epoch_samples' in config dictionary.")

    epoch_samples = params['epoch_samples']
    n_epochs = len(trig_pos)

    if n_epochs == 0:
        raise ValueError("No valid epochs found for the given triggers. Cannot create epochs.")

    # --- Vectorized Epoching ---
    epoch_indices = trig_pos[:, np.newaxis] + epoch_samples # shape n_epochs x n_samples

    if epoch_indices.min() < 0 or epoch_indices.max() >= data.shape[0]:
        raise ValueError("Epoch window extends beyond data boundaries for some triggers.")

    epoched_data = data[epoch_indices, :] # shape n_epochs x n_samples x n_channels

    epoch_labels = np.array(trig_typ)

    return epoched_data, epoch_labels

def balance_runs(epoched_data, epoch_labels, random_state=None):
    """
    Balance the number of no feedback (label=0) and negative feedback (label=1) trials.
    Downsamples label 0 trials at random to match the number of label 1 trials.

    Parameters
    ----------
    epoched_data : np.ndarray
        Epoched data, shape (time_samples, channels, n_epochs)
    epoch_labels : np.ndarray
        Labels for each epoch, shape (n_epochs,)
    random_state : int or None
        Random seed for reproducibility

    Returns
    -------
    balanced_data : np.ndarray
        Balanced epoched data
    balanced_labels : np.ndarray
        Balanced labels
    """
    if not isinstance(epoched_data, np.ndarray) or epoched_data.ndim != 3:
        raise ValueError("'epoched_data' must be a 3D numpy array.")
    if not isinstance(epoch_labels, np.ndarray) or epoch_labels.ndim != 1:
        raise ValueError("'epoch_labels' must be a 1D numpy array.")
    if epoched_data.shape[2] != epoch_labels.shape[0]:
        raise ValueError("Number of epochs in 'epoched_data' and 'epoch_labels' must match.")

    rng = np.random.default_rng(random_state) # if random_state = None --> random seed (not reproducible)

    idx_0 = np.where(epoch_labels == 0)[0]
    idx_1 = np.where(epoch_labels == 1)[0]

    n_1 = len(idx_1)
    n_0 = len(idx_0)

    if n_1 == 0 or n_0 == 0:
        raise ValueError("Both label 0 and label 1 must be present in the data.")

    idx_0_balanced = rng.choice(idx_0, size=n_1, replace=False)
    # Combine and shuffle indices
    balanced_indices = np.concatenate([idx_0_balanced, idx_1])
    balanced_indices = np.sort(balanced_indices)

    balanced_data = epoched_data[:, :, balanced_indices]
    balanced_labels = epoch_labels[balanced_indices]

    return balanced_data, balanced_labels



