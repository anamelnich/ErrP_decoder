import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

def plot_eeg_interactive(
    data,
    params,
    window_sec=10,
    volt_range=50,
    title='EEG',
):
    """
    Interactive EEG viewer using an ipywidgets slider.

    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_channels)
        The EEG data to display.
    params : dict
        Dictionary containing at least 'fsamp' (float) and 'chan_labels' (list of str).
    window_sec : float, optional
        Duration of each page window in seconds (default 10).
    volt_range : float, optional
        ± range (in μV) for each channel's vertical offset (default 50).
    title : str, optional
        Title of the plot (default 'EEG').
    """
    # Extract required parameters
    try:
        sfreq = params['fsamp']
        ch_names = params.get('chan_labels', None)
    except KeyError as e:
        raise ValueError(f"Missing required parameter in params: {e}")

    n_samples, n_channels = data.shape
    window_size = int(window_sec * sfreq)
    t = np.arange(n_samples) / sfreq
    offsets = np.arange(n_channels) * 2 * volt_range

    if ch_names is None:
        ch_names = [f'Ch {i+1}' for i in range(n_channels)]
    else:
        # Truncate or pad ch_names to match n_channels
        ch_names = list(ch_names)[:n_channels] + [f'Ch {i+1}' for i in range(len(ch_names), n_channels)]

    def _plot_window(start):
        end = start + window_size
        plt.figure(figsize=(12, 6))
        for ch in range(n_channels):
            plt.plot(
                t[start:end],
                data[start:end, ch] + offsets[ch],
                lw=0.8
            )
        plt.yticks(offsets, ch_names)
        plt.ylim(-volt_range, offsets[-1] + volt_range)
        plt.xlim(t[start], t[min(end, n_samples-1)])
        plt.xlabel('Time (s)')
        plt.title(f'{title} | {start/sfreq:.1f}–{min(end,n_samples)/sfreq:.1f} s')
        plt.grid(True, axis='x')
        plt.tight_layout()
        plt.show()

    interact(
        _plot_window,
        start=IntSlider(
            min=0,
            max=max(n_samples - window_size, 0),
            step=window_size,
            value=0,
            description='Page',
            continuous_update=False
        )
    )

def plot_erp(
    epochs,                # shape: (time_samples, channels, n_epochs)
    epoch_labels,          # shape: (n_epochs,)
    params,                # dict
    chan,                  # channel index (int) or name (str)
    time_window=(-0.2, 1), # time window to plot (in seconds)
):
    """
    Plot average ERP for positive (0) vs negative (1) feedback for a given electrode.

    Parameters
    ----------
    epochs : np.ndarray
        Epoched data, shape (time_samples, channels, n_epochs)
    epoch_labels : np.ndarray
        Labels for each epoch, shape (n_epochs,)
    params : dict
        Must contain 'epoch_time' and 'chan_labels'
    chan : int or str
        Channel index or channel name to plot
    time_window : tuple, optional
        (start, end) time in seconds to plot (default: (-0.2, 1))
    """
 
    if not isinstance(params, dict) or 'chan_labels' not in params:
        raise ValueError("'params' must contain 'chan_labels'.")
    chan_labels = params['chan_labels']

    if isinstance(chan, str):
        if chan not in chan_labels:
            raise ValueError(f"Channel name '{chan}' not found in chan_labels.")
        chan_idx = chan_labels.index(chan)
        chan_name = chan
    elif isinstance(chan, int):
        if not (0 <= chan < len(chan_labels)):
            raise ValueError(f"chan index {chan} is out of bounds for {len(chan_labels)} channels.")
        chan_idx = chan
        chan_name = chan_labels[chan]
    else:
        raise ValueError("chan must be an int (index) or str (channel name).")

    # --- The rest of your function remains the same ---
    epoch_time = params['epoch_time']
    # Find indices for the time window
    t_mask = (epoch_time >= time_window[0]) & (epoch_time <= time_window[1])
    if not np.any(t_mask):
        raise ValueError(f"No time points found in the specified window {time_window}.")
    
    selected = epochs[t_mask, chan_idx, :] # shape (615, n_epochs)

    pos_trials = selected[:, epoch_labels == 0]   # shape (615, n_pos)
    neg_trials = selected[:, epoch_labels == 1]   # shape (615, n_neg)

    if pos_trials.size == 0:
        raise ValueError("No neutral feedback (label=0) trials found for plotting.")
    if neg_trials.size == 0:
        raise ValueError("No negative feedback (label=1) trials found for plotting.")

    pos_mean = pos_trials.mean(axis=-1)
    neg_mean = neg_trials.mean(axis=-1)
    
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(epoch_time[t_mask], pos_mean, label='No feedback', color='tab:blue')
    plt.plot(epoch_time[t_mask], neg_mean, label='Negative feedback', color='tab:red')
    plt.axvline(0, color='k', linestyle='--', lw=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (μV)')
    plt.title(f'ERP at electrode {chan_name}')
    plt.ylim(-5,9)
    plt.legend()
    plt.tight_layout()
    plt.show()
