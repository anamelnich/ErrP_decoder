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
