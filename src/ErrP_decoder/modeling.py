# modeling.py: Model training and inference

import numpy as np
from sklearn.cross_decomposition import CCA
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def get_cca_spatialfilter(epoch_data, epoch_labels, n_components=None):
    """
    Compute CCA spatial filter for EEG data.
    
    Parameters:
    -----------
    epoch_data : array-like, shape (samples, channels, trials)
        Epoched EEG data
    epoch_labels : array-like, shape (trials,)
        Class labels for each trial
    n_components : int, optional
        Number of CCA components to compute. If None, uses min(channels, n_classes)
    
    Returns:
    --------
    spatial_filter : array, shape (channels, n_components)
        CCA spatial filter weights
    """

    if epoch_data.shape[2] != len(epoch_labels):
        raise ValueError("Number of trials in data doesn't match number of labels")
    
    classes = np.unique(epoch_labels)
    if len(classes) < 2:
        raise ValueError("Need at least 2 classes for CCA")
    
    # Set number of components if not specified
    if n_components is None:
        n_components = min(epoch_data.shape[1], len(classes))
    
    concat_data = []
    concat_ga = []

    for cls in classes:
        idx = epoch_labels == cls
        ex_epochs = epoch_data[:, :, idx]  # shape: (samples, channels, trials)
        ex_epochs = np.transpose(ex_epochs, (1, 0, 2))  # (channels, samples, trials)

        ga = np.mean(ex_epochs, axis=2)  # (channels, samples)
        n_trials = ex_epochs.shape[2]

        ex_flat = ex_epochs.reshape(ex_epochs.shape[0], -1)  # (channels, samples × trials)
        ga_rep = np.tile(ga, (1, n_trials))  # (channels, samples × trials)

        concat_data.append(ex_flat)
        concat_ga.append(ga_rep)

    X = np.concatenate(concat_data, axis=1).T  # shape: (samples×trials, channels)
    Y = np.concatenate(concat_ga, axis=1).T    # same shape
    
    # Center the data (subtract each channel mean)
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)

    cca = CCA(n_components=n_components, scale=False)
    cca.fit(X, Y)
    spatial_filter = cca.x_weights_  

    return spatial_filter

def apply_spatial_filter(data_input, filter_matrix):
    """
    Apply spatial filter to EEG data.

    Parameters
    ----------
    data_input : np.ndarray
        EEG data of shape (samples, channels, trials)
    filter_matrix : np.ndarray
        Spatial filter of shape (channels, components)

    Returns
    -------
    data_output : np.ndarray
        Transformed data of shape (samples, components, trials)
    """
    # Input validation
    if data_input.shape[1] != filter_matrix.shape[0]:
        raise ValueError(f"Channel dimension mismatch: data has {data_input.shape[1]} channels, "
                        f"but filter expects {filter_matrix.shape[0]} channels")
    
    n_samples, n_channels, n_trials = data_input.shape
    n_components = filter_matrix.shape[1]
 
    data_2d = data_input.transpose(0, 2, 1).reshape(-1, n_channels) # (samples × trials, channels)
    
    # Apply filter: (samples × trials, channels) @ (channels, components) → (samples × trials, components)
    filtered_2d = data_2d @ filter_matrix

    data_output = filtered_2d.reshape(n_samples, n_trials, n_components).transpose(0, 2, 1) # (samples, components, trials)
    
    return data_output

def resample_epochs(data, params):
    """
    Downsample EEG data within a specified time window using simple decimation.

    Parameters
    ----------
    data : np.ndarray
        EEG data of shape (samples, components, trials)
    params : dict
        Must contain:
            - 'epoch_roi': 1D array of indices (sample indices to keep)
            - 'fsamp': original sampling frequency (Hz)
            - 'resample': dict with key 'freq' for desired frequency (Hz)

    Returns
    -------
    resampled : np.ndarray
        Downsampled EEG data of shape (samples_new, components, trials)
    """

    if 'epoch_roi' not in params:
        raise ValueError("params must contain 'epoch_roi'")
    if 'fsamp' not in params:
        raise ValueError("params must contain 'fsamp'")
    if 'resample' not in params or 'freq' not in params['resample']:
        raise ValueError("params must contain 'resample' with 'freq' key")
    
    roi_indices = params['epoch_roi']
    fsamp = params['fsamp']
    desired_fs = params['resample']['freq']

    if desired_fs >= fsamp:
        raise ValueError(f"Desired frequency ({desired_fs} Hz) must be less than original frequency ({fsamp} Hz)")
    if desired_fs <= 0:
        raise ValueError(f"Desired frequency must be positive, got {desired_fs}")

    if np.max(roi_indices) >= data.shape[0]:
        raise ValueError(f"ROI indices exceed data dimensions. Max index: {np.max(roi_indices)}, data shape: {data.shape[0]}")

    ratio = round(fsamp / desired_fs)

    downsampled_indices = roi_indices[::ratio]

    return data[downsampled_indices, :, :]

def compute_decoder(training_data, training_labels, params):
    """
    Train a decoder.
    
    Parameters:
    -----------
    training_data : np.ndarray, shape (samples, channels, trials)
        Training EEG data
    training_labels : np.ndarray, shape (trials,)
        Class labels for each trial
    params : dict
        Configuration parameters
    
    Returns:
    --------
    decoder : dict
        Dictionary containing:
            - 'lda': trained LDA model
            - 'spatial_filter': CCA spatial filter matrix
            - 'params': parameters used for preprocessing
    """
    # Input validation
    if training_data.shape[2] != len(training_labels):
        raise ValueError("Number of trials in data doesn't match number of labels")
    
    if 'spatial_filter' not in params or 'n_comp' not in params['spatial_filter']:
        raise ValueError("params must contain 'spatial_filter' with 'n_comp' key")
    
    # Compute CCA spatial filter
    spatial_filter = get_cca_spatialfilter(training_data, training_labels, 
                                         n_components=params['spatial_filter']['n_comp'])
    
    # Apply spatial filter
    filtered_data = apply_spatial_filter(training_data, spatial_filter)
    
    # Downsample data
    resampled = resample_epochs(filtered_data, params)
    
    # Reshape for LDA
    n_samples, n_comp, n_trials = resampled.shape
    X = resampled.reshape(n_samples * n_comp, n_trials).T  # shape: (trials, features)
    y = training_labels  # shape: (trials,)
    
    # Train LDA classifier
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)

    decoder = {
        'lda': lda,
        'spatial_filter': spatial_filter,
        'params': params
    }
    
    return decoder

def single_classification(decoder, test_data, test_labels=None):
    """
    Run inference using the trained decoder.
    
    Parameters:
    -----------
    decoder : dict
        Dictionary containing trained LDA model, spatial filter, and parameters
    test_data : np.ndarray, shape (samples, channels, trials)
        Test EEG data
    test_labels : np.ndarray, shape (trials,), optional
        True labels for evaluation (if provided)
    
    Returns:
    --------
    predictions : np.ndarray, shape (trials,)
        Predicted class labels
    probabilities : np.ndarray, shape (trials, n_classes), optional
        Prediction probabilities (if test_labels provided)
    """
    # Extract components from decoder
    lda = decoder['lda']
    spatial_filter = decoder['spatial_filter']
    params = decoder['params']
    
    # Apply same preprocessing pipeline as training
    # Step 1: Apply spatial filter
    filtered_data = apply_spatial_filter(test_data, spatial_filter)
    
    # Step 2: Resample data
    resampled = resample_epochs(filtered_data, params)
    
    # Step 3: Reshape for LDA (same as training)
    n_samples, n_comp, n_trials = resampled.shape
    X = resampled.reshape(n_samples * n_comp, n_trials).T  # shape: (trials, features)
    
    # Step 4: Make predictions
    predictions = lda.predict(X)
    
    if test_labels is not None:
        # Return predictions and probabilities for evaluation
        probabilities = lda.predict_proba(X)
        return predictions, probabilities
    else:
        # Return only predictions
        return predictions

