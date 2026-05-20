import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import mne
from scipy.signal import butter, lfilter, hilbert, sosfilt
from sklearn.base import BaseEstimator, TransformerMixin

class Scaler4D(BaseEstimator, TransformerMixin):
    """
    Z-score normalization for 4D input: [nEpoch, nChannel, nSequence, nBand].
    Mean/std computed over epochs for each (channel, sequence, band).
    Added clipping to avoid extreme values.
    """
    def __init__(self, eps=1e-6, min_std=1e-3, clip_z=5.0):
        self.eps = eps
        self.min_std = min_std
        self.clip_z = clip_z

    def fit(self, X):
        self.scalers_ = {}
        nChannel, nSequence, nBand = X.shape[1], X.shape[2], X.shape[3]
        for c in range(nChannel):
            for s in range(nSequence):
                for b in range(nBand):
                    values = X[:, c, s, b]   # 所有 epoch 的值
                    mean = values.mean()
                    std = values.std()
                    std = max(std, self.min_std)
                    self.scalers_[(c, s, b)] = (mean, std)
        return self

    def transform(self, X):
        X_scaled = np.empty_like(X, dtype=np.float32)
        nChannel, nSequence, nBand = X.shape[1], X.shape[2], X.shape[3]
        for c in range(nChannel):
            for s in range(nSequence):
                for b in range(nBand):
                    mean, std = self.scalers_[(c, s, b)]
                    z = (X[:, c, s, b] - mean) / (std + self.eps)

                    if self.clip_z is not None:
                        z = np.clip(z, -self.clip_z, self.clip_z)

                    X_scaled[:, c, s, b] = z
        return X_scaled

class Scaler3D(BaseEstimator, TransformerMixin):
    """
    Z-score normalization for 3D input: [nEpoch, nSequence, nFeature].
    Mean/std computed over epochs for each (sequence, feature).
    Added clipping to avoid extreme values.
    """
    def __init__(self, eps=1e-6, min_std=1e-3, clip_z=5.0):
        self.eps = eps
        self.min_std = min_std
        self.clip_z = clip_z

    def fit(self, X):
        self.scalers_ = {}
        nSequence, nFeature = X.shape[1], X.shape[2]
        for d1 in range(nSequence):
            for d2 in range(nFeature):
                values = X[:, d1, d2]
                mean = values.mean()
                std = values.std()
                std = max(std, self.min_std)
                self.scalers_[(d1, d2)] = (mean, std)
        return self

    def transform(self, X):
        X_scaled = np.empty_like(X, dtype=np.float32)
        nSequence, nFeature = X.shape[1], X.shape[2]
        for d1 in range(nSequence):
            for d2 in range(nFeature):
                mean, std = self.scalers_[(d1, d2)]
                z = (X[:, d1, d2] - mean) / (std + self.eps)

                # clip to [-clip_z, clip_z]
                if self.clip_z is not None:
                    z = np.clip(z, -self.clip_z, self.clip_z)

                X_scaled[:, d1, d2] = z
        return X_scaled

class constructDataset(Dataset):
    def __init__(self, ecog, traj):
        self.inputs = torch.tensor(ecog, dtype=torch.float32)
        self.targets = torch.tensor(traj, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class BatchShuffleSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)

    def __iter__(self):
        batches = [list(range(i, min(i+self.batch_size, self.num_samples)))
                   for i in range(0, self.num_samples, self.batch_size)]
        np.random.shuffle(batches) 
        for batch in batches:
            yield from batch

    def __len__(self):
        return self.num_samples

def prepare_taskFormatedData(dataset, data, feature_type, fs_ecog =1000, fs_dg=25, win_size=1, delay = None):
    """ 
    Segment the continuous ECoG data and transform each segment into features. Each segment associates with a target finger trajectory
    sampled at 25Hz. Segmentation proceeds feature extraction to ensure casuality.

    INPUT: 
        - dataset: 'BCIIV' or 'Stanford'
        - data: the loaded data from the dataset
        - feature_type: HGALFS' or 'wavelet' or 'physiologicalBand' or 'raw'
        - fs_ecog: default 1 kHz
        - fs_dg: default 25 Hz. Normally match the sampling rate of behavioral data 
        - win_size: default 1s. The lookback window for decoding
        - delay (float, optional): Compensation for hardware/physiological latency (seconds).
            - delay < 0 (e.g., -0.05s): Standard Causal Lead. Brain signal at time 't'
              is used to predict trajectory at 't + |delay|'. This aligns with the
              physiological reality that neural intent precedes motor execution.
            - delay > 0 (e.g., 0.05s): Hardware Lag Correction. Brain signal at time 't'
              is paired with trajectory at 't - delay'. This is typically used to
              correct cases where the tracking hardware is faster than the ECoG
              acquisition or to perform post-diction.
            - Default (Zero): Avoids alignment bias. Previous BCI studies often omit
              precise lag correction details, and dataset descriptions can be
              ambiguous regarding the absolute synchronization of recording devices.
        
    OUTPUT: ECoG -> [nEpoch, nChannel, nSequence, nFrequency (optional)], trajectory -> [nTarget (=nEpoch), 5]
    """

    if dataset == 'BCIIV':
        data_train, dg_train = data['train_data'].T, data['train_dg']
        data_test, dg_test = data['test_data'].T, data['test_dg']

    elif dataset == 'Stanford':
        data_all = data['data'].T
        dg_all = data['flex']

        # split train and test dataset
        if data_all.shape[0] // fs_ecog >= 600:
            data_train, dg_train = data_all[0:400 * fs_ecog, :], dg_all[0:400 * fs_ecog, :]
            data_test, dg_test = data_all[400 * fs_ecog:, :], dg_all[400 * fs_ecog:, :]
        else:
            len = data_all.shape[0] // 3 * 2
            data_train, dg_train = data_all[0:len, :], dg_all[0:len, :]
            data_test, dg_test = data_all[len:, :], dg_all[len:, :]

    if feature_type == 'HGALFS':
        ECoG_train, trajectory_train = _extract_hga_lfs(data_train, dg_train, fs= fs_ecog, win_size= win_size,
                                                                      hop_size = 1/fs_dg, delay=delay)
        ECoG_test, trajectory_test = _extract_hga_lfs(data_test, dg_test, fs= fs_ecog, win_size= win_size,
                                                                     hop_size = 1/fs_dg, delay=delay)

    elif feature_type == 'wavelet_10_150Hz':
        ECoG_train, trajectory_train = _extract_wavelet(data_train, dg_train, fs= fs_ecog,freqs = np.linspace(10, 150, 15), is_power=False, win_size= win_size,
                                                                      hop_size = 1/fs_dg, batch_size=64, delay=delay)
        ECoG_test, trajectory_test = _extract_wavelet(data_test, dg_test, fs= fs_ecog, freqs = np.linspace(10, 150, 15), is_power=False, win_size= win_size,
                                                                     hop_size = 1/fs_dg, batch_size=64, delay=delay)

    elif feature_type == 'wavelet_5_195Hz':
        ECoG_train, trajectory_train = _extract_wavelet(data_train, dg_train, fs= fs_ecog,freqs = np.geomspace(5, 195, 20), is_power=True, win_size= win_size,
                                                                      hop_size = 1/fs_dg, batch_size=64, delay=delay)
        ECoG_test, trajectory_test = _extract_wavelet(data_test, dg_test, fs= fs_ecog, freqs = np.geomspace(5, 195, 20), is_power=True, win_size= win_size,
                                                                     hop_size = 1/fs_dg, batch_size=64, delay=delay)

    elif feature_type == 'wavelet_40_200Hz':
        ECoG_train, trajectory_train = _extract_wavelet(data_train, dg_train, fs= fs_ecog,freqs = np.geomspace(40, 200, 32), is_power=True, win_size= win_size,
                                                                      hop_size = 1/fs_dg, batch_size=64, delay=delay)
        ECoG_test, trajectory_test = _extract_wavelet(data_test, dg_test, fs= fs_ecog, freqs = np.geomspace(40, 200, 32), is_power=True, win_size= win_size,
                                                                     hop_size = 1/fs_dg, batch_size=64, delay=delay)

    elif feature_type == 'physiologicalBand':
        ECoG_train, trajectory_train = _extract_physiologicalBand(data_train, dg_train, fs= fs_ecog, win_size= win_size,
                                                                 hop_size = 1/fs_dg, delay=delay)
        ECoG_test, trajectory_test = _extract_physiologicalBand(data_test, dg_test, fs= fs_ecog,
                                                                 win_size=win_size, hop_size = 1/fs_dg, delay=delay)

    elif feature_type == 'raw':
        ECoG_train, trajectory_train = _extract_raw(data_train, dg_train, fs= fs_ecog, win_size= win_size,
                                                              hop_size=1 / fs_dg, delay=delay)
        ECoG_test, trajectory_test = _extract_raw(data_test, dg_test, fs= fs_ecog,
                                                                 win_size=win_size, hop_size=1 / fs_dg, delay=delay)

    return ECoG_train, trajectory_train, ECoG_test, trajectory_test

def _extract_hga_lfs(data, traj, fs=1000, fs_feat=200, win_size=1.0, hop_size=0.04,
                                        delay=None):
    """
    Extracts High Gamma Activity (HGA) and Low Frequency Signal (LFS) features with
    causal alignment for BCI finger trajectory prediction.

    This function utilizes global causal filtering to ensure real-time consistency
    while applying a windowed Hilbert transform for envelope extraction. Output
    arrays are optimized for deep learning performance (float32 and contiguous memory).

    Args:
        data (np.ndarray): Raw ECoG signal of shape [Time, Channels].
        traj (np.ndarray): Finger trajectory labels of shape [Time, Dimensions].
        fs (int): Sampling rate of the raw data (Hz). Default is 1000.
        fs_feat (int): Desired temporal resolution of features (Hz). Default is 200.
        win_size (float): Lookback window length in seconds. Default is 1.0.
        hop_size (float): Step size between sliding windows in seconds. Default is 0.04.
        delay (float, optional): Compensation for hardware/physiological latency (seconds).

    Returns:
        X (np.ndarray): Feature tensor of shape [n_samples, n_channels, T, 2].
            - [..., 0]: HGA envelope extracted via Hilbert transform and binning.
            - [..., 1]: LFS extracted via causal low-pass filtering and decimation.
        Y (np.ndarray): Trajectory labels of shape [n_samples, Dimensions].
    """
    data = data.astype(np.float64)
    traj = traj.astype(np.float64)

    # Delay / Causality Alignment
    if delay:
        delay_pts = int(abs(delay) * fs)
        if delay > 0:
            data, traj = data[delay_pts:, :], traj[:-delay_pts, :]
        else:
            data, traj = data[:-delay_pts, :], traj[delay_pts:, :]

    # Parameters
    n_samples, n_channels = data.shape
    hop_pts = int(hop_size * fs)
    win_pts = int(win_size * fs)
    T = int(win_size * fs_feat)
    q = fs // fs_feat

    # Global Causal Filtering
    b_hga, a_hga = butter(4, [70 / (fs / 2), 200 / (fs / 2)], btype='bandpass')
    data_hga_filt = lfilter(b_hga, a_hga, data, axis=0)

    b_lfs, a_lfs = butter(4, 100 / (fs / 2), btype='low')
    data_lfs_filt = lfilter(b_lfs, a_lfs, data, axis=0)

    X, Y = [], []
    for end in range(win_pts, n_samples - hop_pts, hop_pts):
        start = end - win_pts

        # HGA: Windowed Hilbert + Binning
        hga_win = data_hga_filt[start:end, :].T
        analytic = hilbert(hga_win, axis=1)
        envelope = np.abs(analytic)
        hga_feat = envelope[:, :T * q].reshape(n_channels, T, q).mean(axis=2)

        # LFS: Causal Decimation
        lfs_win = data_lfs_filt[start:end, :]
        lfs_feat = lfs_win[::q, :].T
        lfs_feat = lfs_feat[:, :T]

        X.append(np.stack([hga_feat, lfs_feat], axis=-1))
        Y.append(traj[end, :])

    X = np.ascontiguousarray(np.array(X, dtype=np.float32))
    Y = np.ascontiguousarray(np.array(Y, dtype=np.float32))
    return X, Y

def _extract_wavelet(data, traj, fs=1000, fs_feat=100, freqs = np.linspace(10, 150, 15), is_power = True, win_size=1.0, hop_size=0.04,
                     batch_size=128, delay=None):
    """
    Morlet Wavelet feature extraction.
    """
    data = data.astype(np.float64)
    traj = traj.astype(np.float64)

    # Delay / Causality Alignment
    if delay:
        delay_pts = int(abs(delay) * fs)
        if delay > 0:
            data, traj = data[delay_pts:, :], traj[:-delay_pts, :]
        else:
            data, traj = data[:-delay_pts, :], traj[delay_pts:, :]

    # Parameters
    n_samples, n_chans = data.shape
    win_pts = int(win_size * fs)
    hop_pts = int(hop_size * fs)
    T = int(win_size * fs_feat)

    # Causal Downsampling (1kHz -> 500Hz) to save compute
    b_lp, a_lp = butter(4, 200 / (fs / 2), btype='low')
    data_ds = lfilter(b_lp, a_lp, data, axis=0)[::2, :]
    fs_new = fs // 2
    win_pts_ds, hop_pts_ds = win_pts // 2, hop_pts // 2
    n_samples_ds = data_ds.shape[0]

    # Sliding Window Slicing
    X_raw, Y = [], []
    for end_ds in range(win_pts_ds, n_samples_ds - hop_pts_ds, hop_pts_ds):
        start_ds = end_ds - win_pts_ds
        X_raw.append(data_ds[start_ds:end_ds, :].T)
        Y.append(traj[end_ds * 2, :])  # Map back to 1kHz traj

    epochs = np.array(X_raw)
    n_epochs = epochs.shape[0]

    # Batch Wavelet Processing
    n_cycles = freqs / 5.0
    def process_batch(batch):
        if is_power:
            tfr = mne.time_frequency.tfr_array_morlet(
                batch, sfreq=fs_new, freqs=freqs,n_cycles=n_cycles, output='power', n_jobs=1, verbose=False
            )
        else:
            tfr = np.abs(mne.time_frequency.tfr_array_morlet(
                batch, sfreq=fs_new, freqs=freqs, n_cycles=n_cycles, output='complex', n_jobs=1, verbose=False
            ))

        # Time decimation (Binning)
        sub = tfr.shape[-1] // T
        tfr = tfr[..., :sub * T].reshape(batch.shape[0], n_chans, len(freqs), T, sub).mean(axis=-1)
        return np.transpose(tfr, (0, 1, 3, 2)).astype(np.float32)

    # Execute Batches
    features = []
    step = batch_size if batch_size else n_epochs
    for i in range(0, n_epochs, step):
        features.append(process_batch(epochs[i : i + step]))

    return (np.ascontiguousarray(np.concatenate(features, axis=0)),
            np.ascontiguousarray(np.array(Y, dtype=np.float32)))

def _extract_physiologicalBand(data, traj, fs=1000, fs_feat=10, win_size=1.0, hop_size=0.04, delay = None):
    """
    Extracts physiological frequency band power (delta, theta, alpha, beta, gamma)
    using global causal filtering and windowed binning.

    Args:
        data (np.ndarray): Raw ECoG signal [Time, Channels].
        traj (np.ndarray): Finger trajectory [Time, Dimensions].
        fs (int): Raw sampling rate (Hz).
        win_size (float): Lookback window length in seconds.
        hop_size (float): Step size between sliding windows in seconds.
        delay (float, optional): Lag compensation in seconds.

    Returns:
        X (np.ndarray): Feature tensor of shape [n_samples, n_channels, T, n_bands].
        Y (np.ndarray): Trajectory labels of shape [n_samples, Dimensions].
    """
    data = data.astype(np.float64)
    traj = traj.astype(np.float64)

    # Delay / Causality Alignment
    if delay:
        delay_pts = int(abs(delay) * fs)
        if delay > 0:
            data, traj = data[delay_pts:, :], traj[:-delay_pts, :]
        else:
            data, traj = data[:-delay_pts, :], traj[delay_pts:, :]

    # Parameters
    n_samples, n_chans = data.shape
    win_pts = int(win_size * fs)
    hop_pts = int(hop_size * fs)
    T = int(win_size * fs_feat)
    q = fs // fs_feat

    # define functional bands
    bands = {
        "δ": (1.5, 5),
        "θ": (5, 8),
        "α": (8, 12),
        "β1": (12, 24),
        "β2": (24, 34),
        "γ1": (34, 60),
        "γ2": (60, 100),
        "γ3": (100, 130),
    }

    # --- Global Causal Filtering (Pre-calculate all bands) ---
    # We apply sosfilt globally once per band to ensure zero edge artifacts
    # and maintain strict causality for online simulation.
    data_bands = []
    for name, (l, h) in bands.items():
        sos = butter(4, [l / (fs / 2), h / (fs / 2)], btype='bandpass', output='sos')
        filt_signal = sosfilt(sos, data, axis=0)
        amplitude = np.abs(hilbert(filt_signal, axis=0))
        data_bands.append(amplitude)

    data_amplitude_all = np.stack(data_bands, axis=-1)
    
    X, Y = [], []
    for end in range(win_pts, n_samples - hop_pts, hop_pts):
        start = end - win_pts

        win_power = data_amplitude_all[start:end, :, :]
        win_power = win_power.transpose(2, 1, 0) # -> [n_bands, n_chans, win_pts]

        # Binning
        feat_binned = win_power.reshape(len(bands), n_chans, T, q).mean(axis=-1)

        # Reorder to [n_channels, T, n_bands]
        X.append(feat_binned.transpose(1, 2, 0))
        Y.append(traj[end, :])

    X = np.ascontiguousarray(np.array(X, dtype=np.float32))
    Y = np.ascontiguousarray(np.array(Y, dtype=np.float32))
    return X, Y

def _extract_raw(data, traj, fs=1000, win_size=1.0, hop_size=0.04, delay = None):

    data = data.astype(np.float64)
    traj = traj.astype(np.float64)

    # Delay / Causality Alignment
    if delay:
        delay_pts = int(abs(delay) * fs)
        if delay > 0:
            data, traj = data[delay_pts:, :], traj[:-delay_pts, :]
        else:
            data, traj = data[:-delay_pts, :], traj[delay_pts:, :]

    # Parameters
    win_pts = int(win_size * fs)
    hop_pts = int(hop_size * fs)

    # Causal Downsampling (1kHz -> 500Hz) to save compute
    b_lp, a_lp = butter(4, 200 / (fs / 2), btype='low')
    data_ds = lfilter(b_lp, a_lp, data, axis=0)[::2, :]
    win_pts_ds, hop_pts_ds = win_pts // 2, hop_pts // 2
    n_samples_ds = data_ds.shape[0]

    # Sliding Window Slicing
    X, Y = [], []
    for end_ds in range(win_pts_ds, n_samples_ds - hop_pts_ds, hop_pts_ds):
        start_ds = end_ds - win_pts_ds
        X.append(data_ds[start_ds:end_ds, :].T)
        Y.append(traj[end_ds * 2, :])  # Map back to 1kHz traj

    X = np.ascontiguousarray(np.array(X, dtype=np.float32))
    Y = np.ascontiguousarray(np.array(Y, dtype=np.float32))
    return X, Y
    
def select_ecog_features(features, window_len=10, freq_idx=None):
    """
    Select and downsample ECoG features.

    Parameters
    ----------
    features : ndarray, shape (N, C, T, F)
        Input feature tensor: trial × channels × time × frequency
    window_len : int
        Window length for temporal averaging (must divide T).
    freq_idx : list or slice, optional
        Indices of frequencies to keep.
        Example: slice(-5, None) (last 5 freqs) or [0, 2, 4].
        Default: keep all.

    Returns
    -------
    features_new : ndarray, shape (N, C, T_new, F_new)
        Reduced feature tensor.
    """
    N, C, T, F = features.shape

    # downsampling by average pooling
    if T % window_len != 0:
        raise ValueError(f"window_len={window_len} must divide T={T} exactly")
    T_new = T // window_len
    # reshape -> mean pooling
    features_time = features.reshape(N, C, T_new, window_len, F).mean(axis=3)

    if freq_idx is not None:
        features_final = features_time[:, :, :, freq_idx]
    else:
        features_final = features_time

    return features_final
