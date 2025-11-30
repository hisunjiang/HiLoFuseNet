import torch
from torch.utils.data import Dataset, Sampler
import numpy as np
import mne
from mne.filter import resample
from scipy.signal import hilbert
from sklearn.base import BaseEstimator, TransformerMixin
import time

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

def functionalBand_feature_extractor(data, traj, fs, window_size=1000, step_size = 50, T = 200, delay = None):
    """
    INPUT:
        data: [time, channel] ECoG data, 1 kHz
        traj: [time, 5] trajectory of 5 fingers, 1 kHz
        window_size: window size of ECoG in ms
        step_size: step size of sliding windows in ms
        T: number of time points after downsampling
        delay: optional time delay between ECoG and trajectory

    OUTPUT:
        X: [nEpoch, nChannel, T, nBand]
        Y: [nEpoch, 5]
    """
    assert data.shape[0] == traj.shape[0], "ECoG and Trajectory must be same shape"
    data = data.astype(np.float64)
    traj = traj.astype(np.float64)

    if delay is not None:
        data, traj = data[delay:, :], traj[:-delay, :]

    # Downsample trajectory to match step size
    traj_ds = resample(traj.T, down=step_size, npad='auto').T

    # define functional bands
    bands = {
        "δ": (1, 4),
        "θ": (4, 8),
        "α": (8, 12),
        "β1": (12, 24),
        "β2": (24, 34),
        "γ1": (34, 70),
        "γ2": (70, 200),
    }

    X, Y = [], []
    n_samples = len(traj_ds)
    for k in range(n_samples):
        start_time = time.time()

        end = k * step_size
        start = end - window_size
        if start < 0:
            continue
        data_win = data[start:end].T
        traj_point = traj_ds[k]

        n_channels, n_times = data_win.shape
        subwin_len = n_times // T  # for downsampling

        # extract frequency band features for each ECoG segment
        feature_allBand = []
        iir_params = dict(order=4, ftype='butter')
        for name, (l, h) in bands.items():
            data_filtered = mne.filter.filter_data(data_win, sfreq=fs,
                                              l_freq=l, h_freq=h,
                                              method='iir', iir_params=iir_params,
                                              verbose=False)
            # Feature extraction
            feature = data_filtered ** 2

            # if name == "γ2":
            #     analytic = hilbert(data_filtered, axis=1)
            #     feature = np.abs(analytic)  # Hilbert envelope
            # else:
            #     feature = data_filtered ** 2  # instantaneous power

            feature = feature[..., :subwin_len * T]
            feature_downsampled = feature.reshape(n_channels, T, subwin_len).mean(axis=2)

            feature_allBand.append(feature_downsampled)

        # Stack all bands: (channels, T, nBand)
        feature_allBand = np.stack(feature_allBand, axis=-1)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"CPU time: {elapsed_time:.6f} s")

        X.append(feature_allBand)
        Y.append(traj_point)

    return np.array(X), np.array(Y)

def HGALFS_feature_extractor(data, traj, fs, window_size=1000, step_size = 50, T = 200, delay = None):
    """
    INPUT:
        data: [time, channel] ECoG data, 1 kHz
        traj: [time, 5] trajectory of 5 fingers, 1 kHz
        window_size: window size of ECoG in ms
        step_size: step size of sliding windows in ms
        T: number of time points after downsampling
        delay: optional time delay between ECoG and trajectory

    OUTPUT:
        X: [nEpoch, nChannel, T, nBand]
        Y: [nEpoch, 5]
    """
    assert data.shape[0] == traj.shape[0], "ECoG and Trajectory must be same shape"
    data = data.astype(np.float64)
    traj = traj.astype(np.float64)

    if delay is not None:
        data, traj = data[delay:, :], traj[:-delay, :]

    # Downsample trajectory to match step size
    traj_ds = resample(traj.T, down=step_size, npad='auto').T

    X, Y  = [], []
    n_samples = len(traj_ds)
    for k in range(n_samples):
        start_time = time.time()

        end = k * step_size
        start = end - window_size
        if start < 0:
            continue
        data_win = data[start:end].T
        traj_point = traj_ds[k]

        n_channels, n_times = data_win.shape
        subwin_len = n_times // T  # for downsampling

        # extract HGA (n_channels, T)
        iir_params = dict(order=4, ftype='butter')
        data_filtered = mne.filter.filter_data(data_win, sfreq=fs,
                                          l_freq=70 , h_freq=200,
                                          method='iir', iir_params=iir_params,
                                          verbose=False)
        analytic = hilbert(data_filtered, axis=1)
        envelope = np.abs(analytic)  # (n_channels, n_times)
        envelope = envelope[..., :subwin_len * T]

        HGA = envelope.reshape(n_channels, T, subwin_len).mean(axis=2)

        # extract LFS (n_channels, T)
        LFS = resample(data_win, down=5, npad='auto')

        # -> (n_channels, T, n_bands)
        features = np.stack([HGA, LFS], axis=-1)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"CPU time: {elapsed_time:.6f} s")

        X.append(features)
        Y.append(traj_point)

    return np.array(X), np.array(Y)

def wavelet_feature_extractor(data, traj, fs, window_size=1000, step_size = 50, T = 100, batch_size = None, delay = None):
    """
    INPUT:
        data: [time, channel] ECoG data, 1 kHz
        traj: [time, 5] trajectory of 5 fingers, 1 kHz
        window_size: window size of ECoG in ms
        step_size: step size of sliding windows in ms
        T: number of time points after downsampling
        batch_size: samples in every wavelet computation
        delay: optional time delay between ECoG and trajectory

    OUTPUT:
        features: [nEpoch, nChannel, T, frequency]
        trajectory: [nEpoch, 5]
    """
    assert data.shape[0] == traj.shape[0], "ECoG and Trajectory must be same shape"
    data = data.astype(np.float64)
    traj = traj.astype(np.float64)

    if delay is not None:
        data, traj = data[delay:, :], traj[:-delay, :]

    # Downsample trajectory to match step size
    traj_ds = resample(traj.T, down=step_size, npad='auto').T

    # downsample the data to 500 Hz (to reduce computation time in wavelet)
    data = resample(data.T, down=2, npad='auto')
    fs_new = fs // 2

    X, Y = [], []
    n_samples = len(traj_ds)
    for k in range(n_samples):
        end = k * int(fs_new * (step_size / 1000))
        start = end - int(fs_new * (window_size / 1000))
        if start < 0:
            continue
        data_win = data[:, start:end]
        traj_point = traj_ds[k]

        X.append(data_win)
        Y.append(traj_point)

    epochs, trajectory = np.array(X), np.array(Y)

    n_epochs, n_channels, n_times = epochs.shape

    # wavelet transform
    freqs = np.linspace(5, 195, 20)
    n_cycles = freqs / 5.0

    def process_batch(data_batch):
        # Morlet CWT
        tfr = mne.time_frequency.tfr_array_morlet(
            data_batch,
            sfreq=fs_new,
            freqs=freqs,
            n_cycles=n_cycles,
            output='power',
            n_jobs=1
        )  # (batch, channels, freqs, times)

        # Decimate along time
        subwin_len = n_times // T
        tfr = tfr[..., :subwin_len * T]  # (batch, channels, freqs, T*subwin_size)
        tfr_down = tfr.reshape(
            data_batch.shape[0], n_channels, len(freqs), T, subwin_len
        ).mean(axis=-1)

        # Reorder to (batch, C, T, F)
        features = np.transpose(tfr_down, (0, 1, 3, 2))
        return features.astype(np.float32) # reduce storage space

    if batch_size is None:
        features = process_batch(epochs)
    else:
        batches = []
        for i in range(0, n_epochs, batch_size):
            start_time = time.time()

            batch = epochs[i:i + batch_size]
            batches.append(process_batch(batch))

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"CPU time: {elapsed_time:.6f} s")

        features = np.concatenate(batches, axis=0).astype(np.float32)

    return features, trajectory

def select_ecog_features(data, window_len=1, freq_idx=None):
    """
    Select and downsample ECoG features.

    Parameters
    ----------
    data : ndarray, shape (N, C, T, F)
        Input feature tensor: trial × channels × time × frequency
    window_len : int
        Window length for temporal averaging (must divide T).
    freq_idx : list or slice, optional
        Indices of frequencies to keep.
        Example: slice(-5, None) (last 5 freqs) or [0, 2, 4].
        Default: keep all.

    Returns
    -------
    data_selected: ndarray, shape (N, C, T_new, F_new)
        Reduced feature tensor.
    """
    N, C, T, F = data.shape

    # downsampling by average pooling
    if T % window_len != 0:
        raise ValueError(f"window_len={window_len} must divide T={T} exactly")
    T_new = T // window_len
    tmp = data.reshape(N, C, T_new, window_len, F).mean(axis=3)

    if freq_idx is not None:
        data_selected = tmp[:, :, :, freq_idx]
    else:
        data_selected = tmp

    return data_selected
