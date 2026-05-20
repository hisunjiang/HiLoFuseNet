"""
DATASET: BCIIV and Stanford datasets
Train machine learning models to predict 5-finger trajectories.

"""
import argparse

import random
import os
import numpy as np
import pickle

from models.prepareDataset import Scaler3D, Scaler4D, prepare_taskFormatedData, select_ecog_features
from models.bttr.bttr import *
from models.hopls.hopls import *
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV, PredefinedSplit,GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import tensorly as tl
from tensorly.regression import CP_PLSR
from scipy.stats import pearsonr
from scipy.io import loadmat

parser = argparse.ArgumentParser(description='Finger Regression Task')

parser.add_argument('--dataset', type=str, default='BCIIV',
                    help='Dataset name')
parser.add_argument('--decoder', type=str, default='PLS',
                    help='Decoder type')
parser.add_argument('--win_size', type=float, default=1,
                    help='lookback window') 
parser.add_argument('--subject', type=str, help='Subject name')

                    
args = parser.parse_args()

dataset = args.dataset
decoder = args.decoder
win_size = args.win_size
sub = args.subject

if decoder in ['HiLoFuseNet', 'LSTM', 'MLP']:
    feature_type = 'HGALFS'

elif decoder in ['PLS', 'NPLS', 'HOPLS', 'WaT', 'WaTFi', 'WaTEi']:
    feature_type = 'wavelet_10_150Hz'

elif decoder in ['CNN_LSTM', 'RF']:
    feature_type = 'wavelet_5_195Hz'

elif decoder in ['DeepFingerNet']:
    feature_type = 'wavelet_40_200Hz'

elif decoder in ['eBTTR']:
    feature_type = 'physiologicalBand'
    
elif decoder in ['EEGNet']:
    feature_type = 'raw'
else:pass
    
save_root = 'results/o5/varyingWindow/'
os.makedirs(save_root, exist_ok=True)

# calculate mean corr from 5 fingers
def pearson_correlation_scorer(y_true, y_pred):
    correlations = []
    for i in range(y_true.shape[1]):
        if np.std(y_true[:, i]) > 1e-6 and np.std(y_pred[:, i]) > 1e-6:
            r, _ = pearsonr(y_true[:, i], y_pred[:, i])
            correlations.append(r)
        else:
            correlations.append(0)
    return np.mean(correlations)
    
# metadata for different datasets
datasets = {
    "BCIIV": {
        "subjects": 3,
        "subject_name": ['sub1', 'sub2', 'sub3'],
        "fs_ecog": 1000,
        "fs_dg": 25,
        # file path to preprocessed data
        "path": './data/BCIIV/'
    },
    "Stanford": {
        "subjects": 9,
        "subject_name": ['bp', 'cc', 'ht','jc','jp','mv','wc','wm','zt'],
        "fs_ecog": 1000,
        "fs_dg": 25,
        "path": './data/Stanford/'
    },
}

fs_dg = datasets[dataset]['fs_dg']
fs_ecog = datasets[dataset]['fs_ecog']

# load data
data = loadmat(datasets[dataset]['path'] + sub + '.mat')

# data segmentation & feature extraction
ECoG_train, trajectory_train, ECoG_test, trajectory_test = prepare_taskFormatedData(dataset, data, feature_type, fs_ecog, fs_dg, win_size, delay=0)

# get the validation set (1/10) from original training set
val_len = ECoG_train.shape[0] // 10
ECoG_val, trajectory_val = ECoG_train[-val_len:, ], trajectory_train[-val_len:, :]
ECoG_train, trajectory_train = ECoG_train[:-val_len, ], trajectory_train[:-val_len, :]

# z-score normalization for ECoG
if ECoG_train.ndim == 4:
    norm = Scaler4D()

elif ECoG_train.ndim == 3:
    norm = Scaler3D()
    
norm.fit(ECoG_train)
ECoG_train = norm.transform(ECoG_train)
ECoG_val = norm.transform(ECoG_val)
ECoG_test = norm.transform(ECoG_test)

# z-score normalization for trajectories
mean_traj = trajectory_train.mean(axis=0)
std_traj = trajectory_train.std(axis=0)
std_traj[std_traj == 0] = 1e-6

trajectory_train = (trajectory_train - mean_traj) / std_traj
trajectory_val = (trajectory_val - mean_traj) / std_traj
trajectory_test = (trajectory_test - mean_traj) / std_traj

if decoder == 'PLS':
    ##### Required input dimension: [trial, channel * time * frequency]
    
    ECoG_train = select_ecog_features(ECoG_train, window_len=10, freq_idx=None)
    ECoG_val = select_ecog_features(ECoG_val, window_len=10, freq_idx=None)
    ECoG_test = select_ecog_features(ECoG_test, window_len=10, freq_idx=None)

    # flatten the features
    ECoG_train_flat = ECoG_train.reshape(ECoG_train.shape[0], -1)
    ECoG_val_flat = ECoG_val.reshape(ECoG_val.shape[0], -1)
    ECoG_test_flat = ECoG_test.reshape(ECoG_test.shape[0], -1)

    # optimize n_components
    scores = []
    for R in range(1, 51):
        model = PLSRegression(n_components=R)
        model.fit(ECoG_train_flat, trajectory_train)
        y_pred = model.predict(ECoG_val_flat)

        corr_val = []
        for i in range(y_pred.shape[1]):
            corr, _ = pearsonr(y_pred[:, i], trajectory_val[:, i])
            corr_val.append(corr)
        scores.append(np.mean(corr_val))

    # train&test
    best_R = np.argmax(scores) + 1
    best_model = PLSRegression(n_components=best_R)
    best_model.fit(ECoG_train_flat, trajectory_train)
    y_pred = best_model.predict(ECoG_test_flat)

elif decoder == 'NPLS':
    ##### Required input dimension: [trial, channel, time, frequency]
    
    ECoG_train = select_ecog_features(ECoG_train, window_len=10, freq_idx=None)
    ECoG_val = select_ecog_features(ECoG_val, window_len=10, freq_idx=None)
    ECoG_test = select_ecog_features(ECoG_test, window_len=10, freq_idx=None)
    
    X_train = tl.tensor(ECoG_train)
    X_val = tl.tensor(ECoG_val)
    X_test = tl.tensor(ECoG_test)

    y_train = trajectory_train
    y_val = trajectory_val

    del ECoG_train, ECoG_val, ECoG_test, trajectory_train, trajectory_val

    # optimize n_components
    scores = []
    for R in range(1, 51):
        model = CP_PLSR(n_components=R, tol=1e-6, n_iter_max=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        corr_val = []
        for i in range(y_pred.shape[1]):
            corr, _ = pearsonr(y_pred[:, i], y_val[:, i])
            corr_val.append(corr)
        scores.append(np.mean(corr_val))

    # train&test
    best_R = np.argmax(scores) + 1
    best_model = CP_PLSR(n_components=best_R)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    
elif decoder == 'HOPLS':
    ##### Required input dimension: [trial, channel, time * frequency]
    
    ECoG_train = select_ecog_features(ECoG_train, window_len=10, freq_idx=None)
    ECoG_val = select_ecog_features(ECoG_val, window_len=10, freq_idx=None)
    ECoG_test = select_ecog_features(ECoG_test, window_len=10, freq_idx=None)

    # concatenate the time and frequency dimensions
    ECoG_train = ECoG_train.reshape(ECoG_train.shape[0], ECoG_train.shape[1], -1)
    ECoG_val = ECoG_val.reshape(ECoG_val.shape[0],ECoG_train.shape[1], -1)
    ECoG_test = ECoG_test.reshape(ECoG_test.shape[0], ECoG_train.shape[1],-1)

    # optimize R and Ln
    R_max = 50
    Ln_max = 20
    results = []
    for Ln in range(1, Ln_max+1):
        results.append(
            optimize_R(ECoG_train, trajectory_train, ECoG_val, trajectory_val, Ln, R_max)
        )

    old_Q2 = -np.inf
    for i in range(len(results)):
        R, Q2 = results[i]
        if Q2 > old_Q2:
            best_Ln = i + 1
            best_R = R
            old_Q2 = Q2

    # train&test
    model = HOPLS(best_R, [best_Ln] * (len(ECoG_train.shape) - 1))
    model._fit_2d(ECoG_train, trajectory_train)
    y_pred = model.predict(ECoG_test, Yshape=trajectory_test.shape)

elif decoder == 'eBTTR':
    ##### Required input dimension: [trial, channel, time, frequency]
    
    ECoG_train = np.transpose(ECoG_train, (0, 1, 3, 2))
    ECoG_val = np.transpose(ECoG_val, (0, 1, 3, 2))
    ECoG_test = np.transpose(ECoG_test, (0, 1, 3, 2))

    # optimize nFactor (50)
    model = BTTR()
    model.train(ECoG_train, trajectory_train, 50, score_vector_matrix=True)
    y_pred_blocks = model.predict(ECoG_val)

    scores = []
    for k in range(len(y_pred_blocks)):
        y_pred = np.squeeze(y_pred_blocks[k])

        corr_val = []
        for i in range(y_pred.shape[1]):
            corr, _ = pearsonr(y_pred[:, i], trajectory_val[:, i])
            corr_val.append(corr)
        scores.append(np.mean(corr_val))

    # train&test
    best_k = np.argmax(scores) + 1
    model.train(ECoG_train, trajectory_train, best_k, score_vector_matrix=True)
    y_pred_blocks = model.predict(ECoG_test)
    y_pred = np.squeeze(y_pred_blocks[-1])

elif decoder == 'RF':
    ##### Required input dimension: [trial, channel * time * frequency]
    
    ECoG_train = select_ecog_features(ECoG_train, window_len=10, freq_idx=None)
    ECoG_val = select_ecog_features(ECoG_val, window_len=10, freq_idx=None)
    ECoG_test = select_ecog_features(ECoG_test, window_len=10, freq_idx=None)

    # flatten the features
    ECoG_train_flat = ECoG_train.reshape(ECoG_train.shape[0], -1)
    ECoG_val_flat = ECoG_val.reshape(ECoG_val.shape[0], -1)
    ECoG_test_flat = ECoG_test.reshape(ECoG_test.shape[0], -1)
    
    # create PredefinedSplit
    X_combined = np.vstack((ECoG_train_flat, ECoG_val_flat))
    y_combined = np.vstack((trajectory_train, trajectory_val))

    test_fold = np.concatenate([
        -1 * np.ones(ECoG_train_flat.shape[0]),
        zeros_idx := np.zeros(ECoG_val_flat.shape[0])
    ])
    ps = PredefinedSplit(test_fold)

    # define searching space
    param_grid = {
        'max_depth': [10, 20, None],
        'min_samples_leaf': [2, 5, 10]
    }

    # init RF
    rf = RandomForestRegressor(n_estimators=200, max_features='sqrt', random_state=42, n_jobs=1)

    custom_scorer = make_scorer(pearson_correlation_scorer, greater_is_better=True)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid, 
        cv=ps,              
        scoring=custom_scorer,
        verbose=0,       
        n_jobs=-1
    )

    # optimization & training & test
    grid_search.fit(X_combined, y_combined)
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(ECoG_test_flat)
    
# test
corr_test = []
for i in range(y_pred.shape[1]):
    corr, _ = pearsonr(y_pred[:, i], trajectory_test[:, i])
    corr_test.append(corr)
    
corr_mean = np.mean(corr_test)
corr_test_str = ", ".join([f"{c:.4f}" for c in corr_test])
print(f"subject{sub}: test corr = {corr_test_str}, mean = {corr_mean:.4f}\n", flush=True)

corr_sub_array = np.array(corr_test)

save_fileName_cc = f"{dataset}_{decoder}_win{win_size}_sub_{sub}_o5_cc.npy"
cc_save_path = os.path.join(save_root, save_fileName_cc)
np.save(cc_save_path, corr_sub_array)

# ---- save single subject trajectory ----
traj_save = {
    'true': trajectory_test,
    'pred': y_pred
}
save_fileName_traj = f"{dataset}_{decoder}_win{win_size}_sub_{sub}_o5_trajectory.npz"
traj_save_path = os.path.join(save_root, save_fileName_traj)
np.savez(traj_save_path, **traj_save)
