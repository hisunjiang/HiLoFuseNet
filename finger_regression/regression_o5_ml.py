"""
DATASET: BCIIV and Stanford datasets
Train machine learning models to predict 5-finger trajectories.

"""
import argparse

import random
import os
import numpy as np
import pickle

from models.prepareDataset import Scaler4D, select_ecog_features
from models.bttr.bttr import *
from models.hopls.hopls import *
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import pearsonr


parser = argparse.ArgumentParser(description='Finger Regression Task')

parser.add_argument('--dataset', type=str, default='BCIIV',
                    choices=['BCIIV', 'Stanford'],
                    help='Dataset name')
parser.add_argument('--decoder', type=str, default='PLS',
                    choices=['PLS', 'HOPLS', 'BTTR'],
                    help='Decoder type')
args = parser.parse_args()

dataset = args.dataset
decoder = args.decoder

save_root = 'results/o5/'
os.makedirs(save_root, exist_ok=True)

# metadata for different datasets
datasets = {
    "BCIIV": {
        "subjects": 3,
        "subject_name": ['sub1', 'sub2', 'sub3'],
        "fs_ecog": 1000,
        "fs_dg": 25,
        # file path to extracted features
        "path": '/lustre1/scratch/355/vsc35565/finger_ECoG/BCIIV/'
    },
    "Stanford": {
        "subjects": 9,
        "subject_name": ['bp', 'cc', 'ht','jc','jp','mv','wc','wm','zt'],
        "fs_ecog": 1000,
        "fs_dg": 25,
        "path": '/lustre1/scratch/355/vsc35565/finger_ECoG/Stanford/'
    },
}

fileLoc = datasets[dataset]['path']
fs_dg = datasets[dataset]['fs_dg']
fs_ecog = datasets[dataset]['fs_ecog']

corr_allSub = np.zeros((5, datasets[dataset]['subjects']))
traj_true = {f'sub{iS}': [] for iS in range(datasets[dataset]['subjects'])}
traj_pred = {f'sub{iS}': [] for iS in range(datasets[dataset]['subjects'])}
    
# loop subjects
for iS in range(datasets[dataset]['subjects']):
    # load extracted features
    filename = fileLoc + datasets[dataset]['subject_name'][iS] + "_wavelet_features_100seq.pkl"
        
    with open(filename, "rb") as f:
        ECoG_train, trajectory_train, ECoG_test, trajectory_test = pickle.load(f)

    # get the validation set (1/10) from original training set
    val_len = ECoG_train.shape[0] // 10
    ECoG_val, trajectory_val = ECoG_train[-val_len:, :, :, :], trajectory_train[-val_len:, :]
    ECoG_train, trajectory_train = ECoG_train[:-val_len, :, :, :], trajectory_train[:-val_len, :]

    # z-score normalization for ECoG
    norm = Scaler4D()
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

    # select a model
    if decoder == 'PLS':
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

    elif decoder == 'HOPLS':
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

    elif decoder == 'BTTR':
        ECoG_train = select_ecog_features(ECoG_train, window_len=10, freq_idx=None)
        ECoG_val = select_ecog_features(ECoG_val, window_len=10, freq_idx=None)
        ECoG_test = select_ecog_features(ECoG_test, window_len=10, freq_idx=None)

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

    # test
    corr_test = []
    for i in range(y_pred.shape[1]):
        corr, _ = pearsonr(y_pred[:, i], trajectory_test[:, i])
        corr_test.append(corr)
        
    corr_mean = np.mean(corr_test)
    corr_test_str = ", ".join([f"{c:.4f}" for c in corr_test])
    print(f"subject{iS}: test corr = {corr_test_str}, mean = {corr_mean:.4f}\n", flush=True)

    corr_allSub[:, iS] = corr_test
    traj_true[f'sub{iS}'].append(trajectory_test)
    traj_pred[f'sub{iS}'].append(y_pred)
            
print(f"all subject corr:\n{corr_allSub} \n mean = {np.mean(corr_allSub): .4f} ", flush=True)

# ---- save group corr ----
save_fileName = dataset + '_' + decoder + '_o5_cc.npy'
cc_save_path = os.path.join(save_root, save_fileName)
np.save(cc_save_path, corr_allSub)

# ---- save group trajectory ----
traj_save = {
    'true': traj_true,
    'pred': traj_pred
}
save_fileName = dataset + '_' + decoder + '_o5_trajectory.npz'
traj_save_path = os.path.join(save_root, save_fileName)
np.savez(traj_save_path, **traj_save)
