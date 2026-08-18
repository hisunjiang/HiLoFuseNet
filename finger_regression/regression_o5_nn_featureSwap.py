"""
DATASET: BCIIV and Stanford datasets
Train the neural nets with validation dataset. Output 5 finger trajectories

- 2026/8/6
    Test LSTM, MLP, HiLoFuseNet with wavelet feature input (same input as CNN-LSTM).

"""

import argparse

import random
import os
import numpy as np
from scipy.io import loadmat
import gc

import pickle
from models.prepareDataset import Scaler3D, Scaler4D, constructDataset, BatchShuffleSampler, prepare_taskFormatedData, select_ecog_features
from models.nn_regressors import MLP, LSTM, EEGNet, DeepFingerNet, CNN_LSTM, WaT, WaTFi, WaTEi, HiLoFuseNet
from models.nn_train_and_test import train, validation, test, EarlyStopping_performance
from models.nn_lossFunc import MSELoss

import torch
from torch.utils.data import DataLoader

cuda_available = torch.cuda.is_available()
print(f"I am in! CUDA is available: {cuda_available}")

device = torch.device('cuda:0')

# random seeds for model initialization
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_float32_matmul_precision("highest")
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# random seeds for dataloader
def seed_worker(worker_id):
    np.random.seed(torch.initial_seed() % (2**32))
    random.seed(torch.initial_seed() % (2**32))
            
# some settings listed below:
epoches = 200
batch_size = 128 # 16, 32, 48, 64, 80, 96, 112, 128, 142, 160

# input from the .slurm file
parser = argparse.ArgumentParser(description='Finger Regression Task')
parser.add_argument('--dataset', type=str, default='BCIIV',
                    help='Dataset name') 
parser.add_argument('--decoder', type=str, default='HiLoFuseNet',
                    help='Decoder type') 
parser.add_argument('--lossFunc', type=str, default='mse',
                    help='Loss function')
parser.add_argument('--seed', type=int, default=42,
                    help='seed value')
parser.add_argument('--win_size', type=float, default=1,
                    help='lookback window') 

args = parser.parse_args()

dataset = args.dataset
decoder = args.decoder
lossFunc = args.lossFunc
seed = args.seed
win_size = args.win_size
feature_type = 'wavelet_5_195Hz'

save_root = 'results/o5/'
os.makedirs(save_root, exist_ok=True)

# metadata for different datasets
datasets = {
    "BCIIV": {
        "subjects": 3,
        "subject_name": ['sub1', 'sub2', 'sub3'],
        "fs_ecog": 1000,
        "fs_dg": 25,
        # file path to preprocessed data
        "path": '/data/leuven/355/vsc35565/data/BCIIV/'
    },
    "Stanford": {
        "subjects": 9,
        "subject_name": ['bp', 'cc', 'ht','jc','jp','mv','wc','wm','zt'],
        "fs_ecog": 1000,
        "fs_dg": 25,
        "path": '/data/leuven/355/vsc35565/data/Stanford/'
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
    
    # initialize the random seeds
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    
    # load data
    data = loadmat(fileLoc + datasets[dataset]['subject_name'][iS] + '.mat')
    
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
    
    # select a model
    if decoder == 'MLP':
        ECoG_train = select_ecog_features(ECoG_train, window_len=10)
        ECoG_val = select_ecog_features(ECoG_val, window_len=10)
        ECoG_test = select_ecog_features(ECoG_test, window_len=10)
    
        _, C, T, F = ECoG_train.shape
        model = MLP(input_size=C*T*F, hidden_size=256, output_size=5,
                     dropout_prob=0.5)
                     
    elif decoder == 'LSTM':
        ECoG_train = select_ecog_features(ECoG_train, window_len=10)
        ECoG_val = select_ecog_features(ECoG_val, window_len=10)
        ECoG_test = select_ecog_features(ECoG_test, window_len=10)
    
        _, C, T, F = ECoG_train.shape
        model = LSTM(input_size=C*F, hidden_size=256, output_size=5,
                     dropout_prob=0.5)
    
    elif decoder == 'HiLoFuseNet':
        _, C, T, F = ECoG_train.shape
        model = HiLoFuseNet(C=C, F=F, lstm_hidden=256, D=20,
                          output_size=5, dropout_prob=0.5)
                        
    model.to(device)
    
    # Pytorch data format
    trainDataset = constructDataset(ECoG_train, trajectory_train)
    valDataset = constructDataset(ECoG_val, trajectory_val)
    testDataset = constructDataset(ECoG_test, trajectory_test)
    
    # sampler
    sampler_train = BatchShuffleSampler(trainDataset, batch_size)
    train_loader = DataLoader(trainDataset, batch_size=batch_size, sampler=sampler_train, worker_init_fn=seed_worker,
                              generator=g)
    val_loader = DataLoader(valDataset, batch_size=64, shuffle=False, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(testDataset, batch_size=64, shuffle=False, worker_init_fn=seed_worker, generator=g)
    
    if lossFunc == 'mse':
        loss_function = MSELoss()
    else:
        pass
    
    optimizer = torch.optim.Adam(list(model.parameters()), lr=1*1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
        )
    
    # training
    early_stopping = EarlyStopping_performance(patience=15)
    for epoch in range(epoches):
        # -------- train --------
        loss_train, corr_train = train(train_loader, model, optimizer, loss_function, device)
        # corr_train_str = ", ".join([f"{c:.4f}" for c in corr_train])
        # print(f"[Subject: {iS}, Epoch: {epoch}] Train Loss: {loss_train:.4f}, Corr: {corr_train_str}")
    
        # -------- validation --------
        loss_val, corr_val = validation(val_loader, model, loss_function, device)
        corr_val_mean = np.mean(corr_val)
        # corr_val_str = ", ".join([f"{c:.4f}" for c in corr_val])
        # print(f"[...] Val   Loss: {loss_val:.4f}, Corr: {corr_val_str}")
    
        # update LR scheduler
        scheduler.step(corr_val_mean)
    
        # -------- Early Stopping --------
        stop_now = early_stopping(corr_val_mean, model, epoch)
        if stop_now:
            print(f"Early stopping triggered at epoch {epoch}.")
            break
    
    # test
    early_stopping.load_best_model(model)
    corr_test, traj_target, traj_predict = test(test_loader, model, device)
    corr_mean = np.mean(corr_test)
    corr_test_str = ", ".join([f"{c:.4f}" for c in corr_test])
    print(f"subject{iS}: test corr = {corr_test_str}, mean = {corr_mean:.4f}\n", flush=True)
    
    corr_allSub[:, iS] = corr_test
    traj_true[f'sub{iS}'].append(traj_target)
    traj_pred[f'sub{iS}'].append(traj_predict)
    
    # clean cache
    del data, ECoG_train, ECoG_val, ECoG_test
    del trainDataset, valDataset, testDataset
    del train_loader, val_loader, test_loader
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

print(f"all subject corr:\n{corr_allSub} \n mean = {np.mean(corr_allSub): .4f} ", flush=True)

# ---- save group corr ----
save_fileName_cc = f"{dataset}_{decoder}_featureSwap_{lossFunc}_seed{seed}_win{win_size}_o5_cc.npy"
cc_save_path = os.path.join(save_root, save_fileName_cc)
np.save(cc_save_path, corr_allSub)

# ---- save group trajectory ----
traj_save = {
    'true': traj_true,
    'pred': traj_pred
}
save_fileName_traj = f"{dataset}_{decoder}_featureSwap_{lossFunc}_seed{seed}_win{win_size}_o5_trajectory.npz"
traj_save_path = os.path.join(save_root, save_fileName_traj)
np.savez(traj_save_path, **traj_save)
