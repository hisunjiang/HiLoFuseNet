"""
DESCRIPTION:
    Prepare the ECoG and trajectory data in a task format, i.e., pair prior 1-s ECoG epoch with the next trjectory point.
"""

from scipy.io import loadmat
import pickle
from models.prepareDataset import functionalBand_feature_extractor, HGALFS_feature_extractor, wavelet_feature_extractor

# metadata for different datasets
datasets = {
    "BCIIV": {
        "subjects": 3,
        "fs_ecog": 1000,
        "fs_dg": 25,
        # note: this is the path of preprocessed data
        "path": 'data_preprocessing/preprocessed_data/BCIIV/'
    },
    "Stanford": {
        "subjects": 9,
        "subject_name": ['bp', 'cc', 'ht','jc','jp','mv','wc','wm','zt'], 
        "fs_ecog": 1000,
        "fs_dg": 25,
        "path": 'data_preprocessing/preprocessed_data/Stanford/'
    }
}

# select one dataset for analysis
selected = 'BCIIV' # BCIIV Stanford
fileLoc = datasets[selected]['path']

if selected == 'BCIIV':
    fs_ecog = datasets[selected]['fs_ecog']
    fs_dg = datasets[selected]['fs_dg']

    for iS in range(datasets[selected]['subjects']):
        fileName = fileLoc + 'sub' + str(iS+1) + '.mat'
        data = loadmat(fileName) # train_data [channel, time], test_data [channel, time], train_dg [time, 5], test_dg [time, 5]
        
        # Segment the continuous ECoG data and transform each segment into features. Each segment associates with a target finger trajectory
        # sampled at 25Hz.
        # OUTPUT: ECoG -> [nEpoch, nChannel, nSequence, nFrequency], trajectory -> [nTarget (=nEpoch), 5]
        
        # ---- HGALFS_feature_extractor ------ #
        ECoG_train, trajectory_train = HGALFS_feature_extractor(data['train_data'].T, data['train_dg'], fs_ecog, window_size= 1*fs_ecog,
                                                                       step_size = fs_ecog // fs_dg, T=200, delay=40)
        ECoG_test, trajectory_test = HGALFS_feature_extractor(data['test_data'].T, data['test_dg'], fs_ecog, window_size= 1*fs_ecog,
                                                                     step_size = fs_ecog // fs_dg, T=200, delay=40)
        filename = fileLoc + 'features/' + f"sub{iS + 1}_HGALFS_features_200seq.pkl"

        # ---- functionalBand_feature_extractor ------ #
        # ECoG_train, trajectory_train = functionalBand_feature_extractor(data['train_data'].T, data['train_dg'], fs_ecog,
        #                                                        window_size=1 * fs_ecog,
        #                                                        step_size=fs_ecog // fs_dg, T=200, delay=40)
        # ECoG_test, trajectory_test = functionalBand_feature_extractor(data['test_data'].T, data['test_dg'], fs_ecog,
        #                                                      window_size=1 * fs_ecog,
        #                                                      step_size=fs_ecog // fs_dg, T=200, delay=40)
        # filename = fileLoc + 'features/' + f"sub{iS + 1}_functionalBand_features_200seq.pkl"

        # ---- wavelet_feature_extractor ------ #
        # ECoG_train, trajectory_train = wavelet_feature_extractor(data['train_data'].T, data['train_dg'], fs_ecog, window_size= 1*fs_ecog,
        #                                                          step_size = fs_ecog // fs_dg, T=100, batch_size=64, delay=40)
        # ECoG_test, trajectory_test = wavelet_feature_extractor(data['test_data'].T, data['test_dg'], fs_ecog,
        #                                                          window_size=1 * fs_ecog,
        #                                                          step_size = fs_ecog // fs_dg, T=100,
        #                                                          batch_size=64, delay=40)
        # filename = fileLoc + 'features/' + f"sub{iS + 1}_wavelet_features_100seq.pkl"

        # save features
        with open(filename, "wb") as f:
            pickle.dump([ECoG_train, trajectory_train, ECoG_test, trajectory_test], f)
        
        print(f"Finished subject: {iS+1}\n", flush=True)
            
elif selected == 'Stanford':
    fs_ecog = datasets[selected]['fs_ecog']
    fs_dg = datasets[selected]['fs_dg']
    
    for iS in range(datasets[selected]['subjects']): #datasets[selected]['subjects']
        fileName = fileLoc + datasets[selected]['subject_name'][iS] + '.mat'
        data1 = loadmat(fileName) # data [channel, time], flex [time, 5]
    
        # get data
        ECoG_1k = data1['data'].T
        trajectory_1k = data1['flex']

        # split train and test dataset
        if ECoG_1k.shape[0] // fs_ecog >= 600:
            ECoG_1k_train, trajectory_1k_train = ECoG_1k[0:400*fs_ecog, :], trajectory_1k[0:400*fs_ecog, :]
            ECoG_1k_test, trajectory_1k_test = ECoG_1k[400*fs_ecog:, :], trajectory_1k[400*fs_ecog:, :]
        else:
            len = ECoG_1k.shape[0] // 3 *2
            ECoG_1k_train, trajectory_1k_train = ECoG_1k[0:len, :], trajectory_1k[0:len, :]
            ECoG_1k_test, trajectory_1k_test = ECoG_1k[len:, :], trajectory_1k[len:, :]

        # ---- HGALFS_feature_extractor ------ #
        ECoG_train, trajectory_train = HGALFS_feature_extractor(ECoG_1k_train, trajectory_1k_train, fs_ecog,
                                                                       window_size=1 * fs_ecog,
                                                                       step_size=fs_ecog // fs_dg, T=200, delay=40)
        ECoG_test, trajectory_test = HGALFS_feature_extractor(ECoG_1k_test, trajectory_1k_test, fs_ecog,
                                                                     window_size=1 * fs_ecog,
                                                                     step_size=fs_ecog // fs_dg, T=200, delay=40)
        filename = fileLoc + 'features/' + f"{datasets[selected]['subject_name'][iS]}_HGALFS_features_200seq.pkl"

        # ---- wavelet_feature_extractor ------ #
        # ECoG_train, trajectory_train = wavelet_feature_extractor(ECoG_1k_train, trajectory_1k_train, fs_ecog,
        #                                                          window_size=1 * fs_ecog,
        #                                                          step_size=fs_ecog // fs_dg, T=100, batch_size=64,
        #                                                          delay=40)
        # ECoG_test, trajectory_test = wavelet_feature_extractor(ECoG_1k_test, trajectory_1k_test, fs_ecog,
        #                                                        window_size=1 * fs_ecog,
        #                                                        step_size=fs_ecog // fs_dg, T=100,
        #                                                        batch_size=64, delay=40)
        # filename = fileLoc + 'features/' + f"{datasets[selected]['subject_name'][iS]}_wavelet_features_100seq.pkl"

        with open(filename, "wb") as f:
            pickle.dump([ECoG_train, trajectory_train, ECoG_test, trajectory_test], f)
        
        print(f"Finished subject: {iS}\n", flush=True)

else: pass

    
    