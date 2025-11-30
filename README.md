# 🌟Overview
This repository provides the official PyTorch implementation of the **Hi**gh-gamma and **Lo**w-frequency ECoG signal **Fus**ion **Net**work (**HiLoFuseNet**) proposed in the paper: 
>Sun et al., "Spectro-Temporal Fusion of High-Gamma and Low-Frequency ECoG Signals for Intracranial Finger Movement Decoding," 2025. *under review*.

which achieves SOTA decoding performance on the public BCI Competition IV and Stanford datasets, with Pearson correlation coefficients between true and predicted finger movement trajectories of 0.631 and 0.534, representing improvements of 5.0\% and 11.9\%, respectively, over the previous best methods.

<img src="SOTA_comparison.png" alt="Comparison with previous previous studies on the BCIIV (blue) and Stanford (red) datasets." width="50%" />

## The proposed decoding framework
We aim to develop a performant and neurophysiologically-sound framework for continuous finger movement decoding from ECoG. This was achieved by proposing (a) a streamlined ECoG feature extraction pipeline and (b) a compact neural network for learning spectro-temporal information.

<img src="model.png" alt="The model." width="70%" />

## The core takeaway functions
- High-Gamma Activity (HGA) and Low-Frequency Signals (LFS) extraction
  > finger_regression/models/prepareDataset.py/HGALFS_feature_extractor
- HiLoFuseNet architecture
  > finger_regression/models/nn_regressors.py/HiLoFuseNet
