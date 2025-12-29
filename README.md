
## 🧠 **Hi**gh-gamma and **Lo**w-frequency ECoG signal **Fus**ion **Net**work (**HiLoFuseNet**) for **continuous finger movement decoding**
This repository provides the official PyTorch implementation of the finger movement decoding framework detailed in the paper:
> Sun et al., "Spectro-Temporal Fusion of High-Gamma and Low-Frequency ECoG Signals for Intracranial Finger Movement Decoding," 2025. *under review*.

The model achieved SOTA decoding performance on the public BCI Competition IV and Stanford datasets, with Pearson correlation coefficients between true and predicted finger movement trajectories of 0.631 and 0.534, representing improvements of 5.0% and 11.9%, respectively, over the previous best methods.

<img src="SOTA_comparison.png" alt="Comparison with previous previous studies on the BCIIV (blue) and Stanford (red) datasets." width="45%" />

---

## 🛠️ Decoding Framework
The proposed framework is characterized by (a) a streamlined ECoG feature extraction pipeline and (b) a compact neural network for learning spectro-temporal information.

<img src="model.png" alt="The HiLoFuseNet model architecture." width="70%" />

### Core Functions

| Functionality | Implementation Path |
| :--- | :--- |
| HGA and LFS Feature Extraction | `finger_regression/models/prepareDataset.py/HGALFS_feature_extractor` |
| HiLoFuseNet Architecture | `finger_regression/models/nn_regressors.py/HiLoFuseNet` |

---

## 💾 Quick Start & Reproducibility

### 1. Download Datasets
* BCIIV: [https://www.bbci.de/competition/iv/#dataset4](https://www.bbci.de/competition/iv/#dataset4)
* Stanford-FingerFlex: [https://searchworks.stanford.edu/view/zk881ps0522](https://searchworks.stanford.edu/view/zk881ps0522)

### 2. Signal Preprocessing
Raw ECoG signals were preprocessed using **MATLAB FieldTrip-20230926**.
* BCIIV Preprocessing: `data_preprocessing/data_preprocessing_BCI4.m`
* Stanford Preprocessing: `data_preprocessing/data_preprocessing_Stanford.m`

### 3. Feature Extraction
Features were extracted using **MNE-Python (v1.8.0)**.
* Script: `finger_regression/prepare_taskFormatedData.py`

### 4. Run Experiments
Configure the pytorch environment via `finger_regression/environment.yml`. The following table summarizes the scripts used to reproduce the paper's findings. Raw output files are provided in `finger_regression/results`. The .slum file provides the code to interact with a supercomputing cluster. If you run the script locally, please change the inputs in the .py file according to the corresponding settings from the .slurm file. 

| Experiment | Execution Script(s) | Results Folder |
| :--- | :--- | :--- |
| DNN Multi-Output Regression | `regression_o5_nn.py`, `submit_o5_nn.slurm` | `finger_regression/results/o5/varyingSeed` |
| ML Multi-Output Regression | `regression_o5_ml.py`, `submit_o5_ml.slurm` | `finger_regression/results/o5` |
| Model Interpretation | `regression_o5_nn_interpretModel.py`, `submit_o5_nn_interpretModel.slurm` | `finger_regression/results/o5/interpretModel` |
| Ablation Study | `regression_o5_nn_ablation.py`, `submit_o5_nn_ablation.slurm` | `finger_regression/results/o5/ablation` |
| Hyperparameter Test | `regression_o5_nn_hyperparameter.py`, `submit_o5_nn_hyperparameter.slurm` | `finger_regression/results/o5/hyperparameter` |
| DNN Single-Output Regression | `regression_o1_nn.py`, `submit_o1_nn.slurm` | `finger_regression/results/o1/varyingSeed` |

## Acknowledgement
A sincere thanks to the code contributors for BTTR and HOPLS!
* BTTR: [https://github.com/TheAxeC/block-term-tensor-regression](https://github.com/TheAxeC/block-term-tensor-regression)
* HOPLS: [https://github.com/arthurdehgan/HOPLS](https://github.com/arthurdehgan/HOPLS)

## Citation
Hope this model helps your research. We would be appreciated if u cite us.

```
to be appeared...
