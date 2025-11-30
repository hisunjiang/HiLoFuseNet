
## 🧠 **Hi**gh-gamma and **Lo**w-frequency ECoG signal **Fus**ion **Net**work (**HiLoFuseNet**) for **continuous finger movement decoding**
This repository provides the official PyTorch implementation of the **Hi**gh-gamma and **Lo**w-frequency ECoG signal **Fus**ion **Net**work (**HiLoFuseNet**), a novel framework for **continuous finger movement decoding** from electrocorticography (ECoG).

HiLoFuseNet is designed to efficiently integrate neurophysiologically distinct spectral components, achieving **State-of-the-Art (SOTA)** performance in Brain-Computer Interface (BCI) applications.

The method is detailed in the paper:
> Sun et al., "Spectro-Temporal Fusion of High-Gamma and Low-Frequency ECoG Signals for Intracranial Finger Movement Decoding," 2025. *under review*.

---

## 🚀 Performance
HiLoFuseNet significantly outperforms previous best methods on public datasets, as measured by the Pearson correlation coefficient ($r$):

| Dataset | Metric (Pearson $r$) | HiLoFuseNet Score | Improvement over Previous Best |
| :--- | :---: | :---: | :---: |
| **BCI Competition IV** | 0.631 | **5.0%** |
| **Stanford-FingerFlex** | 0.534 | **11.9%** |

<img src="SOTA_comparison.png" alt="Comparison with previous previous studies on the BCIIV (blue) and Stanford (red) datasets." width="45%" />

---

## 🛠️ Decoding Framework
The core contribution is a performant and neurophysiologically-sound framework featuring:

1.  **Streamlined ECoG Feature Extraction:** Focusing on the crucial High-Gamma Activity (HGA) and Low-Frequency Signals (LFS).
2.  **Compact Neural Network (HiLoFuseNet):** A specialized architecture for learning fused spectro-temporal information.

<img src="model.png" alt="The HiLoFuseNet model architecture." width="70%" />

### Core Functions

| Functionality | Implementation Path |
| :--- | :--- |
| HGA and LFS Feature Extraction | `finger_regression/models/prepareDataset.py/HGALFS_feature_extractor` |
| HiLoFuseNet Architecture | `finger_regression/models/nn_regressors.py/HiLoFuseNet` |

---

## 💾 Quick Start & Reproducibility

### 1. Download Datasets
* **BCIIV:** [https://www.bbci.de/competition/iv/#dataset4](https://www.bbci.de/competition/iv/#dataset4)
* **Stanford-FingerFlex:** [https://searchworks.stanford.edu/view/zk881ps0522](https://searchworks.stanford.edu/view/zk881ps0522)

### 2. Signal Preprocessing
Raw ECoG signals were preprocessed using **MATLAB FieldTrip-20230926**.
* **BCIIV Preprocessing:** `data_preprocessing/data_preprocessing_BCI4.m`
* **Stanford Preprocessing:** `data_preprocessing/data_preprocessing_Stanford.m`

### 3. Feature Extraction
Features were extracted using **MNE-Python (v1.8.0)**.
* **Extraction Script:** `finger_regression/prepare_taskFormatedData.py`

### 4. Run Experiments
The environment is managed via `finger_regression/environment.yml`. The following table summarizes the scripts used to reproduce the paper's findings. Raw output files are provided in `finger_regression/results`.

| Experiment | Execution Script(s) | Results Folder |
| :--- | :--- | :--- |
| DNN Multi-Output Regression | `regression_o5_nn.py`, `submit_o5_nn.slurm` | `finger_regression/results/o5/varyingSeed` |
| ML Multi-Output Regression | `regression_o5_ml.py`, `submit_o5_ml.slurm` | `finger_regression/results/o5` |
| Model Interpretation | `regression_o5_nn_interpretModel.py`, `submit_o5_nn_interpretModel.slurm` | `finger_regression/results/o5/interpretModel` |
| Ablation Study | `regression_o5_nn_ablation.py`, `submit_o5_nn_ablation.slurm` | `finger_regression/results/o5/ablation` |
| Hyperparameter Test | `regression_o5_nn_hyperparameter.py`, `submit_o5_nn_hyperparameter.slurm` | `finger_regression/results/o5/hyperparameter` |
| DNN Single-Output Regression | `regression_o1_nn.py`, `submit_o1_nn.slurm` | `finger_regression/results/o1/varyingSeed` |
