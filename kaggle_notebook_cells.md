# Kaggle Notebook Outline for SST-DPN

This outline recreates the full SST-DPN training pipeline inside a Kaggle Notebook without requiring the Git repository. Every module from the project is reproduced as code cells, and all run parameters are exposed as local variables that you can edit directly in the notebook.

### How this notebook mirrors the repository flow

| Repository component | Purpose | Notebook cell(s) |
| --- | --- | --- |
| `load_data.py` | Prepares and preprocesses the BNCI EEG datasets. | **Cell 4 – Data Loading Utilities** |
| `dpl_utils.py` | Implements the prototype-learning regularisers. | **Cell 5 – Prototype Loss Utilities** |
| `model_utils.py` | Provides shared neural-network building blocks. | **Cell 6 – Model Building Blocks** |
| `model.py` | Defines the SST-DPN architecture. | **Cell 7 – SST-DPN Model Definition** |
| `train.py` | Handles configuration, data preparation, two-phase training, checkpointing, and evaluation. | **Cells 2, 3, 8, 9, and 10** |

The optional `compare_model/` baselines are not required for the core training loop. You can add new notebook cells after Cell 10 if you wish to port those experiments.

---

## Cell 1 – Environment Setup
Install the runtime dependencies that Kaggle does not provide by default. Enable **Internet** in the notebook settings before running this cell.
```python
!pip install --quiet braindecode==0.8.1 moabb==1.1.0 resampy==0.4.2 scipy==1.11.4 einops==0.8.0
```

---

## Cell 2 – Configuration and Paths
Set up all configurable parameters as local variables and prepare the working directories that later cells will use. Adjust any values in the `CONFIG` dictionary to experiment with different settings.
```python
import os
from pathlib import Path

WORKING_ROOT = Path("/kaggle/working")
MOABB_CACHE = WORKING_ROOT / "moabb_datasets"
MOABB_CACHE.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "dataset_id": "2a",        # "2a" or "2b"
    "subject_id": 5,             # 1–9 for BNCI2014_001, 1–9 for BNCI2014_004
    "test_size": 0.2,            # validation split ratio
    "random_state": 42,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-2,
    "lr_isp": 1e-3,
    "lr_icp": 1e-3,
    "epochs_phase1": 1000,
    "patience_phase1": 200,
    "epochs_phase2": 300,
    "loss_weight_pl": 1e-3,
    "loss_weight_icp": 1e-5,
    "mvp_pool_kernels": [50, 100, 200],
    "num_classes": 4,
    "preprocessing": {
        "sfreq": 250,
        "low_cut": None,
        "high_cut": None,
        "start": 0,
        "stop": 0,
        "z_scale": False,
    },
}

# Expose the MOABB cache location to downstream libraries.
os.environ["MOABB_DATASETS"] = str(MOABB_CACHE)
BEST_MODEL_PATH = WORKING_ROOT / "best_model.pth"
```

---

## Cell 3 – Dataset Download and Extraction
Fetch the BNCI 2a/2b archives from the official BBCI links and extract them into the MOABB cache. This cell runs once per dataset; future sessions reuse the cached files.
```python
import zipfile
from urllib.request import urlretrieve

DATA_URLS = {
    "2a": "https://www.bbci.de/competition/download/competition_iv/BCICIV_2a_gdf.zip",
    "2b": "https://www.bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip",
}
DATASET_FOLDERS = {"2a": "BNCI2014_001", "2b": "BNCI2014_004"}
ARCHIVE_NAMES = {"2a": "BCICIV_2a_gdf.zip", "2b": "BCICIV_2b_gdf.zip"}
EXTRACTED_FOLDERS = {"2a": "BCICIV_2a_gdf", "2b": "BCICIV_2b_gdf"}

dataset_id = CONFIG["dataset_id"]
archive_path = MOABB_CACHE / ARCHIVE_NAMES[dataset_id]
target_folder = MOABB_CACHE / DATASET_FOLDERS[dataset_id]
extracted_folder = MOABB_CACHE / EXTRACTED_FOLDERS[dataset_id]

if not target_folder.exists() or not any(target_folder.glob("*.gdf")):
    if not archive_path.exists():
        print(f"Downloading BCIC {dataset_id.upper()} dataset…")
        urlretrieve(DATA_URLS[dataset_id], archive_path)
    else:
        print(f"Archive already present: {archive_path.name}")

    if not extracted_folder.exists():
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(MOABB_CACHE)

    if extracted_folder.exists() and not target_folder.exists():
        extracted_folder.rename(target_folder)

print("Dataset ready in:", target_folder)
```

---

## Cell 4 – Data Loading Utilities (`load_data.py`)
These functions reproduce the dataset preparation logic from `load_data.py`, including optional utilities for other BNCI datasets.
```python
import os
from typing import Dict

import numpy as np
import resampy
import scipy
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import (
    Preprocessor,
    create_windows_from_events,
    preprocess,
)
from numpy import multiply
from scipy.io import loadmat
from scipy.signal import filtfilt
from sklearn.preprocessing import StandardScaler


def z_scale(X, X_test):
    for ch_idx in range(X.shape[1]):
        sc = StandardScaler()
        X[:, ch_idx, :] = sc.fit_transform(X[:, ch_idx, :])
        X_test[:, ch_idx, :] = sc.transform(X_test[:, ch_idx, :])
    return X, X_test


def load_bcic(
    dataset_id: str = "2a",
    subject_id: int = 1,
    preprocessing_dict: Dict = None,
    verbose: str = "WARNING",
):
    dataset_name = "BNCI2014_001" if dataset_id == "2a" else "BNCI2014_004"
    dataset = MOABBDataset(dataset_name, subject_ids=[subject_id])

    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False, verbose=verbose),
        Preprocessor(lambda data: multiply(data, 1e6)),
    ]

    # filtering or not
    l_freq, h_freq = preprocessing_dict["low_cut"], preprocessing_dict["high_cut"]
    if l_freq is not None or h_freq is not None:
        preprocessors.append(
            Preprocessor("filter", l_freq=l_freq, h_freq=h_freq, verbose=verbose)
        )

    # resample or not
    if dataset.datasets[0].raw.info["sfreq"] != preprocessing_dict["sfreq"]:
        preprocessors.append(
            Preprocessor(
                "resample", sfreq=preprocessing_dict["sfreq"], verbose=verbose
            ),
        )

    preprocess(dataset, preprocessors)

    # create windows
    sfreq = dataset.datasets[0].raw.info["sfreq"]
    trial_start_offset_samples = int(preprocessing_dict["start"] * sfreq)
    trial_stop_offset_samples = int(preprocessing_dict["stop"] * sfreq)
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=trial_stop_offset_samples,
        preload=False,
    )

    # split the data
    splitted_ds = windows_dataset.split("session")
    if dataset_id == "2a":
        train_dataset, test_dataset = (
            splitted_ds["0train"],
            splitted_ds["1test"],
        )

        # load the data
        X = np.stack([sample[0] for sample in train_dataset], axis=0)
        y = np.stack([sample[1] for sample in train_dataset], axis=0)
        X_test = np.stack([sample[0] for sample in test_dataset], axis=0)
        y_test = np.stack([sample[1] for sample in test_dataset], axis=0)

    elif dataset_id == "2b":
        train_datasets = [splitted_ds[f"{session}train"] for session in [0, 1, 2]]
        test_datasets = [splitted_ds[f"{session}test"] for session in [3, 4]]
        # load the data
        X_sess, y_sess, X_test_sess, y_test_sess = [], [], [], []
        for train_dataset in train_datasets:
            X_sess.append(np.stack([sample[0] for sample in train_dataset], axis=0))
            y_sess.append(np.stack([sample[1] for sample in train_dataset], axis=0))
        for test_dataset in test_datasets:
            X_test_sess.append(np.stack([sample[0] for sample in test_dataset], axis=0))
            y_test_sess.append(np.stack([sample[1] for sample in test_dataset], axis=0))

        X = np.concatenate(X_sess)
        y = np.concatenate(y_sess)
        X_test = np.concatenate(X_test_sess)
        y_test = np.concatenate(y_test_sess)
    if preprocessing_dict["z_scale"]:
        X, X_test = z_scale(X, X_test)
    return X, y, X_test, y_test


def bandpass_cheby2(data, low_cut_hz, high_cut_hz, fs, n=6, rs=60):
    b, a = scipy.signal.cheby2(
        N=n,
        rs=rs,
        Wn=[low_cut_hz, high_cut_hz],
        btype="bandpass",
        analog=False,
        output="ba",
        fs=fs,
    )
    data_bandpassed = filtfilt(
        b, a, data, axis=-1, padlen=3 * (max(len(b), len(a)) - 1)
    )
    return data_bandpassed


# BCIC 3
def load_bci3(dataPath, subject_id, preprocessing_dict):
    sub = {1: "aa", 2: "al", 3: "av", 4: "aw", 5: "ay"}
    path = os.path.join(
        dataPath,
        f"data_set_IVa_{sub[subject_id]}_mat",
        "100Hz",
        f"data_set_IVa_{sub[subject_id]}.mat",
    )
    label_path = os.path.join(dataPath, f"true_labels_{sub[subject_id]}.mat")
    mat = loadmat(path)
    mat_labels = loadmat(label_path)
    data = mat["cnt"].T
    marker = mat["mrk"][0][0][0]
    labels = mat_labels["true_y"]
    test_idx = mat_labels["test_idx"]

    sfreq = mat["nfo"]["fs"][0][0][0][0]
    ch_names = [_[0] for _ in mat["nfo"]["clab"][0][0][0]]

    channels = ["C3", "Cz", "C4"]
    channel_selection = preprocessing_dict.get("channel_selection", False)
    if channel_selection:
        channels_indices = [ch_names.index(ch) for ch in channels]
        data = data[channels_indices, :]

    l_freq, h_freq = preprocessing_dict["low_cut"], preprocessing_dict["high_cut"]
    if l_freq is not None or h_freq is not None:
        data = bandpass_cheby2(data, l_freq, h_freq, sfreq)

    trial_length_second = preprocessing_dict["stop"] - preprocessing_dict["start"]

    start = int(sfreq * preprocessing_dict["start"])
    stop = int(sfreq * preprocessing_dict["stop"])
    trial_length = stop - start
    trials = np.zeros((labels.shape[-1], data.shape[0], trial_length))
    for i, m in enumerate(marker[0]):
        trials[i, ::] = data[:, m + start : m + stop]

    if preprocessing_dict["sfreq"] != sfreq:
        trials = resampy.resample(
            trials,
            sfreq,
            preprocessing_dict["sfreq"],
            axis=-1,
        )

    labels = labels.squeeze() - 1

    X = trials[test_idx == 0, :, :]
    y = labels[test_idx == 0]

    X_test = trials[test_idx == 1, :, :]
    y_test = labels[test_idx == 1]

    if preprocessing_dict["z_scale"]:
        X, X_test = z_scale(X, X_test)

    return X, y, X_test, y_test


def load_hgd(dataPath, subject_id, preprocessing_dict):
    path = os.path.join(
        dataPath,
        f"subject_{subject_id:02d}",
        f"subject_{subject_id:02d}.mat",
    )

    mat = loadmat(path)

    sfreq = mat["nfo"]["fs"][0][0]

    data = mat["cnt"].T

    marker = mat["mrk"]["pos"][0]
    labels = mat["mrk"]["y"][0][0][0]

    ch_names = [_[0] for _ in mat["nfo"]["clab"][0]]

    channels = ["C3", "Cz", "C4"]
    channel_selection = preprocessing_dict.get("channel_selection", False)
    if channel_selection:
        channels_indices = [ch_names.index(ch) for ch in channels]
        data = data[channels_indices, :]

    l_freq, h_freq = preprocessing_dict["low_cut"], preprocessing_dict["high_cut"]
    if l_freq is not None or h_freq is not None:
        data = bandpass_cheby2(data, l_freq, h_freq, sfreq)

    start = int(sfreq * preprocessing_dict["start"])
    stop = int(sfreq * preprocessing_dict["stop"])
    trial_length = stop - start

    trials = np.zeros((labels.shape[-1], data.shape[0], trial_length))
    for i, m in enumerate(marker):
        trials[i, ::] = data[:, m + start : m + stop]

    if preprocessing_dict["sfreq"] != sfreq:
        trials = resampy.resample(
            trials,
            sfreq,
            preprocessing_dict["sfreq"],
            axis=-1,
        )

    labels = labels.squeeze() - 1

    X = trials
    y = labels

    X_test = trials
    y_test = labels

    if preprocessing_dict["z_scale"]:
        X, X_test = z_scale(X, X_test)

    return X, y, X_test, y_test
```
```

---

## Cell 5 – Prototype Loss Utilities (`dpl_utils.py`)
```python
import torch
import torch.nn as nn


class PrototypeLoss(nn.Module):
    def forward(self, features, proxy, labels):
        label_prototypes = torch.index_select(proxy, dim=0, index=labels)
        pl = huber_loss(features, label_prototypes, sigma=1)
        pl_loss = torch.mean(pl)
        return pl_loss


def huber_loss(input, target, sigma=1):
    beta = 1.0 / (sigma**2)
    diff = torch.abs(input - target)
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return torch.sum(loss, dim=1)


class NormIncreaseLoss(nn.Module):
    def forward(self, mat):
        norms = torch.norm(mat, p=2, dim=1)
        loss = -norms
        return loss.mean()
```

---

## Cell 6 – Model Building Blocks (`model_utils.py`)
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LightweightConv1d(nn.Module):
    def __init__(
        self,
        in_channels,
        num_heads=1,
        depth_multiplier=1,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
        weight_softmax=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.num_heads = num_heads
        self.padding = padding
        self.weight_softmax = weight_softmax
        self.weight = nn.Parameter(
            torch.Tensor(num_heads * depth_multiplier, 1, kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_heads * depth_multiplier))
        else:
            self.bias = None

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, inp):
        B, C, T = inp.size()
        H = self.num_heads

        weight = self.weight
        if self.weight_softmax:
            weight = F.softmax(weight, dim=-1)

        inp = rearrange(inp, "b (h c) t -> (b c) h t", h=H)
        if self.bias is None:
            output = F.conv1d(
                inp,
                weight,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        else:
            output = F.conv1d(
                inp,
                weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                groups=self.num_heads,
            )
        output = rearrange(output, "(b c) h t -> b (h c) t", b=B)

        return output


class VarMaxPool1D(nn.Module):
    def __init__(self, T, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, x):
        mean_of_squares = F.avg_pool1d(
            x**2, self.kernel_size, self.stride, self.padding
        )
        square_of_mean = (
            F.avg_pool1d(x, self.kernel_size, self.stride, self.padding) ** 2
        )
        variance = mean_of_squares - square_of_mean
        out = F.avg_pool1d(variance, variance.shape[-1])
        return out


class VarPool1D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = kernel_size
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        self.padding = padding

    def forward(self, x):
        mean_of_squares = F.avg_pool1d(
            x**2, self.kernel_size, self.stride, self.padding
        )
        square_of_mean = (
            F.avg_pool1d(x, self.kernel_size, self.stride, self.padding) ** 2
        )
        variance = mean_of_squares - square_of_mean
        return variance


class SSA(nn.Module):
    # Spatial-Spectral Attention
    def __init__(self, T, num_channels, epsilon=1e-5, mode="var", after_relu=False):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

        self.GP = VarMaxPool1D(T, 250)

    def forward(self, x):
        B, C, T = x.shape

        if self.mode == "l2":
            embedding = (x.pow(2).sum((2), keepdim=True) + self.epsilon).pow(0.5)
            norm = self.gamma / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        elif self.mode == "l1":
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2), keepdim=True)
            norm = self.gamma / (
                torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon
            )

        elif self.mode == "var":
            embedding = (self.GP(x) + self.epsilon).pow(0.5) * self.alpha
            norm = (self.gamma) / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        gate = 1 + torch.tanh(embedding * norm + self.beta)

        return x * gate, gate


class Mixer1D(nn.Module):
    def __init__(self, dim, kernel_sizes=[50, 100, 250]):
        super().__init__()
        self.var_layers = nn.ModuleList()
        self.L = len(kernel_sizes)
        for k in kernel_sizes:
            self.var_layers.append(
                nn.Sequential(
                    VarPool1D(kernel_size=k, stride=int(k / 2)),
                    nn.Flatten(start_dim=1),
                )
            )

    def forward(self, x):
        B, d, L = x.shape
        x_split = torch.split(x, d // self.L, dim=1)
        out = []
        for i in range(len(x_split)):
            x = self.var_layers[i](x_split[i])
            out.append(x)
        y = torch.concat(out, dim=1)
        return y
```

---

## Cell 7 – SST-DPN Model Definition (`model.py`)
```python
import math

import torch
from torch import nn


class Efficient_Encoder(nn.Module):
    def __init__(
        self,
        samples,
        chans,
        F1=16,
        F2=36,
        time_kernel1=75,
        pool_kernels=[50, 100, 250],
    ):
        super().__init__()

        self.time_conv = LightweightConv1d(
            in_channels=chans,
            num_heads=1,
            depth_multiplier=F1,
            kernel_size=time_kernel1,
            stride=1,
            padding="same",
            bias=True,
            weight_softmax=False,
        )
        self.ssa = SSA(samples, chans * F1)

        self.chanConv = nn.Sequential(
            nn.Conv1d(
                chans * F1,
                F2,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm1d(F2),
            nn.ELU(),
        )

        self.mixer = Mixer1D(dim=F2, kernel_sizes=pool_kernels)

    def forward(self, x):
        x = self.time_conv(x)
        x, _ = self.ssa(x)
        x_chan = self.chanConv(x)
        feature = self.mixer(x_chan)
        return feature


class SST_DPN(nn.Module):
    def __init__(
        self,
        chans,
        samples,
        num_classes=4,
        F1=9,
        F2=48,
        time_kernel1=75,
        pool_kernels=[50, 100, 200],
    ):
        super().__init__()
        self.encoder = Efficient_Encoder(
            samples=samples,
            chans=chans,
            F1=F1,
            F2=F2,
            time_kernel1=time_kernel1,
            pool_kernels=pool_kernels,
        )
        self.features = None

        x = torch.ones((1, chans, samples))
        out = self.encoder(x)
        feat_dim = out.shape[-1]

        # Inter-class Separation Prototype (ISP)
        self.isp = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        # Intra-class Compactness (ICP)
        self.icp = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        nn.init.kaiming_normal_(self.isp)

    def get_features(self):
        if self.features is not None:
            return self.features
        raise RuntimeError("No features available. Run forward() first.")

    def forward(self, x):
        features = self.encoder(x)
        self.features = features
        self.isp.data = torch.renorm(self.isp.data, p=2, dim=0, maxnorm=1)
        logits = torch.einsum("bd,cd->bc", features, self.isp)
        return logits
```

---

## Cell 8 – Training Helpers
Set up PyTorch, define checkpoint utilities, and create helper functions for validation and both training phases.
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, cohen_kappa_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, optimizer, optimizer4isp, optimizer4icp, filepath):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "optimizer4isp_state_dict": optimizer4isp.state_dict(),
        "optimizer4icp_state_dict": optimizer4icp.state_dict(),
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, optimizer4isp, optimizer4icp, filepath, map_location):
    checkpoint = torch.load(filepath, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer4isp.load_state_dict(checkpoint["optimizer4isp_state_dict"])
    optimizer4icp.load_state_dict(checkpoint["optimizer4icp_state_dict"])


def validate(model, val_loader, criterion, loss_pl, loss_icp, config):
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == y_batch).sum().item()
            total_val += y_batch.size(0)
            cls_loss = criterion(outputs, y_batch)
            icp_loss = loss_icp(model.icp)
            feature = model.get_features()
            proxy = model.icp
            pl_loss = loss_pl(feature, proxy, y_batch)
            total_loss = (
                cls_loss
                + config["loss_weight_pl"] * pl_loss
                + config["loss_weight_icp"] * icp_loss
            )
            val_loss += cls_loss.item()
    val_accuracy = correct_val / max(total_val, 1)
    return val_loss / max(len(val_loader), 1), val_accuracy


def train_phase_one(
    model,
    train_loader,
    val_loader,
    optimizer,
    optimizer4isp,
    optimizer4icp,
    criterion,
    loss_pl,
    loss_icp,
    config,
    checkpoint_path,
):
    best_loss = float("inf")
    patience_counter = 0
    train_acc_history = []
    val_acc_history = []

    for epoch in range(config["epochs_phase1"]):
        model.train()
        correct_train = 0
        total_train = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            optimizer4isp.zero_grad()
            optimizer4icp.zero_grad()
            outputs = model(X_batch)
            cls_loss = criterion(outputs, y_batch)
            icp_loss = loss_icp(model.icp)
            feature = model.get_features()
            proxy = model.icp
            pl_loss = loss_pl(feature, proxy, y_batch)
            total_loss = (
                cls_loss
                + config["loss_weight_pl"] * pl_loss
                + config["loss_weight_icp"] * icp_loss
            )
            total_loss.backward()
            optimizer.step()
            optimizer4isp.step()
            optimizer4icp.step()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)

        train_accuracy = correct_train / max(total_train, 1)
        val_loss, val_accuracy = validate(
            model, val_loader, criterion, loss_pl, loss_icp, config
        )
        train_acc_history.append(train_accuracy)
        val_acc_history.append(val_accuracy)
        print(
            f"Epoch {epoch + 1}/{config['epochs_phase1']} - Train Acc: {train_accuracy:.4f} - Val Acc: {val_accuracy:.4f} - Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, optimizer4isp, optimizer4icp, checkpoint_path)
        else:
            patience_counter += 1

        if patience_counter >= config["patience_phase1"]:
            print("Early stopping triggered in phase one.")
            break

    return train_acc_history, val_acc_history


def train_phase_two(
    model,
    train_tensor,
    val_tensor,
    optimizer,
    optimizer4isp,
    optimizer4icp,
    criterion,
    loss_pl,
    loss_icp,
    config,
):
    train_dataset_full = TensorDataset(
        torch.cat([train_tensor[0], val_tensor[0]]),
        torch.cat([train_tensor[1], val_tensor[1]]),
    )
    train_loader_full = DataLoader(
        train_dataset_full,
        batch_size=config["batch_size"],
        shuffle=True,
    )

    phase2_acc_history = []
    for epoch in range(config["epochs_phase2"]):
        model.train()
        correct_train = 0
        total_train = 0
        for X_batch, y_batch in train_loader_full:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            optimizer4isp.zero_grad()
            optimizer4icp.zero_grad()
            outputs = model(X_batch)
            cls_loss = criterion(outputs, y_batch)
            icp_loss = loss_icp(model.icp)
            feature = model.get_features()
            proxy = model.icp
            pl_loss = loss_pl(feature, proxy, y_batch)
            total_loss = (
                cls_loss
                + config["loss_weight_pl"] * pl_loss
                + config["loss_weight_icp"] * icp_loss
            )
            total_loss.backward()
            optimizer.step()
            optimizer4isp.step()
            optimizer4icp.step()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)
        train_accuracy = correct_train / max(total_train, 1)
        phase2_acc_history.append(train_accuracy)
        if (epoch + 1) % 25 == 0 or epoch == config["epochs_phase2"] - 1:
            print(
                f"Phase Two Epoch {epoch + 1}/{config['epochs_phase2']} - Train Acc: {train_accuracy:.4f}"
            )
    return phase2_acc_history


def evaluate(model, data_loader):
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(y_batch.numpy())
    acc = accuracy_score(targets, preds)
    kappa = cohen_kappa_score(targets, preds)
    return acc, kappa
```

---

## Cell 9 – Data Preparation and Training
Load the dataset, create data loaders, instantiate the model, and run the two training phases.
```python
# Load data from MOABB using the in-notebook utilities.
X, y, X_test, y_test = load_bcic(
    CONFIG["dataset_id"],
    CONFIG["subject_id"],
    CONFIG["preprocessing"],
)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=CONFIG["test_size"],
    random_state=CONFIG["random_state"],
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(
    train_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=CONFIG["batch_size"],
    shuffle=False,
)

sst_dpn = SST_DPN(
    chans=X_train_tensor.shape[1],
    samples=X_train_tensor.shape[2],
    num_classes=CONFIG["num_classes"],
    pool_kernels=CONFIG["mvp_pool_kernels"],
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    sst_dpn.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=CONFIG["weight_decay"],
)
optimizer4isp = torch.optim.Adam([{"params": sst_dpn.isp, "lr": CONFIG["lr_isp"]}])
optimizer4icp = torch.optim.Adam([{"params": sst_dpn.icp, "lr": CONFIG["lr_icp"]}])

loss_icp = NormIncreaseLoss()
loss_pl = PrototypeLoss()

train_acc_history, val_acc_history = train_phase_one(
    sst_dpn,
    train_loader,
    val_loader,
    optimizer,
    optimizer4isp,
    optimizer4icp,
    criterion,
    loss_pl,
    loss_icp,
    CONFIG,
    BEST_MODEL_PATH,
)

if BEST_MODEL_PATH.exists():
    load_checkpoint(
        sst_dpn,
        optimizer,
        optimizer4isp,
        optimizer4icp,
        BEST_MODEL_PATH,
        map_location=device,
    )

phase2_acc_history = train_phase_two(
    sst_dpn,
    (X_train_tensor.to(device), y_train_tensor.to(device)),
    (X_val_tensor.to(device), y_val_tensor.to(device)),
    optimizer,
    optimizer4isp,
    optimizer4icp,
    criterion,
    loss_pl,
    loss_icp,
    CONFIG,
)
```

---

## Cell 10 – Evaluation
Evaluate the trained model on the held-out test set and report accuracy and Cohen’s kappa.
```python
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor),
    batch_size=CONFIG["batch_size"],
    shuffle=False,
)

test_acc, test_kappa = evaluate(sst_dpn, test_loader)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Cohen's Kappa: {test_kappa:.4f}")
```

---

### Optional next steps
- Add visualisations (confusion matrices, learning curves) using the history lists returned in Cell 9.
- Port the baseline comparisons from `compare_model/` into additional cells if required.
- Save trained weights to Kaggle output by copying `BEST_MODEL_PATH` into `/kaggle/working` or `/kaggle/outputs`.
