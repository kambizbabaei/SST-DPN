# Kaggle Notebook Outline for SST-DPN

The following cells reproduce the behaviour of `train.py` in a Kaggle Notebook without requiring any edits to the original script. Each cell is self-contained and uses local variables so you can adjust configuration without touching other cells or modifying `train.py`. Update the `CONFIGURATION` cell to change the dataset, subject, or training hyper-parameters.

### How this notebook mirrors the repository flow

| Repository component | Purpose | Notebook cell(s) |
| --- | --- | --- |
| `load_data.py` | Downloads (when necessary) and prepares the EEG datasets. | **Cell 6 – Data Loading and Dataloaders** |
| `model.py` & `model_utils.py` | Define the SST-DPN architecture and its helper building blocks. | **Cell 4 – Imports and Utility Functions** and **Cell 7 – Model, Losses, and First Phase Training** |
| `dpl_utils.py` | Provides the prototype-learning regularisers used during optimisation. | **Cell 4** and **Cell 5 – Helper Functions** |
| `train.py` | Orchestrates data splitting, two-phase training, checkpointing, and evaluation. | **Cells 3–9**, which follow the same sequence of operations | 

The optional `compare_model/` baselines are not part of `train.py`; you can import and experiment with them after running the core training cells if needed.

---

## Cell 1 – Environment Setup
```python
!pip install --quiet braindecode==0.8.1 moabb==1.1.0 resampy==0.4.2 scipy==1.11.4
```

> **Tip:** Kaggle disables internet by default. Enable "Internet" in the notebook settings before running this cell.

---

## Cell 2 – Repository Layout
Create the working directory and copy the repository files into it. If you uploaded the repository as a Kaggle Dataset named `sst-dpn-repo`, this cell will mirror the layout under `/kaggle/working`.
```python
import os
import shutil
from pathlib import Path

REPO_SOURCE = Path("/kaggle/input/sst-dpn-repo")  # adjust if you used a different dataset name
REPO_ROOT = Path("/kaggle/working/SST-DPN")
MOABB_CACHE = Path("/kaggle/working/moabb_datasets")
MOABB_CACHE.mkdir(parents=True, exist_ok=True)

if REPO_ROOT.exists():
    shutil.rmtree(REPO_ROOT)
REPO_ROOT.mkdir(parents=True, exist_ok=True)

for item in REPO_SOURCE.iterdir():
    target = REPO_ROOT / item.name
    if item.is_dir():
        shutil.copytree(item, target)
    else:
        shutil.copy2(item, target)

print("Repository ready at", REPO_ROOT)
```

If you prefer to clone directly from GitHub (with internet enabled), replace the copying logic with:
```python
!git clone https://github.com/hancan16/SST-DPN.git /kaggle/working/SST-DPN
```

---

## Cell 3 – Configuration
Define all run parameters as local variables so they are easy to tweak for experimentation.
```python
from pathlib import Path

REPO_ROOT = Path("/kaggle/working/SST-DPN")
DATASET_ID = "2a"          # "2a" or "2b"
SUBJECT_ID = 5              # 1–9 for BNCI2014_001
TEST_SIZE = 0.2             # validation split ratio
RANDOM_STATE = 42
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
LR_ISP = 1e-3
LR_ICP = 1e-3
EPOCHS_PHASE1 = 1000
PATIENCE_PHASE1 = 200
EPOCHS_PHASE2 = 300
LOSS_WEIGHT_PL = 1e-3
LOSS_WEIGHT_ICP = 1e-5
MVP_POOL_KERNELS = [50, 100, 200]
NUM_CLASSES = 4
PREPROCESSING_CONFIG = {
    "sfreq": 250,
    "low_cut": None,
    "high_cut": None,
    "start": 0,
    "stop": 0,
    "z_scale": False,
}
```

---

## Cell 4 – Imports and Utility Functions
```python
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split

import sys
sys.path.append(str(REPO_ROOT))

from dpl_utils import NormIncreaseLoss, PrototypeLoss
from load_data import load_bcic
from model import SST_DPN
```

---

## Cell 5 – Helper Functions
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ.setdefault("MOABB_DATASETS", str(MOABB_CACHE))

def save_checkpoint(model, optimizer, optimizer4isp, optimizer4icp, filepath):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "optimizer4isp_state_dict": optimizer4isp.state_dict(),
            "optimizer4icp_state_dict": optimizer4icp.state_dict(),
        },
        filepath,
    )

def load_checkpoint(model, optimizer, optimizer4isp, optimizer4icp, filepath):
    checkpoint = torch.load(filepath, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    optimizer4isp.load_state_dict(checkpoint["optimizer4isp_state_dict"])
    optimizer4icp.load_state_dict(checkpoint["optimizer4icp_state_dict"])

def validate(model, val_loader, criterion, loss_icp, loss_pl, device):
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
            loss = criterion(outputs, y_batch)
            icp_loss = loss_icp(model.icp)
            feature = model.get_features()
            proxy = model.icp
            pl_loss = loss_pl(feature, proxy, y_batch)
            _ = loss + LOSS_WEIGHT_PL * pl_loss + LOSS_WEIGHT_ICP * icp_loss
            val_loss += loss.item()
    return val_loss / len(val_loader), correct_val / total_val

def train_phase(model, train_loader, val_loader, criterion, optimizer, optimizer4isp, optimizer4icp, loss_icp, loss_pl, device, epochs, patience):
    first_phase_train_acc = []
    first_phase_val_acc = []
    best_loss = float("inf")
    no_improve_epochs = 0
    for epoch in range(epochs):
        model.train()
        correct_train = 0
        total_train = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            optimizer4isp.zero_grad()
            optimizer4icp.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            icp_loss = loss_icp(model.icp)
            feature = model.get_features()
            proxy = model.icp
            pl_loss = loss_pl(feature, proxy, y_batch)
            total_loss = loss + LOSS_WEIGHT_PL * pl_loss + LOSS_WEIGHT_ICP * icp_loss
            total_loss.backward()
            optimizer.step()
            optimizer4isp.step()
            optimizer4icp.step()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)
        first_phase_train_acc.append(correct_train / total_train)
        val_loss, val_acc = validate(model, val_loader, criterion, loss_icp, loss_pl, device)
        first_phase_val_acc.append(val_acc)
        print(f"Epoch {epoch + 1}/{epochs}, Validation Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            no_improve_epochs = 0
            save_checkpoint(model, optimizer, optimizer4isp, optimizer4icp, "best_model.pth")
        else:
            no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping triggered")
            break
    return first_phase_train_acc, first_phase_val_acc

def train_second_phase(model, train_loader, criterion, optimizer, optimizer4isp, optimizer4icp, loss_icp, loss_pl, device, epochs):
    second_phase_train_acc = []
    for epoch in range(epochs):
        correct_train = 0
        total_train = 0
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            optimizer4isp.zero_grad()
            optimizer4icp.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            icp_loss = loss_icp(model.icp)
            feature = model.get_features()
            proxy = model.icp
            pl_loss = loss_pl(feature, proxy, y_batch)
            total_loss = loss + LOSS_WEIGHT_PL * pl_loss + LOSS_WEIGHT_ICP * icp_loss
            total_loss.backward()
            optimizer.step()
            optimizer4isp.step()
            optimizer4icp.step()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)
        second_phase_train_acc.append(correct_train / total_train)
        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {loss.item():.4f}")
    return second_phase_train_acc

def test_model(model, X_test, y_test, device):
    model.eval()
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
```

---

## Cell 6 – Data Loading and Dataloaders
```python
# `load_bcic` leverages MOABB/Braindecode and will automatically download the
# BNCI dataset into `MOABB_DATASETS` the first time it is called. With Kaggle
# Internet enabled, the files are cached under `/kaggle/working/moabb_datasets`
# and reused on subsequent runs.

X, y, X_test, y_test = load_bcic(DATASET_ID, SUBJECT_ID, PREPROCESSING_CONFIG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
```

---

## Cell 7 – Model, Losses, and First Phase Training
```python
model = SST_DPN(
    chans=X_train_tensor.shape[1],
    samples=X_train_tensor.shape[2],
    num_classes=NUM_CLASSES,
    F1=9,
    F2=48,
    time_kernel1=75,
    pool_kernels=MVP_POOL_KERNELS,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
optimizer4isp = torch.optim.Adam([{"params": model.isp, "lr": LR_ISP}])
optimizer4icp = torch.optim.Adam([{"params": model.icp, "lr": LR_ICP}])
loss_icp = NormIncreaseLoss()
loss_pl = PrototypeLoss()

first_phase_train_acc, first_phase_val_acc = train_phase(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    optimizer4isp,
    optimizer4icp,
    loss_icp,
    loss_pl,
    device,
    epochs=EPOCHS_PHASE1,
    patience=PATIENCE_PHASE1,
)
```

---

## Cell 8 – Second Phase Training and Evaluation
```python
load_checkpoint(model, optimizer, optimizer4isp, optimizer4icp, "best_model.pth")

train_dataset_full = TensorDataset(
    torch.cat([X_train_tensor, X_val_tensor]),
    torch.cat([y_train_tensor, y_val_tensor])
)
train_loader_full = DataLoader(train_dataset_full, batch_size=BATCH_SIZE, shuffle=True)

second_phase_train_acc = train_second_phase(
    model,
    train_loader_full,
    criterion,
    optimizer,
    optimizer4isp,
    optimizer4icp,
    loss_icp,
    loss_pl,
    device,
    epochs=EPOCHS_PHASE2,
)

test_model(model, X_test, y_test, device)
```

---

## Cell 9 – (Optional) Persist Metrics
```python
import json

metrics = {
    "first_phase_train_acc": first_phase_train_acc,
    "first_phase_val_acc": first_phase_val_acc,
    "second_phase_train_acc": second_phase_train_acc,
}

with open("/kaggle/working/training_metrics.json", "w") as fp:
    json.dump(metrics, fp)

print("Metrics saved to /kaggle/working/training_metrics.json")
```

This final cell saves the collected metrics so they can be downloaded from the Kaggle output pane.
