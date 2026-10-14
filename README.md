# FMIML-UTAMamba

**Unbalanced Transport Assignment and BiMamba Reasoning for Federated Multi-Instance Multi-Label Learning**

- **FMIML-UTAMamba:** federated training
- **MIML-UTAMamba:** centralized training
- **Core modules:** AA-UTA (Absence-Aware Unbalanced Transport Assignment) and BiMamba (Bidirectional Mamba)
- **Datasets:** VOC 2007, VOC 2012, Scene, MSRC-v2

## Directory Structure

```text
FMIML-UTAMamba/
├── README.md
├── main.py                       # Federated training entry point
├── train_centralized.py          # Centralized training entry point
├── model.py                      # AA-UTA and BiMamba
├── client.py                     # Local client training
├── server.py                     # Federated aggregation and evaluation
├── dataset.py                    # Dataset loaders
├── config.py                     # Paths and hyperparameters
├── utils.py                      # Metrics and utilities
├── dict.py                       # MSRC preprocessing utility
└── fed_train_bimamba_voc2007.py  # Legacy script (incomplete dependencies)
```

## Run

First, install compatible PyTorch/torchvision and dependencies:

```bash
pip install numpy scipy scikit-learn pillow tqdm pycocotools
pip install mamba-ssm
```

Set `DATA_ROOT` (and optionally `GLOVE_PATH`) in `config.py`. Datasets should be placed under `VOCdevkit/VOC2007`, `VOCdevkit/VOC2012`, `scene/` and `MSRCv2/msrc2_seg`, respectively. `mamba-ssm` is needed to run BiMamba rather than the code's Bi-LSTM fallback.

### Federated Training (FMIML-UTAMamba)

```bash
# IID
python main.py --dataset voc2007 --partition_mode iid --rounds 50 --gpu 0

# Dirichlet non-IID
python main.py --dataset voc2007 --partition_mode dirichlet --dirichlet_alpha 0.5 --rounds 50 --gpu 0

# Label-based partition (not identical to the paper's PLV setting)
python main.py --dataset voc2007 --partition_mode label --labels_per_client 6 --rounds 50 --gpu 0
```

### Centralized Training (MIML-UTAMamba)

```bash
python train_centralized.py --dataset voc2007 --epochs 50 --batch_size 16 --gpu 0
python train_centralized.py --dataset voc2012 --epochs 50 --batch_size 16 --gpu 0
python train_centralized.py --dataset scene   --epochs 50 --batch_size 16 --gpu 0
python train_centralized.py --dataset msrc    --epochs 50 --batch_size 16 --gpu 0
```

For other options, run `python main.py --help` or `python train_centralized.py --help`.
