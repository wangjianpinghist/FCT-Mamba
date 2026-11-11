# FCT-Mamba

**FCT-Mamba: Factorized Condition-Aware Temporal Mamba**

Compact PyTorch implementation of the core network for four-class EEG classification of typically developing controls (TDC), ADHD, Tourette syndrome (TS), and TS+ADHD during multi-condition cognitive control tasks.

## Source layout

```text
FCT-Mamba-Core/
├── fct_mamba/
│   ├── __init__.py       Public imports
│   ├── config.py         Model settings, frequency bands, scalp regions
│   ├── graph.py          Condition-aware multiband signed graphs and propagation
│   ├── mamba.py          Selective Mamba and bidirectional Mamba blocks
│   ├── pooling.py        Attention-based feature aggregation
│   ├── encoder.py        Parallel graph / within-trial temporal encoder and reliability head
│   ├── decoder.py        Factorized comorbidity decoder
│   ├── model.py          Cross-trial encoding and four-class prediction
│   └── utils.py          Session windows and subject-level aggregation
├── requirements.txt
└── README.md
```

## Main components

- **CA-MSG:** Adaptive fusion of theta (4–8 Hz), alpha (8–13 Hz), and beta (13–30 Hz) channel correlations according to task condition and EEG signal statistics, followed by separate positive and negative graph processing. A coarse regional prior is applied only to the positive graph.
- **HBM:** Bidirectional Mamba within each trial, in parallel with graph feature extraction, followed by bidirectional Mamba across locally ordered trials after feature fusion.
- **RG:** Estimated trial reliability conditions cross-trial representations and modulates trial aggregation. The reliability embedding and pooling weights use detached reliability values.
- **FCD:** TS/ADHD factor logits and prediction-level comorbidity interaction and bounded residual corrections.

## Main settings

| Parameter | Setting |
| --- | --- |
| EEG channels / samples / sampling rate | 32 / 160 / 128 Hz |
| Class order | TDC, ADHD, TS, TS+ADHD |
| Trial conditions | Go, NoGo, Conflict |
| Feature dimension | 80 |
| Signed graph layers | 4 |
| Graph neighbors per node | 8 |
| Mamba state dimension | 32 |
| Mamba expansion factor | 2 |
| Mamba convolution width | 4 |
| Within-trial bidirectional Mamba layers | 1 |
| Cross-trial bidirectional Mamba layers | 2 |
| Dropout | 0.20 |
| Local sequence length / stride | 12 / 6 |

The experimental protocol uses AdamW, initial learning rate `4e-4`, weight decay `1.5e-4`, batch size `16`, a maximum of `150` epochs, early-stopping patience `25`, gradient clipping norm `4.0`, `ReduceLROnPlateau`, mixed precision, and subject-independent five-fold cross-validation. This repository provides the core architecture, not the training workflow.

## Installation and usage

Python 3.10+ and PyTorch 2.2+ are required. From the `FCT-Mamba-Core` directory, install the requirements:

```bash
pip install -r requirements.txt
```

```python
import torch
from fct_mamba import FCTMamba, ModelConfig

model = FCTMamba(ModelConfig()).eval()
eeg = torch.randn(1, 12, 32, 160)
trial_type = torch.zeros(1, 12, dtype=torch.long)
session = torch.zeros(1, 12, dtype=torch.long)
block = torch.zeros(1, 12, dtype=torch.long)

with torch.no_grad():
    outputs = model(eeg, trial_type, session, block)

sequence_logits = outputs["logits"]
```

The model expects subject-wise channel-standardized EEG epochs. `trial_type` uses Go = 0, NoGo = 1, Conflict = 2. `session` uses 0–1 and `block` uses 0–5. The EEG channel order is:

`Fp1, Fp2, AF3, AF4, F7, F3, Fz, F4, F8, FC5, FC1, FC2, FC6, T7, C3, Cz, C4, T8, CP5, CP1, CP2, CP6, P7, P3, Pz, P4, P8, PO7, PO3, PO4, PO8, Oz`.

The forward pass returns sequence-level four-class `logits`, `ts_logit`, `adhd_logit`, `interaction_logit`, and trial-level `reliability_logit`.

## Trial organization and aggregation

Each subject has two sessions of 36 trials, for 72 trials total. Local windows of 12 trials with stride 6 are constructed separately within sessions, yielding 10 sequences per subject. Use `session_window_starts` to obtain the sequence start positions, and `aggregate_subject_probabilities` to average sequence logits and apply softmax:

```python
from fct_mamba import aggregate_subject_probabilities, session_window_starts

starts = session_window_starts()
subject_probabilities = aggregate_subject_probabilities(sequence_logits)
```


