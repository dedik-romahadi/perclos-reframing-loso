"""T1 network factories for deep_train_t1.run_loso (PREREG section 4.3 / 4.4).

MLPTrunk (M7 TC-MLPLN / M8 direct MLP, class 'hybrid'): per-epoch trunk on the litc-trad+kf feature
vector (209): Linear(209, 256) - GELU - Dropout 0.2 - Linear(256, 128) - GELU - Dropout 0.2 -
Linear(128, n_out). n_out = 2 for the logit-normal head (mu, softplus -> sigma) or the number of classes.
The trainer supplies median-imputed, z-scored, +/-8-clipped features (fitting persons only).
"""
from __future__ import annotations

import pathlib as _pl
import sys as _sys

_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))

import t1_common  # noqa: F401  (paths)
import torch.nn as nn

import deep_models as DM

MLP_FEATSET = "litc-trad+kf"
MLP_NFEAT = 209


class MLPTrunk(DM.DeepModel):
    model_class = "hybrid"

    def __init__(self, n_chans, n_times, n_classes, n_feat=MLP_NFEAT, feat_set=MLP_FEATSET, drop=0.2):
        super().__init__(n_chans, n_times, n_classes, 1, False, uses_raw=False, feat_set=feat_set, n_feat=n_feat)
        self.trunk = nn.Sequential(nn.Linear(n_feat, 256), nn.GELU(), nn.Dropout(drop),
                                   nn.Linear(256, 128), nn.GELU(), nn.Dropout(drop))
        self.head = nn.Linear(128, n_classes)
        self.emb_dim = 128

    def forward_with_emb(self, x=None, mask=None, feats=None):
        h = self.trunk(feats.float())
        return self.head(h), h


def mlp_trunk(n_chans, n_times, n_classes):
    return MLPTrunk(n_chans, n_times, n_classes)
