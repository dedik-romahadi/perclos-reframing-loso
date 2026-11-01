"""Deep-learning and hybrid model zoo for raw SEED-VIG EOG (125 Hz, 8-s epochs = 1000 samples).

    import deep_models as DM
    m = DM.build("resnet1d_gru", n_chans=6, n_times=1000, n_classes=3, K=8)
    logits = m(x, mask=mask)              # x (B, K, C, T) for sequence models, (B, C, T) otherwise
    logits, emb = m.forward_with_emb(x, mask=mask, feats=f)   # emb (B, m.emb_dim)
    DM.REGISTRY["resnet1d_gru"]["model_class"]                # 'dl_e2e'

Common interface (class DeepModel): forward(x=None, mask=None, feats=None) -> logits (no softmax);
embed(...) -> penultimate embedding (B, emb_dim); forward_with_emb(...) -> both in one pass.
Attributes: model_class, seq_len (K; 1 = single-epoch model taking (B, C, T)), uses_raw (False for
feature-only models, which get x=None), feat_set (feature_sets name of the injected handcrafted
features or None), n_feat, emb_dim, instance_norm. Sequence models take x (B, K, C, T) with
mask (B, K) (1 = real epoch, 0 = left padding; deep_data.seq_index) and feats (B, K, n_feat).
Every model accepts instance_norm=True: per-epoch, per-channel standardisation over the 1000
samples of each epoch (a within-epoch, label-free, deployable operation).

Model classes (vocabulary fixed for the programme):
  dl_e2e      'eegnet', 'shallow', 'deep4', 'conformer' (braindecode 1.5.2 classes adapted to
              125 Hz; see BRAINDECODE_125HZ), 'resnet1d' (small 1-D ResNet, ~0.2 M parameters),
              'resnet1d_gru' (resnet1d epoch encoder + causal GRU over the current and previous
              K-1 epochs, K = 8 = 64 s, end to end), 'biot_ft' (braindecode BIOT fine-tuned from
              the pretrained weights in RESEARCH/DEEP LEARNING MODEL/BIOSIGNAL/TF/BIOT; input
              resampled 125 -> 200 Hz by linear interpolation inside the model).
  hybrid      'feat_gru': handcrafted feature vectors (default litc-trad, deployable) of the current
              and previous K-1 epochs -> GRU -> head. Imputation and z-scoring with TRAINING-person
              statistics are done by deep_train inside the fold (median impute, z-score, clip +/-8).
  hybrid_e2e  'sinc_hybrid': learnable SincNet-style band-pass filterbank (Ravanelli & Bengio, IEEE SLT
              2018, arXiv:1808.00158) initialised at the Publication 2 eye-movement bands
              (fixation 0.5-1, blink 1-2, saccade 2-5, micro-saccade 5-20 Hz) plus a 0.1-0.5 Hz
              slow-eye-movement band; per band x channel six differentiable descriptors (log band
              energy, relative energy across bands, Hjorth mobility and complexity, log mean |first
              difference|, log RMS of the first difference = LRSSV analogue); concatenated with a
              small CNN-branch embedding and (inject_features=True, default) the litc-trad vector of
              the epoch; causal GRU over K epochs; head. 'sinc_hybrid_noseq' = the same with K = 1.
The CNN / Conformer architectures are trained FROM SCRATCH here: the other weights in DEEP LEARNING
MODEL/BIOSIGNAL are task-trained on a different 8-channel 250-Hz EEG dataset and are not used.
LaBraM is not included (see SKIPPED).

Sinc filterbank details: kernels are linear-phase windowed sinc band-passes (Hamming, L = 501 taps
= 4 s, unit pass-band gain), cut-offs f1 = 0.05 + |a|, f2 = f1 + 0.1 + |b| Hz (clamped below
61.5 Hz), initialised exactly at the stated bands. They are applied as CAUSAL FIR filters
(y[t] = sum_j h[j] x[t-j], F.conv1d on the left-padded stream) to the continuous stream formed by the K epochs of the window
(padded slots are zeroed = zero history), so a descriptor of epoch k uses samples up to the end of
epoch k only (group delay (L-1)/2 = 2 s). With K = 1 the epoch is filtered alone from zero history
(start-up transient in the first ~4 s). Frequency resolution of a 4-s Hamming kernel is ~1 Hz
(main lobe), so the 0.1-0.5 Hz band acts as a < ~0.6 Hz low band.
"""
from __future__ import annotations

import math
import pathlib
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
BIOT_DIR = HERE.parents[1] / "DEEP LEARNING MODEL" / "BIOSIGNAL" / "TF" / "BIOT"
BIOT_WEIGHTS = BIOT_DIR / "biot_model.pt"
warnings.filterwarnings("ignore", message="A window was not provided")   # BIOT's own STFT (as in the original)
FS = 125
PUB2_BANDS = ((0.1, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 20.0))
BAND_NAMES = ("sem", "fixation", "blink", "saccade", "microsaccade")
DESCRIPTORS = ("log_energy", "rel_energy", "hjorth_mobility", "hjorth_complexity", "log_mean_abs_diff",
               "log_rms_diff")

# braindecode defaults and the values used here at 125 Hz / 1000 samples (time constants kept in seconds).
# Sources: EEGNet = Lawhern et al. 2018 doi:10.1088/1741-2552/aace8c; ShallowFBCSPNet / Deep4Net = Schirrmeister et al.
# 2017 doi:10.1002/hbm.23730; EEGConformer = Song et al. 2023 doi:10.1109/TNSRE.2022.3230250.
BRAINDECODE_125HZ = {
    "eegnet": {"note": "EEGNet defaults were designed for 128 Hz, kept: kernel_length 64 (0.51 s), "
                       "pool1 4, pool2 8, F1 8, D 2, F2 16, drop 0.25",
               "kw": dict(kernel_length=64, pool1_kernel_size=4, pool2_kernel_size=8, F1=8, D=2, drop_prob=0.25)},
    "shallow": {"note": "250-Hz defaults filter_time_length 25 (100 ms), pool 75 / stride 15 (300 / 60 ms) -> "
                        "13 (104 ms), 38 / 8 (304 / 64 ms)",
                "kw": dict(filter_time_length=13, pool_time_length=38, pool_time_stride=8)},
    "deep4": {"note": "250-Hz defaults: all temporal filter lengths 10 (40 ms) -> 5 (40 ms); max-pool 3 / stride 3 "
                      "kept (cannot be halved to an integer; total down-sampling 81 samples = 0.65 s)",
              "kw": dict(filter_time_length=5, filter_length_2=5, filter_length_3=5, filter_length_4=5,
                         pool_time_length=3, pool_time_stride=3)},
    "conformer": {"note": "250-Hz defaults filter_time_length 25, pool 75 / stride 15 -> 13, 38 / 8 (119 tokens of "
                          "40 dims), 6 layers, 10 heads",
                  "kw": dict(filter_time_length=13, pool_time_length=38, pool_time_stride=8, num_layers=6,
                             num_heads=10)},
}

SKIPPED = {
    "labram": "Not included. LaBraM (Jiang, Zhao & Lu, ICLR 2024, arXiv:2405.18765; appendix checked 28 Oct 2026) was "
              "pretrained on ~2,500 h that include the SEED emotion series (SEED, SEED-IV, SEED-GER, SEED-FRA) and "
              "342 h of undisclosed 'self-collected' 62-channel NeuroScan EEG from the same SJTU lab that recorded "
              "SEED-VIG, so exclusion of SEED-VIG recordings / participants cannot be verified. Its spatial embedding "
              "is also indexed by EEG channel (10-20 positions), which the 7 EOG electrodes do not have.",
}


# =============================================================================== base
def instance_normalise(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-epoch, per-channel standardisation over the last (time) axis."""
    m = x.mean(-1, keepdim=True)
    v = x.var(-1, unbiased=False, keepdim=True)
    return (x - m) / torch.sqrt(v + eps)


class DeepModel(nn.Module):
    """Base class: see the module docstring for the interface."""

    model_class = "dl_e2e"

    def __init__(self, n_chans: int, n_times: int, n_classes: int, seq_len: int = 1,
                 instance_norm: bool = False, uses_raw: bool = True, feat_set: str | None = None,
                 n_feat: int = 0):
        super().__init__()
        self.n_chans, self.n_times, self.n_classes = n_chans, n_times, n_classes
        self.seq_len, self.instance_norm, self.uses_raw = int(seq_len), bool(instance_norm), uses_raw
        self.feat_set, self.n_feat = feat_set, int(n_feat)
        self.emb_dim: int | None = None

    def _prep(self, x):
        if x is None:
            return None
        x = x.float()
        return instance_normalise(x) if self.instance_norm else x

    def forward_with_emb(self, x=None, mask=None, feats=None):
        raise NotImplementedError

    def forward(self, x=None, mask=None, feats=None):
        return self.forward_with_emb(x, mask, feats)[0]

    def embed(self, x=None, mask=None, feats=None):
        return self.forward_with_emb(x, mask, feats)[1]


def _mask_or_ones(mask, B, K, ref):
    if mask is None:
        return torch.ones(B, K, 1, device=ref.device, dtype=ref.dtype)
    return mask.to(device=ref.device, dtype=ref.dtype)[..., None]


# =============================================================================== braindecode wrappers
class BraindecodeNet(DeepModel):
    """Single-epoch braindecode network. The embedding is the input of its final layer; when that
    is a feature map it is averaged over time (EEGNet 16, Shallow 40, Deep4 200 maps; Conformer 32)."""

    def __init__(self, net: nn.Module, n_chans, n_times, n_classes, instance_norm=False):
        super().__init__(n_chans, n_times, n_classes, 1, instance_norm)
        self.net = net
        self._cap = None
        self.net.final_layer.register_forward_hook(self._hook)
        was = self.net.training
        self.net.eval()
        with torch.no_grad():
            _, e = self.forward_with_emb(torch.zeros(2, n_chans, n_times))
        self.net.train(was)
        self.emb_dim = int(e.shape[1])

    def _hook(self, mod, inp, out):
        self._cap = inp[0]

    def forward_with_emb(self, x=None, mask=None, feats=None):
        logits = self.net(self._prep(x))
        h, self._cap = self._cap, None
        emb = h.flatten(2).mean(-1) if h.ndim > 2 else h
        return logits, emb


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _bd(cls_name, key):
    def make(n_chans, n_times, n_classes, instance_norm=False, **kw):
        import braindecode.models as BM
        p = dict(BRAINDECODE_125HZ[key]["kw"])
        p.update(kw)
        net = _quiet(getattr(BM, cls_name), n_chans=n_chans, n_outputs=n_classes, n_times=n_times, **p)
        return BraindecodeNet(net, n_chans, n_times, n_classes, instance_norm)
    return make


# =============================================================================== ResNet1D
class _Block(nn.Module):
    def __init__(self, cin, cout, stride, k=5):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, k, stride, k // 2, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, 1, k // 2, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.sc = (nn.Identity() if stride == 1 and cin == cout else
                   nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)))

    def forward(self, x):
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.sc(x))


class ResNet1DEncoder(nn.Module):
    """(N, C, 1000) -> (N, widths[-1]). Stem conv 7 / stride 2 + max-pool 2, four basic blocks
    (strides 1, 2, 2, 2; kernel 5), global average pooling. ~0.2 M parameters with 6 channels."""

    def __init__(self, n_chans, widths=(32, 48, 64, 128), k=5):
        super().__init__()
        w = widths
        self.stem = nn.Sequential(nn.Conv1d(n_chans, w[0], 7, 2, 3, bias=False), nn.BatchNorm1d(w[0]),
                                  nn.ReLU(), nn.MaxPool1d(2))
        self.blocks = nn.Sequential(_Block(w[0], w[0], 1, k), _Block(w[0], w[1], 2, k), _Block(w[1], w[2], 2, k),
                                    _Block(w[2], w[3], 2, k))
        self.out_dim = w[-1]

    def forward(self, x):
        return self.blocks(self.stem(x)).mean(-1)


class ResNet1D(DeepModel):
    def __init__(self, n_chans, n_times, n_classes, instance_norm=False, widths=(32, 48, 64, 128), drop=0.3):
        super().__init__(n_chans, n_times, n_classes, 1, instance_norm)
        self.enc = ResNet1DEncoder(n_chans, widths)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(self.enc.out_dim, n_classes)
        self.emb_dim = self.enc.out_dim

    def forward_with_emb(self, x=None, mask=None, feats=None):
        h = self.enc(self._prep(x))
        return self.head(self.drop(h)), h


class ResNet1DGRU(DeepModel):
    """resnet1d epoch encoder applied to each of the K epochs of the window, then a unidirectional
    (causal) GRU over the K embeddings (padded slots zeroed, mask appended as an input); the GRU
    state at the current epoch is the embedding. Trained end to end."""

    def __init__(self, n_chans, n_times, n_classes, K=8, instance_norm=False, widths=(32, 48, 64, 128),
                 hidden=128, drop=0.3):
        super().__init__(n_chans, n_times, n_classes, K, instance_norm)
        self.enc = ResNet1DEncoder(n_chans, widths)
        self.gru = nn.GRU(self.enc.out_dim + 1, hidden, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(hidden, n_classes)
        self.emb_dim = hidden

    def forward_with_emb(self, x=None, mask=None, feats=None):
        x = self._prep(x)
        B, K = x.shape[:2]
        z = self.enc(x.reshape(B * K, *x.shape[2:])).view(B, K, -1)
        m = _mask_or_ones(mask, B, K, z)
        out, _ = self.gru(torch.cat([z * m, m], -1))
        h = out[:, -1]
        return self.head(self.drop(h)), h


# =============================================================================== hybrid: features -> GRU
class FeatGRU(DeepModel):
    model_class = "hybrid"

    def __init__(self, n_chans, n_times, n_classes, n_feat, K=8, feat_set="litc-trad", hidden=64, drop=0.3,
                 instance_norm=False):
        super().__init__(n_chans, n_times, n_classes, K, False, uses_raw=False, feat_set=feat_set, n_feat=n_feat)
        self.inp = nn.Sequential(nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.gru = nn.GRU(hidden + 1, hidden, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(hidden, n_classes)
        self.emb_dim = hidden

    def forward_with_emb(self, x=None, mask=None, feats=None):
        f = feats.float()
        if f.ndim == 2:
            f = f[:, None]
        B, K = f.shape[:2]
        z = self.inp(f)
        m = _mask_or_ones(mask, B, K, z)
        out, _ = self.gru(torch.cat([z * m, m], -1))
        h = out[:, -1]
        return self.head(self.drop(h)), h


# =============================================================================== hybrid_e2e: sinc + descriptors
class SincFilterBank(nn.Module):
    """Learnable windowed-sinc band-pass filters applied as causal FIR filters (see module doc)."""

    def __init__(self, bands=PUB2_BANDS, fs=FS, kernel_len=501, min_low=0.05, min_band=0.1):
        super().__init__()
        kernel_len = int(kernel_len) | 1
        lo = torch.tensor([b[0] for b in bands], dtype=torch.float32)
        hi = torch.tensor([b[1] for b in bands], dtype=torch.float32)
        self.low_ = nn.Parameter(lo - min_low)
        self.band_ = nn.Parameter(hi - lo - min_band)
        self.fs, self.min_low, self.min_band, self.L = float(fs), float(min_low), float(min_band), kernel_len
        n = torch.arange(kernel_len, dtype=torch.float32) - (kernel_len - 1) / 2
        self.register_buffer("tt", n / fs, persistent=False)
        self.register_buffer("win", torch.hamming_window(kernel_len, periodic=False), persistent=False)

    def cutoffs(self):
        f1 = self.min_low + self.low_.abs()
        f2 = torch.clamp(f1 + self.min_band + self.band_.abs(), max=self.fs / 2 - 1.0)
        return f1, f2

    def kernels(self):
        f1, f2 = self.cutoffs()
        t = self.tt[None]
        h = 2 * f2[:, None] * torch.sinc(2 * f2[:, None] * t) - 2 * f1[:, None] * torch.sinc(2 * f1[:, None] * t)
        return h / self.fs * self.win[None]

    def forward(self, s, method: str = "conv"):
        """s (N, T) float32 continuous streams -> (N, n_bands, T): y[t] = sum_j h[j] s[t-j] (zero
        history before t = 0). method 'conv' (default; F.conv1d on the left-padded stream with the
        flipped kernel) or 'fft' (same result, more memory)."""
        T = s.shape[-1]
        h = self.kernels()
        if method == "fft":
            nfft = 1 << int(math.ceil(math.log2(T + self.L - 1)))
            Sx = torch.fft.rfft(s, n=nfft)
            return torch.fft.irfft(Sx[:, None, :] * torch.fft.rfft(h, n=nfft)[None], n=nfft)[..., :T]
        return F.conv1d(F.pad(s[:, None, :], (self.L - 1, 0)), h.flip(-1)[:, None, :])


def band_descriptors(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """y (..., n_bands, T) -> (..., n_bands, 6): log energy, relative energy across bands, Hjorth
    mobility, Hjorth complexity, log mean |first difference|, log RMS of the first difference."""
    d1 = y[..., 1:] - y[..., :-1]
    d2 = d1[..., 1:] - d1[..., :-1]
    v0 = y.var(-1, unbiased=False)
    v1 = d1.var(-1, unbiased=False)
    v2 = d2.var(-1, unbiased=False)
    e = (y ** 2).mean(-1)
    mob = torch.sqrt(v1 / (v0 + eps) + eps)
    comp = torch.sqrt(v2 / (v1 + eps) + eps) / mob
    return torch.stack([torch.log(e + eps), e / (e.sum(-2, keepdim=True) + eps), mob, comp,
                        torch.log(d1.abs().mean(-1) + eps), 0.5 * torch.log((d1 ** 2).mean(-1) + eps)], -1)


class SincHybrid(DeepModel):
    model_class = "hybrid_e2e"

    def __init__(self, n_chans, n_times, n_classes, K=8, inject_features=True, n_feat=0, feat_set="litc-trad",
                 instance_norm=False, bands=PUB2_BANDS, sinc_len=501, cnn_dim=32, hidden=128, drop=0.3):
        if inject_features and n_feat <= 0:
            raise ValueError("sinc_hybrid with inject_features=True needs n_feat (deep_train passes it)")
        super().__init__(n_chans, n_times, n_classes, K, instance_norm,
                         feat_set=feat_set if inject_features else None, n_feat=n_feat if inject_features else 0)
        self.inject = bool(inject_features)
        self.sinc = SincFilterBank(bands, FS, sinc_len)
        self.n_bands = len(bands)
        self.desc_dim = n_chans * self.n_bands * len(DESCRIPTORS)
        self.desc_bn = nn.BatchNorm1d(self.desc_dim)
        self.cnn = nn.Sequential(
            nn.Conv1d(n_chans, 16, 7, 2, 3, bias=False), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, cnn_dim, 7, 2, 3, bias=False), nn.BatchNorm1d(cnn_dim), nn.ReLU(),
            nn.Conv1d(cnn_dim, cnn_dim, 7, 2, 3, bias=False), nn.BatchNorm1d(cnn_dim), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.proj = nn.Sequential(nn.Linear(self.desc_dim + cnn_dim + self.n_feat, hidden), nn.ReLU(), nn.Dropout(drop))
        self.gru = nn.GRU(hidden + 1, hidden, batch_first=True) if self.seq_len > 1 else None
        self.drop = nn.Dropout(drop)
        self.head = nn.Linear(hidden, n_classes)
        self.emb_dim = hidden

    def descriptors(self, x, mask=None):
        """x (B, K, C, T) -> (B, K, C*n_bands*6) causal band descriptors (float32)."""
        B, K, C, T = x.shape
        m = _mask_or_ones(mask, B, K, x)[..., None]                      # (B, K, 1, 1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            s = (x.float() * m.float()).permute(0, 2, 1, 3).reshape(B * C, K * T)
            y = self.sinc(s).reshape(B, C, self.n_bands, K, T).permute(0, 3, 1, 2, 4)
            return band_descriptors(y).reshape(B, K, -1)

    def forward_with_emb(self, x=None, mask=None, feats=None):
        x = self._prep(x)
        if x.ndim == 3:
            x = x[:, None]
        B, K, C, T = x.shape
        d = self.desc_bn(self.descriptors(x, mask).reshape(B * K, -1))
        parts = [d, self.cnn(x.reshape(B * K, C, T))]
        if self.inject:
            f = feats.float()
            parts.append((f if f.ndim == 3 else f[:, None]).reshape(B * K, -1))
        z = self.proj(torch.cat([p.float() for p in parts], -1)).view(B, K, -1)
        if self.gru is not None:
            m = _mask_or_ones(mask, B, K, z)
            out, _ = self.gru(torch.cat([z * m, m], -1))
            h = out[:, -1]
        else:
            h = z[:, -1]
        return self.head(self.drop(h)), h


# =============================================================================== foundation: BIOT
class BIOTFineTune(DeepModel):
    """braindecode BIOT (Yang et al., NeurIPS 2023, arXiv:2305.10351) built with the pretrained
    configuration (16 channel tokens, 200 Hz, n_fft 200, hop 100, 256 dims, 8 heads, 4 layers),
    loaded from BIOT_WEIGHTS (read-only; the classification head is re-initialised) and fine-tuned
    end to end. The 125-Hz epoch is resampled to 200 Hz (1000 -> 1600 samples) by linear
    interpolation; the C <= 16 montage channels use channel tokens 0..C-1. Embedding = the encoder
    output (mean over tokens, 256)."""

    def __init__(self, n_chans, n_times, n_classes, instance_norm=False, pretrained=True, sfreq=200,
                 weights=None):
        super().__init__(n_chans, n_times, n_classes, 1, instance_norm)
        from braindecode.models import BIOT
        if n_chans > 16:
            raise ValueError("BIOT pretrained weights have 16 channel tokens")
        self.n_times_rs = int(round(n_times * sfreq / FS))
        self.net = _quiet(BIOT, n_outputs=n_classes, n_chans=16, sfreq=sfreq, n_times=self.n_times_rs,
                          hop_length=100, embed_dim=256, num_heads=8, num_layers=4)
        self.loaded_keys, self.skipped_keys = 0, []
        if pretrained:
            sd = torch.load(weights or BIOT_WEIGHTS, map_location="cpu", weights_only=True)
            own = self.net.state_dict()
            keep = {}
            for k, v in sd.items():
                if k.startswith("final_layer") or k not in own or own[k].shape != v.shape:
                    self.skipped_keys.append(k)
                else:
                    keep[k] = v
            self.net.load_state_dict(keep, strict=False)
            self.loaded_keys = len(keep)
        self.emb_dim = 256

    def forward_with_emb(self, x=None, mask=None, feats=None):
        x = F.interpolate(self._prep(x), size=self.n_times_rs, mode="linear", align_corners=False)
        emb = self.net.encoder(x)
        return self.net.final_layer(emb), emb


# =============================================================================== registry
def _resnet1d(n_chans, n_times, n_classes, **kw):
    return ResNet1D(n_chans, n_times, n_classes, **kw)


def _resnet1d_gru(n_chans, n_times, n_classes, **kw):
    return ResNet1DGRU(n_chans, n_times, n_classes, **kw)


def _feat_gru(n_chans, n_times, n_classes, **kw):
    return FeatGRU(n_chans, n_times, n_classes, **kw)


def _sinc(n_chans, n_times, n_classes, **kw):
    return SincHybrid(n_chans, n_times, n_classes, **kw)


def _sinc_noseq(n_chans, n_times, n_classes, **kw):
    kw["K"] = 1
    return SincHybrid(n_chans, n_times, n_classes, **kw)


def _biot(n_chans, n_times, n_classes, **kw):
    return BIOTFineTune(n_chans, n_times, n_classes, **kw)


# seq: takes K epochs; features: default handcrafted feature set (None = raw only); raw: False for
# feature-only models (no raw input is loaded);
# train: per-model training defaults that override deep_train.TRAIN_DEFAULTS (micro_batch = gradient-
# accumulation chunk that keeps a process under the 1.5-GB GPU cap; the optimiser step is still 256).
REGISTRY = {
    "eegnet": {"model_class": "dl_e2e", "builder": _bd("EEGNet", "eegnet"), "seq": False, "features": None,
               "train": {}, "desc": "braindecode EEGNet (v4), 125-Hz settings in BRAINDECODE_125HZ"},
    "shallow": {"model_class": "dl_e2e", "builder": _bd("ShallowFBCSPNet", "shallow"), "seq": False,
                "features": None, "train": {"amp": False},
                "desc": "braindecode ShallowFBCSPNet (float32: square + log overflow in float16)"},
    "deep4": {"model_class": "dl_e2e", "builder": _bd("Deep4Net", "deep4"), "seq": False, "features": None,
              "train": {}, "desc": "braindecode Deep4Net"},
    "conformer": {"model_class": "dl_e2e", "builder": _bd("EEGConformer", "conformer"), "seq": False,
                  "features": None, "train": {"micro_batch": 128, "amp": False, "eval_batch": 128},
                  "desc": "braindecode EEGConformer (float32: with AMP its attention needs ~10 MB per sample)"},
    "resnet1d": {"model_class": "dl_e2e", "builder": _resnet1d, "seq": False, "features": None, "train": {},
                 "desc": "1-D ResNet, widths 32-48-64-128, ~0.2 M parameters"},
    "resnet1d_gru": {"model_class": "dl_e2e", "builder": _resnet1d_gru, "seq": True, "features": None,
                     "train": {"micro_batch": 64}, "desc": "resnet1d encoder + causal GRU over K epochs"},
    "feat_gru": {"model_class": "hybrid", "builder": _feat_gru, "seq": True, "features": "litc-trad", "train": {},
                 "raw": False,
                 "desc": "handcrafted litc-trad feature sequence -> GRU"},
    "sinc_hybrid": {"model_class": "hybrid_e2e", "builder": _sinc, "seq": True, "features": "litc-trad",
                    "train": {"micro_batch": 64, "eval_batch": 64},
                    "desc": "sinc filterbank + descriptors + CNN + features -> GRU"},
    "sinc_hybrid_noseq": {"model_class": "hybrid_e2e", "builder": _sinc_noseq, "seq": False,
                          "features": "litc-trad", "train": {}, "desc": "sinc_hybrid with K = 1"},
    "biot_ft": {"model_class": "dl_e2e", "builder": _biot, "seq": False, "features": None,
                "train": {"micro_batch": 64, "eval_batch": 128},
                "desc": "pretrained BIOT fine-tuned (foundation model)"},
}


def model_spec(name: str, **kw) -> dict:
    """What deep_train needs before building: model_class, seq (bool), feat_set (or None), train overrides."""
    r = REGISTRY[name]
    feat = kw.get("feat_set", r["features"]) if r["features"] else None
    if name.startswith("sinc_hybrid") and not kw.get("inject_features", True):
        feat = None
    return {"model_class": r["model_class"], "seq": r["seq"], "feat_set": feat, "train": dict(r["train"]),
            "uses_raw": r.get("raw", True)}


def build(name: str, n_chans: int, n_times: int, n_classes: int, **kw) -> DeepModel:
    """Build a registered model. kw: model options (instance_norm, K for sequence models,
    inject_features / n_feat / feat_set for feature models, architecture options). K is ignored by
    single-epoch models."""
    r = REGISTRY[name]
    kw = dict(kw)
    if not r["seq"]:
        kw.pop("K", None)
    if r["features"] is None:
        kw.pop("feat_set", None)
        kw.pop("n_feat", None)
    m = r["builder"](n_chans, n_times, n_classes, **kw)
    m.model_class = r["model_class"]
    m.registry_name = name
    return m


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
