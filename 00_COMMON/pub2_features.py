"""Python port of the Publication 2 EOG feature extractor (shared baseline).

Source: PIVOTAL DATA/MATLAB TOOLBOX/COSTUM_FUNCTION/FEATURES_EXTRACTION/extract_eog_features.m
and the jfeeg toolbox descriptors (jHjorthMobility, jNormalizedSecondDifference,
jFirstDifference, jRenyiEntropy, jLogRootSumOfSequentialVariation) and
differential_entropy.m (1-D default = m-spacings estimator, nats).

Channels (as in Publication 2): electrodes 1-4 + HEO (ch1-ch2) + VEO (ch3-ch4) -> 6.
  type 1: 4 eye-movement bands x {rel. energy, HM, N2D, 1D, Renyi, LRSSV} + 4 ratios = 28/channel -> 168
  type 2: 5 EEG-style bands x {rel. energy, DE, HM, N2D, 1D, Renyi, LRSSV} + 4 ratios = 39/channel -> 234
  type 3: 15 eye-event features from HEO/VEO (re-implemented, see `event_features`)

Declared deviations from the MATLAB pipeline (documented in 00_COMMON/README.md):
  D1 Pre-filter: the original eog_noise_removal designed its 0.5-50 Hz band-pass and 49-51 Hz notch
     with the EEG rate (200 Hz) while the EOG is 125 Hz (effective 0.31-31 Hz pass, 31 Hz notch).
     Here the band-pass is designed at 125 Hz (0.5-50 Hz) plus a 49-51 Hz notch, applied per epoch.
     Whole-recording detrend/filloutliers are not reproduced (they use the whole 2-h record).
  D2 Type 3: the original detector thresholds raw microvolt peaks at 3 uV (blinks) and 0.3/0.35
     (saccade derivative), which does not transfer across subjects. It is replaced by a
     scale-free detector (robust epoch-local thresholds). Feature names/order are kept.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import filtfilt, find_peaks, firwin, peak_widths
from scipy.special import digamma

FS = 125
FILTORD = 250
EYE_BANDS = [("blink", 1.0, 2.0), ("saccade", 2.0, 5.0), ("micro_saccade", 5.0, 20.0), ("fixation", 0.5, 1.0)]  # MATLAB column order
EEG_BANDS = [("delta", 1.0, 3.0), ("theta", 4.0, 7.0), ("alpha", 8.0, 13.0), ("beta", 14.0, 30.0), ("gamma", 31.0, 50.0)]
CHANNELS = ["E1", "E2", "E3", "E4", "HEO", "VEO"]
T1_DESC = ["energy", "hm", "n2d", "d1", "re", "lrssv"]
T1_RATIOS = ["rA_(sac+msac)/fix", "rB_blink/fix", "rC_(sac+msac)/blink", "rD_msac/sac"]
T2_DESC = ["energy", "de", "hm", "n2d", "d1", "re", "lrssv"]
T2_RATIOS = ["r_(th+al)/be", "r_ga/de", "r_be/al", "r_al/th"]
T3_NAMES = ["blink_number", "blink_amp_mean", "blink_amp_std", "blink_amp_var", "blink_dur_mean",
            "blink_dur_std", "blink_dur_total", "sacc_number", "sacc_amp_mean", "sacc_amp_std",
            "sacc_amp_var", "sacc_dur_mean", "sacc_dur_std", "fix_number", "fix_dur_mean"]
_PADLEN = 3 * FILTORD  # MATLAB filtfilt pads 3*(nfilt-1)


def _fir(lo: float, hi: float, stop: bool = False) -> np.ndarray:
    return firwin(FILTORD + 1, [lo, hi], pass_zero=stop, fs=FS)


def _bp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return filtfilt(_fir(lo, hi), [1.0], x, axis=-1, padlen=_PADLEN)


def preprocess(eog7: np.ndarray) -> np.ndarray:
    """(n, 7, T) raw electrodes -> (n, 6, T): E1..E4, HEO, VEO, band-passed 0.5-50 Hz + notch."""
    x = np.concatenate([eog7[:, :4], eog7[:, 0:1] - eog7[:, 1:2], eog7[:, 2:3] - eog7[:, 3:4]], axis=1)
    x = x.astype(np.float64) - x.mean(-1, keepdims=True)
    x = filtfilt(_fir(0.5, 50.0), [1.0], x, axis=-1, padlen=_PADLEN)
    x = filtfilt(_fir(49.0, 51.0, stop=True), [1.0], x, axis=-1, padlen=_PADLEN)
    return x


# ---------------------------------------------------------------- jfeeg descriptors (vectorised over (..., T))
def hjorth_mobility(x):
    x1 = np.diff(x, axis=-1, prepend=0.0)
    return x1.std(-1, ddof=1) / x.std(-1, ddof=1)


def norm_second_diff(x):
    return np.abs(x[..., 2:] - x[..., :-2]).mean(-1) / x.std(-1, ddof=1)


def first_diff(x):
    return np.abs(np.diff(x, axis=-1)).mean(-1)


def renyi_entropy(x, alpha: float = 2.0):
    p = x ** 2 / np.sum(x ** 2, -1, keepdims=True)
    return (1.0 / (1.0 - alpha)) * np.log2(np.sum(p ** alpha, -1))


def lrssv(x):
    return np.log10(np.sqrt(np.sum(np.diff(x, axis=-1) ** 2, -1)))


def diff_entropy_spacings(x):
    """differential_entropy.m 1-D default (m-spacings, nats); m = max(floor(0.2 n^0.3), 1)."""
    n = x.shape[-1]
    m = max(int(np.floor(0.2 * n ** 0.3)), 1)
    xs = np.sort(x, axis=-1)
    sp = np.maximum(xs[..., m:] - xs[..., :-m], np.finfo(float).tiny)
    return (np.sum(np.log(sp), -1) + np.log(n / m) * (n - m)) / n + np.log(m) - digamma(m)


def _band_block(x, bands, desc):
    """x (n, C, T) -> list of per-band feature dicts, energies normalised across channels."""
    out = {}
    for name, lo, hi in bands:
        b = _bp(x, lo, hi)
        e = np.sum(b ** 2, -1)
        out[(name, "energy")] = e / e.sum(-1, keepdims=True)  # energy share of each channel (as in MATLAB)
        if "de" in desc:
            out[(name, "de")] = diff_entropy_spacings(b)
        out[(name, "hm")] = hjorth_mobility(b)
        out[(name, "n2d")] = norm_second_diff(b)
        out[(name, "d1")] = first_diff(b)
        out[(name, "re")] = renyi_entropy(b)
        out[(name, "lrssv")] = lrssv(b)
    return out


def type1(x6: np.ndarray) -> np.ndarray:
    f = _band_block(x6, EYE_BANDS, T1_DESC)
    E = {n: f[(n, "energy")] for n, _, _ in EYE_BANDS}
    ratios = [(E["saccade"] + E["micro_saccade"]) / E["fixation"], E["blink"] / E["fixation"],
              (E["saccade"] + E["micro_saccade"]) / E["blink"], E["micro_saccade"] / E["saccade"]]
    cols = [f[(n, d)] for d in T1_DESC for n, _, _ in EYE_BANDS] + ratios     # each (n, C)
    return np.stack(cols, -1).reshape(x6.shape[0], -1)                       # channel-major, 28 per channel


def type2(x6: np.ndarray) -> np.ndarray:
    f = _band_block(x6, EEG_BANDS, T2_DESC)
    E = {n: f[(n, "energy")] for n, _, _ in EEG_BANDS}
    ratios = [(E["theta"] + E["alpha"]) / E["beta"], E["gamma"] / E["delta"],
              E["beta"] / E["alpha"], E["alpha"] / E["theta"]]
    cols = [f[(n, d)] for d in T2_DESC for n, _, _ in EEG_BANDS] + ratios
    return np.stack(cols, -1).reshape(x6.shape[0], -1)                       # 39 per channel


def feature_names(kind: str) -> list[str]:
    if kind == "type1":
        per = [f"{n}_{d}" for d in T1_DESC for n, _, _ in EYE_BANDS] + T1_RATIOS
    elif kind == "type2":
        per = [f"{n}_{d}" for d in T2_DESC for n, _, _ in EEG_BANDS] + T2_RATIOS
    else:
        return list(T3_NAMES)
    return [f"{c}:{p}" for c in CHANNELS for p in per]


# ---------------------------------------------------------------- type 3 (re-implemented event detector)
def _mad_sigma(v):
    return 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-9


def _segments(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)  # start, stop (exclusive)


def event_features_epoch(heo: np.ndarray, veo: np.ndarray) -> np.ndarray:
    """15 eye-event features from one epoch of HEO/VEO (already 0.5-50 Hz filtered).

    Blinks: VEO peaks (10 Hz low-pass) with prominence >= max(20 uV, 4 robust SD of the epoch),
    width at half prominence 40-800 ms, >= 150 ms apart. Amplitude = prominence (uV),
    duration = width at half prominence (s).
    Saccades: |velocity| of 10 Hz low-passed HEO or VEO above 4 robust SD of that velocity,
    segment duration 16-200 ms, not overlapping a blink. Amplitude = |position change| (uV).
    Fixations: blink/saccade-free intervals >= 200 ms.
    """
    lp = firwin(FILTORD + 1, 10.0, fs=FS)
    h = filtfilt(lp, [1.0], heo, padlen=_PADLEN)
    v = filtfilt(lp, [1.0], veo, padlen=_PADLEN)
    v = v - np.median(v)
    T = len(v)
    prom_min = max(20.0, 4 * _mad_sigma(v))
    pk, pr = find_peaks(v, prominence=prom_min, distance=int(0.15 * FS))
    blink_mask = np.zeros(T, bool)
    b_amp, b_dur = [], []
    if len(pk):
        w, _, l, r = peak_widths(v, pk, rel_height=0.5)
        keep = (w >= 0.04 * FS) & (w <= 0.8 * FS)
        for p, a, ww, ll, rr in zip(pk[keep], pr["prominences"][keep], w[keep], l[keep], r[keep]):
            b_amp.append(a)
            b_dur.append(ww / FS)
            blink_mask[max(0, int(ll) - 5):min(T, int(np.ceil(rr)) + 5)] = True
    s_amp, s_dur = [], []
    sacc_mask = np.zeros(T, bool)
    for sig in (h, v):
        vel = np.gradient(sig) * FS
        thr = 4 * _mad_sigma(vel)
        st, en = _segments(np.abs(vel) > thr)
        for a, b in zip(st, en):
            dur = (b - a) / FS
            if not (0.016 <= dur <= 0.2) or blink_mask[a:b].any():
                continue
            s_amp.append(abs(sig[min(b, T - 1)] - sig[a]))
            s_dur.append(dur)
            sacc_mask[a:b] = True
    st, en = _segments(~(blink_mask | sacc_mask))
    fix = [(b - a) / FS for a, b in zip(st, en) if (b - a) >= 0.2 * FS]

    def stats(x):
        x = np.asarray(x, float)
        if len(x) == 0:
            return 0.0, 0.0, 0.0
        return float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0, float(x.var(ddof=1)) if len(x) > 1 else 0.0

    bam, bas, bav = stats(b_amp)
    bdm, bds, _ = stats(b_dur)
    sam, sas, sav = stats(s_amp)
    sdm, sds, _ = stats(s_dur)
    fdm, _, _ = stats(fix)
    return np.array([len(b_amp), bam, bas, bav, bdm, bds, float(np.sum(b_dur)),
                     len(s_amp), sam, sas, sav, sdm, sds, len(fix), fdm])


def type3(x6: np.ndarray) -> np.ndarray:
    return np.stack([event_features_epoch(x6[i, 4], x6[i, 5]) for i in range(x6.shape[0])])


def extract_all(eog7: np.ndarray) -> dict:
    x6 = preprocess(eog7)
    return {"type1": type1(x6), "type2": type2(x6), "type3": type3(x6)}
