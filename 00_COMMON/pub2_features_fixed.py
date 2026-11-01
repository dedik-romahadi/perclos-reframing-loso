"""Corrected Publication 2 EOG feature extractor, plus a 'faithful' (as-published) mode.

Same three feature types, same column names and order as pub2_features.feature_names()
(type 1: 168, type 2: 234, type 3: 15). pub2_features.py itself is NOT changed (a running job
uses it); this module imports its descriptor functions and names only.

mode="faithful"  reproduces the original MATLAB chain INCLUDING its bugs, for an as-published
                 comparison (audit ids in brackets):
  * eog_noise_removal.m: fir1(250, [0.5 50]/(200/2)) band-pass and fir1(250, [49 51]/(200/2),
    'stop') designed at the EEG rate 200 Hz but applied to 125-Hz EOG (effective 0.31-31.25 Hz
    pass, notch at 30.6-31.9 Hz) [signal-1]; MATLAB filtfilt (odd padding 750 samples);
    linear detrend of the whole ~2-h record; filloutliers(...,'linear','ThresholdFactor',4)
    per column on the whole record [signal-2, evaluation-13]. HEO/VEO are filtered and
    clipped as separate columns, as in eeg_eog_preprocessing.m.
  * type 1/2: per-epoch filtfilt of 1000 samples with 251-tap FIRs [signal-9], energy share
    across the 6 channels [signal-8], m-spacing DE in type 2 only [signal-10], code bands
    (alpha 8-13, beta 14-30, gamma 31-50 Hz) [signal-11]. (pub2_features.type1/type2.)
  * type 3: literal port of extract_eog_features.m type 3 + detectBlinks_custom.m,
    detectVerticalSaccades.m, detectHorizontalSaccades.m, combinedEyeMovementLabeling.m,
    applyDurationThresholds.m (250-Hz sample counts at 125 Hz) [signal-6],
    getLabeledEvents.m (diff == +/-k pairing bug) [signal-5], analyzeEyeMovements.m,
    window samples 125:1000 only [signal-7], absolute thresholds 3 / 0.3 / 0.35 [signal-3],
    onset else-break, post_idx reuse and the [10 12 24] catch fallback [signal-4].
    Where MATLAB would have thrown an uncaught error (no blink peak at all, or no horizontal
    saccade candidate; status 'matlab_error'), this port returns a row of 15 NaN and counts
    the case in the status returned by faithful_type3; such epochs did not occur in the
    published run (it completed) and occur in 0 of the 23 x 885 epochs here.

mode="fixed" corrects every confirmed signal-processing / feature error:
  * filters designed at the true 125 Hz; zero-phase 4th-order Butterworth (SOS) 0.5-50 Hz
    band-pass (-6 dB at 0.5 and 50 Hz after filtfilt) and a 50-Hz IIR notch (Q = 30)
    [signal-1]; NO amplitude clipping, HEO/VEO kept as exact electrode differences
    [signal-2]; per-epoch QC flags (flat line, extreme amplitude) are returned instead.
  * every band is filtered ONCE on the continuous recording (label-free) and then epoched,
    so the 0.5-1 Hz band is resolved and no epoch-edge artefacts arise [signal-9].
  * 'energy' = per-channel RELATIVE band power E_b(c) / E_0.5-50(c); ratio features are
    ratios of ABSOLUTE band powers within the same channel, i.e. the classic
    (theta+alpha)/beta fatigue index [signal-8].
  * 'de' = Gaussian differential entropy 0.5 ln(2 pi e var) of the band-limited epoch
    (Duan 2013, doi:10.1109/NER.2013.6695876) [signal-10]. Type 1 still has no DE column
    (names are kept; see audit).
  * type-2 bands are contiguous: delta 1-4, theta 4-8, alpha 8-13, beta 13-30, and the
    'gamma' column is a 30-45 Hz high-frequency (EMG) index below the notch [signal-1,
    signal-11]. Type-1 'eye-movement' bands keep their names but are frequency bands only
    (fixation 0.5-1, blink 1-2, saccade 2-5, micro_saccade 5-20 Hz).
  * type 3 uses the scale-free continuous-record detectors of eog_features_lit.py (blinks:
    velocity thresholds in trailing robust-SD units, 10 % amplitude bounds, 50-1000 ms;
    saccades: adaptive velocity threshold, 24-200 ms; fixations: blink- and saccade-free
    runs >= 200 ms = 25 samples at 125 Hz), all segmented on the CONTINUOUS record, so an
    event that crosses an epoch border is not split; each event belongs to the epoch holding
    the blink peak / saccade onset / fixation onset; run-length segmentation; no fabricated
    events [signal-3, -4, -5, -6, -7]. Units as in the original (uV, ms). Undefined
    statistics (no event in the epoch) are NaN and must be imputed from TRAINING persons only.
  * look-ahead: the zero-phase filters on the continuous record and onset-assigned
    fixations make these OFFLINE features (features of epoch e change when data after the
    end of epoch e are removed; see check_lookahead.py / results/feature_check/lookahead.md).
    They are not deployable without that latency.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import detrend, filtfilt, find_peaks, firwin, iirnotch

import eog_features_lit as L
import pub2_features as P

FS = 125
N_SAMP = 1000
CHANNELS = P.CHANNELS                       # E1, E2, E3, E4, HEO, VEO
EYE_BANDS = P.EYE_BANDS                     # names/order kept
EEG_BANDS_FIXED = [("delta", 1.0, 4.0), ("theta", 4.0, 8.0), ("alpha", 8.0, 13.0), ("beta", 13.0, 30.0),
                   ("gamma", 30.0, 45.0)]
feature_names = P.feature_names
MAD_C = 1.482602218505602                   # MATLAB isoutlier scaled-MAD constant


def _six(eog7: np.ndarray) -> np.ndarray:
    """(n, 7, T) -> continuous (6, n*T): E1..E4, HEO = E1-E2, VEO = E3-E4 (float64)."""
    x = np.asarray(eog7, np.float64).transpose(1, 0, 2).reshape(eog7.shape[1], -1)
    return np.vstack([x[:4], x[0] - x[1], x[2] - x[3]])


def _epochs(c: np.ndarray, n_ep: int) -> np.ndarray:
    return c.reshape(c.shape[0], n_ep, N_SAMP).transpose(1, 0, 2)


# =============================================================================== faithful mode
def fir1_200(lo: float, hi: float, stop: bool = False) -> np.ndarray:
    """MATLAB fir1(250, [lo hi]/(200/2)) (Hamming, scaled) = firwin(251, ..., fs=200)."""
    return firwin(251, [lo, hi], pass_zero=stop, fs=200.0)


def filloutliers_linear(x: np.ndarray, k: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB filloutliers(x, 'linear', 'ThresholdFactor', k) for one column: outliers are
    samples farther than k scaled MADs from the median; they are replaced by linear
    interpolation of the neighbouring non-outliers (linear extrapolation at the ends)."""
    med = np.median(x)
    mad = MAD_C * np.median(np.abs(x - med))
    out = (x < med - k * mad) | (x > med + k * mad)
    if not out.any():
        return x.copy(), out
    idx = np.arange(len(x))
    good = idx[~out]
    y = x.copy()
    y[out] = np.interp(idx[out], good, x[good])
    if len(good) >= 2:
        bad = idx[out]
        for ii, (i0, i1) in ((bad[bad < good[0]], (good[0], good[1])),
                             (bad[bad > good[-1]], (good[-2], good[-1]))):
            if ii.size:
                y[ii] = x[i0] + (x[i1] - x[i0]) / (i1 - i0) * (ii - i0)
    return y, out


def faithful_preprocess(eog7: np.ndarray) -> np.ndarray:
    """eog_noise_removal.m on [E1..E4, HEO, VEO] as in eeg_eog_preprocessing.m -> (n, 6, 1000)."""
    c = _six(eog7)
    c = filtfilt(fir1_200(0.5, 50.0), [1.0], c, axis=-1, padlen=750)
    c = filtfilt(fir1_200(49.0, 51.0, stop=True), [1.0], c, axis=-1, padlen=750)
    c = detrend(c, axis=-1, type="linear")
    c = np.vstack([filloutliers_linear(row, 4.0)[0] for row in c])
    return _epochs(c, eog7.shape[0])


def _ml_findpeaks(y: np.ndarray) -> np.ndarray:
    """MATLAB findpeaks(y, 'MinPeakHeight', 0): 0-based indices of local maxima with y > 0
    (endpoints excluded, flat peaks reported at their first sample)."""
    pk, pr = find_peaks(y, plateau_size=(None, None))
    left = pr["left_edges"]
    return left[y[left] > 0]


def _ml_detect_blinks(V, Vd, Hd, thr):
    """detectBlinks_custom.m (1-based arrays with a dummy at index 0)."""
    n, n_der = len(V) - 1, len(Vd) - 1
    locs = _ml_findpeaks(V[1:] - thr) + 1
    locs = locs[locs > 5]
    blinks = []
    pre_idx = post_idx = None
    failed = False
    for loc in locs:
        for adj in range(1, 101):
            index = loc - adj
            if index < 2:
                break
            p1, p2 = Vd[index], Vd[index - 1]
            if p1 == 0:
                pre_idx = index
            elif p2 == 0:
                pre_idx = index - 1
            elif p1 * p2 < 0:
                pre_idx = index if abs(p1) < abs(p2) else index - 1
            else:
                pre_idx = index - 1          # the added else-break [signal-4a]
            break
        for adj in range(1, 101):
            index = loc + adj
            if index + 1 >= n_der:
                break
            p1, p2 = Vd[index], Vd[index + 1]
            if p1 == 0:
                post_idx = index
                break
            if p2 == 0:
                post_idx = index + 1
                break
            if p1 * p2 < 0:
                post_idx = index if abs(p1) < abs(p2) else index + 1
                break
        if pre_idx is None or post_idx is None:      # undefined variable -> catch [signal-4c]
            failed = True
            break
        blinks.append((pre_idx, int(loc), post_idx))  # post_idx may be reused [signal-4b]
    lab = np.ones(n + 1, int)
    Vdt = Vd.copy()
    Hd = Hd.copy()
    if failed:
        return np.array([[10, 12, 24]]), lab, Vdt, Hd, "fallback"
    for b1, _, b3 in blinks:
        s, e = max(1, b1), min(n, b3)
        lab[s:e + 1] = 3
        Vdt[s:e + 1] = 0
        Hd[s:e + 1] = 0
    return np.array(blinks, int).reshape(-1, 3), lab, Vdt, Hd, ("ok" if blinks else "noblink")


def _ml_sacc_search(der, locs, adj_range):
    """Onset/offset search of detectVertical/HorizontalSaccades.m (NaN-initialised per peak)."""
    n_der = len(der) - 1
    out = []
    for loc in locs:
        pre = post = None
        for adj in adj_range:
            index = loc - adj
            if index < 2:
                break
            p1, p2 = der[index], der[index - 1]
            if p1 == 0:
                pre = index
                break
            if p2 == 0:
                pre = index - 1
                break
            if p1 * p2 < 0:
                pre = index if abs(p1) < abs(p2) else index - 1
                break
        for adj in adj_range:
            index = loc + adj
            if index + 1 >= n_der:
                break
            p1, p2 = der[index], der[index + 1]
            if p1 == 0:
                post = index
                break
            if p2 == 0:
                post = index + 1
                break
            if p1 * p2 < 0:
                post = index if abs(p1) < abs(p2) else index + 1
                break
        if pre is not None and post is not None:
            out.append((pre, post))
    return np.array(out, int).reshape(-1, 2)


def _ml_peaks_both(d, thr):
    locs = np.r_[_ml_findpeaks(d[1:] - thr), _ml_findpeaks(-d[1:] - thr)] + 1
    return np.sort(locs, kind="stable")


def _overlaps(s, periods):
    """removeOverlappingSaccades.m mask: s_start <= p_end and s_end >= p_start."""
    if len(s) == 0 or len(periods) == 0:
        return np.zeros(len(s), bool)
    return ((s[:, :1] <= periods[None, :, -1]) & (s[:, 1:2] >= periods[None, :, 0])).any(1)


def _ml_apply_threshold(lab, target, thr, repl):
    """applyThreshold.m (in place on runs of `target` shorter than thr)."""
    lab = lab.copy()
    for val, s, e in L.run_length_segments(lab):
        if val == target and (e - s + 1) < thr:
            lab[s:e + 1] = repl
    return lab


def ml_get_labeled_events(label: np.ndarray, k: int) -> np.ndarray:
    """Literal getLabeledEvents.m (returns 1-based [start, end] rows). Buggy for 0/1/2/3
    label vectors: starts are diff == +k and ends diff == -k paired by order [signal-5]."""
    lab = np.asarray(label)
    d = np.diff(np.r_[0, lab, 0])
    starts = np.flatnonzero(d == k) + 1
    ends = np.flatnonzero(d == -k)
    if len(starts) > len(ends):
        ends = np.r_[ends, len(lab)]
    elif len(ends) > len(starts):
        ends = ends[1:]
    m = min(len(starts), len(ends))
    return np.c_[starts[:m], ends[:m]].astype(int)


def _mstd(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return np.nan
    return float(x.std(ddof=1)) if x.size > 1 else 0.0


def _mvar(x):
    x = np.asarray(x, float)
    if x.size == 0:
        return np.nan
    return float(x.var(ddof=1)) if x.size > 1 else 0.0


def faithful_type3_epoch(eog_h: np.ndarray, eog_v: np.ndarray, fs: int = FS) -> tuple[np.ndarray, str]:
    """extract_eog_features.m type 3 for one epoch of cleaned HEO/VEO (1000 samples each)."""
    t0, t1 = round(1 * fs), round(8 * fs)                 # time_epoch = [1 8] [signal-7]
    h = np.r_[np.nan, eog_h[t0 - 1:t1]]                   # 1-based, 876 samples
    v = np.r_[np.nan, eog_v[t0 - 1:t1]]
    n = len(h) - 1
    Hd = np.r_[np.nan, np.diff(h[1:])]
    Vd = np.r_[np.nan, np.diff(v[1:])]
    blinks, _, Vdt, Hd, status = _ml_detect_blinks(v, Vd, Hd, 3.0)
    # vertical saccades (thr 0.3) [signal-3]
    sv = _ml_sacc_search(Vd, _ml_peaks_both(Vdt, 0.3), range(1, 51))
    if len(sv) and len(blinks):
        bad = np.zeros(len(sv), bool)
        for b1, _, b3 in blinks:
            bad |= ((sv[:, 0] >= b1) & (sv[:, 0] <= b3)) | ((sv[:, 1] >= b1) & (sv[:, 1] <= b3))
        sv = sv[~bad]
    lab_v = np.ones(n + 1, int)
    for s, e in sv:
        lab_v[max(1, s):min(n, e) + 1] = 2
    for b1, _, b3 in blinks:
        lab_v[max(1, b1):min(n, b3) + 1] = 3
    # horizontal saccades (thr 0.35, search 3:50)
    sh = _ml_sacc_search(Hd, _ml_peaks_both(Hd, 0.35), range(3, 51))
    if status == "noblink" or len(sh) == 0:
        status = "matlab_error"           # removeOverlappingSaccades would throw on a 0x0 input
    sh_f = sh[~_overlaps(sh, blinks)] if len(blinks) else sh
    if len(sv):
        sh_f = sh_f[~_overlaps(sh_f, sv)]
    lab_h = np.ones(n + 1, int)
    for s, e in sh_f:
        lab_h[max(1, s):min(n, e) + 1] = 2
    lab = np.maximum(lab_v, lab_h)[1:]
    lab = _ml_apply_threshold(lab, 2, 10, 1)                # 10 samples = 80 ms at 125 Hz [signal-6]
    lab = _ml_apply_threshold(lab, 1, 50, 0)                # 50 samples = 400 ms
    fix_c = ml_get_labeled_events(lab, 1)
    sac_c = ml_get_labeled_events(lab, 2)
    if len(blinks):
        b_dur = (blinks[:, 2] - blinks[:, 0]) / fs * 1000.0
        b_amp = v[blinks[:, 1]] - v[blinks[:, 0]]
    else:
        b_dur = b_amp = np.zeros(0)
    if len(fix_c):
        f_dur = (fix_c[:, 1] - fix_c[:, 0]) / fs * 1000.0
    else:
        f_dur = np.zeros(1)
    if len(sac_c):
        s_dur = np.maximum(sac_c[:, 1] - sac_c[:, 0], 0) / fs * 1000.0
        s_amp = np.zeros(len(sac_c))
        for i, (a, b) in enumerate(sac_c):
            hi_ = np.flatnonzero(sh[:, 0] == a) if len(sh) else np.zeros(0, int)
            vi_ = np.flatnonzero(sv[:, 0] == a) if len(sv) else np.zeros(0, int)
            if hi_.size:
                s_amp[i] = abs(h[sh[hi_[0], 1]] - h[sh[hi_[0], 0]])
            elif vi_.size:
                s_amp[i] = abs(v[sv[vi_[0], 1]] - v[sv[vi_[0], 0]])
            else:
                s_amp[i] = np.sqrt((v[b] - v[a]) ** 2 + (h[b] - h[a]) ** 2)
    else:
        s_dur = s_amp = np.zeros(1)
    feat = np.array([len(blinks), np.mean(b_amp) if len(b_amp) else np.nan, _mstd(b_amp), _mvar(b_amp),
                     np.mean(b_dur) if len(b_dur) else np.nan, _mstd(b_dur), np.sum(b_dur),
                     len(sac_c), np.mean(s_amp), _mstd(s_amp), _mvar(s_amp), np.mean(s_dur), _mstd(s_dur),
                     len(fix_c), np.mean(f_dur)])
    if status == "matlab_error":                       # MATLAB would have stopped here: no values
        feat[:] = np.nan
    return feat, status


def faithful_type3(x6: np.ndarray) -> tuple[np.ndarray, dict]:
    out, stat = [], {}
    for i in range(x6.shape[0]):
        f, s = faithful_type3_epoch(x6[i, 4], x6[i, 5])
        out.append(f)
        stat[s] = stat.get(s, 0) + 1
    return np.array(out), stat


# =============================================================================== fixed mode
def fixed_prefilter(c: np.ndarray) -> np.ndarray:
    """0.5-50 Hz zero-phase Butterworth (order 4, SOS) + 50-Hz notch (Q = 30), at 125 Hz."""
    y = L.zero_phase(c, 0.5, 50.0, FS, order=4)
    b, a = iirnotch(50.0, 30.0, fs=FS)
    return filtfilt(b, a, y, axis=-1)


def qc_flags(eog7: np.ndarray, x6c: np.ndarray, n_ep: int, flat_uv: float = 1.0,
             extreme_uv: float = 2000.0) -> np.ndarray:
    """Per-epoch QC (replaces amplitude clipping): column 0 = flat line (any traditional
    electrode with epoch SD < flat_uv after band-pass), column 1 = extreme amplitude
    (|HEO| or |VEO| after band-pass > extreme_uv). Flags are reported, not used to drop data."""
    ep = _epochs(x6c, n_ep)
    flat = (ep[:, :4].std(-1) < flat_uv).any(1)
    ext = (np.abs(ep[:, 4:6]).max(-1) > extreme_uv).any(1)
    return np.c_[flat, ext].astype(np.int8)


def gaussian_de(b: np.ndarray) -> np.ndarray:
    """Gaussian differential entropy 0.5 ln(2 pi e var) along the last axis (nats)."""
    return 0.5 * np.log(2 * np.pi * np.e * np.maximum(b.var(-1, ddof=1), 1e-30))


def _fixed_band_block(c: np.ndarray, broad_e: np.ndarray, bands, with_de: bool, n_ep: int):
    out = {}
    for name, lo, hi in bands:
        b = _epochs(L.zero_phase(c, lo, hi, FS, order=4), n_ep)          # continuous filtering
        e = np.sum(b ** 2, -1)
        out[(name, "abs")] = e
        out[(name, "energy")] = e / np.maximum(broad_e, 1e-30)            # relative power per channel
        if with_de:
            out[(name, "de")] = gaussian_de(b)
        out[(name, "hm")] = P.hjorth_mobility(b)
        out[(name, "n2d")] = P.norm_second_diff(b)
        out[(name, "d1")] = P.first_diff(b)
        out[(name, "re")] = P.renyi_entropy(b)
        out[(name, "lrssv")] = P.lrssv(b)
    return out


def fixed_type12(eog7: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_ep = eog7.shape[0]
    c = fixed_prefilter(_six(eog7))
    broad = np.sum(_epochs(c, n_ep) ** 2, -1)
    f1 = _fixed_band_block(c, broad, EYE_BANDS, False, n_ep)
    E = {n: f1[(n, "abs")] for n, _, _ in EYE_BANDS}
    r1 = [(E["saccade"] + E["micro_saccade"]) / E["fixation"], E["blink"] / E["fixation"],
          (E["saccade"] + E["micro_saccade"]) / E["blink"], E["micro_saccade"] / E["saccade"]]
    cols = [f1[(n, d)] for d in P.T1_DESC for n, _, _ in EYE_BANDS] + r1
    t1 = np.stack(cols, -1).reshape(n_ep, -1)
    f2 = _fixed_band_block(c, broad, EEG_BANDS_FIXED, True, n_ep)
    E = {n: f2[(n, "abs")] for n, _, _ in EEG_BANDS_FIXED}
    r2 = [(E["theta"] + E["alpha"]) / E["beta"], E["gamma"] / E["delta"], E["beta"] / E["alpha"],
          E["alpha"] / E["theta"]]
    cols = [f2[(n, d)] for d in P.T2_DESC for n, _, _ in EEG_BANDS_FIXED] + r2
    t2 = np.stack(cols, -1).reshape(n_ep, -1)
    return t1, t2, c


def fixed_type3(eog7: np.ndarray) -> tuple[np.ndarray, dict]:
    """15 type-3 features (names of the original) from scale-free continuous-record detectors."""
    n_ep = eog7.shape[0]
    sig = L.montage_signals(eog7, "trad")                 # H = ch1-ch2, V = ch4-ch3 (blinks up)
    H, V = sig["H"], sig["V"]
    N = len(H)
    bl = L.detect_blinks(V, H)
    bm = L.blink_mask(bl, N, pad=int(0.05 * FS))
    sh = L.detect_saccades(H, bm)
    sv = L.detect_saccades(V, bm)
    smask = np.zeros(N, bool)
    for s in (sh, sv):
        for a, b in zip(s["on"].astype(int), s["off"].astype(int)):
            smask[a:b + 1] = True
    st, en = L.runs(smask)                                # union of H and V saccades
    Hf, Vf = sh["Xf"], sv["Xf"]
    a0, a1 = np.maximum(st - 1, 0), np.minimum(en, N - 1)
    s_amp = np.hypot(Hf[a1] - Hf[a0], Vf[a1] - Vf[a0])
    s_dur = (en - st) / FS * 1000.0
    s_ep = st // N_SAMP
    b_ep = np.floor(bl["peak"] / N_SAMP).astype(int)
    b_amp, b_dur = bl["A"], bl["Dtot"] * 1000.0
    busy = L.blink_mask(bl, N) | smask
    # fixations = blink/saccade-free runs >= 200 ms on the CONTINUOUS record (not cut at epoch
    # borders), assigned to the epoch holding their onset (as saccades)
    f_st, f_en = L.runs(~busy)
    keep = (f_en - f_st) >= L._ns(0.2, FS)
    f_st, f_en = f_st[keep], f_en[keep]
    f_dur = (f_en - f_st) / FS * 1000.0
    f_ep = f_st // N_SAMP
    X = np.full((n_ep, 15), np.nan)
    for e in range(n_ep):
        bi = b_ep == e
        si = s_ep == e
        fdur = f_dur[f_ep == e]
        ba, bd, sa, sd = b_amp[bi], b_dur[bi], s_amp[si], s_dur[si]
        X[e] = [bi.sum(), ba.mean() if ba.size else np.nan, _mstd(ba), _mvar(ba),
                bd.mean() if bd.size else np.nan, _mstd(bd), bd.sum(),
                si.sum(), sa.mean() if sa.size else np.nan, _mstd(sa), _mvar(sa),
                sd.mean() if sd.size else np.nan, _mstd(sd),
                fdur.size, fdur.mean() if fdur.size else np.nan]
    info = {"blinks_per_min": len(b_amp) / (N / FS / 60), "blink_Dtot_median_ms": float(np.median(b_dur)) if len(b_dur) else np.nan,
            "saccades": int(len(st)), "fixations": int(len(f_st)),
            "fix_dur_median_ms": float(np.median(f_dur)) if len(f_dur) else np.nan,
            "nan_frac": float(np.isnan(X).mean())}
    return X, info


# =============================================================================== driver
def extract_all(eog7: np.ndarray, mode: str = "fixed") -> dict:
    """Return {'type1', 'type2', 'type3', ...} for one recording (n_ep, 7, 1000) raw uV."""
    if mode == "faithful":
        x6 = faithful_preprocess(eog7)
        t3, stat = faithful_type3(x6)
        return {"type1": P.type1(x6), "type2": P.type2(x6), "type3": t3,
                "type3_status": np.array([f"{k}={v}" for k, v in sorted(stat.items())])}
    if mode == "fixed":
        t1, t2, c = fixed_type12(eog7)
        t3, info = fixed_type3(eog7)
        return {"type1": t1, "type2": t2, "type3": t3, "qc": qc_flags(eog7, c, eog7.shape[0]),
                "type3_info": np.array([f"{k}={v}" for k, v in info.items()])}
    raise ValueError(f"unknown mode {mode!r}")
