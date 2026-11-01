"""Literature-best EOG feature set for SEED-VIG (traditional or forehead montage).

Every block below cites the paper(s) it implements. All detectors run on the CONTINUOUS
recording (885 contiguous 8-s epochs = 885,000 samples at 125 Hz), so events are never cut
at epoch borders and no per-epoch filter edge effects arise. Events are assigned to the
epoch that holds their peak (blinks) or onset (saccades, fixations).

Causality / leakage rules used here (see 00_COMMON/README.md, rule 3):
  * Detection thresholds use TRAILING robust scales (median of per-epoch 1.4826*MAD over the
    current and previous 36 epochs, i.e. ~5 min), never statistics of the whole recording and
    never labels. Event features are aggregated over a TRAILING 64-s window (current epoch +
    7 previous) inside the recording. Nothing crosses a recording boundary. Where a trailing
    estimate does not exist yet (leading epochs) the detector is DISABLED for those epochs
    (no whole-recording median fallback); the number of such epochs is reported in `info`.
  * Two variants (extract_recording(..., causal=...)):
      causal=False  "offline zero-phase" (caches lit_<montage>_*): sosfiltfilt on the
                    continuous record. Label-free, but NOT causal: the 0.05-Hz high-pass and the
                    0.1-0.6 / 0.1-1 Hz SEM filters have impulse responses of +/-13-24 s, and
                    fixations are assigned to their onset epoch although they end later.
                    Measured look-ahead (check_lookahead.py, 3 recordings x K = 150/450,
                    results/feature_check/lookahead.md): changes > 5 % of the column scale reach
                    2 epochs back, > 1e-3 reach 7 epochs (56 s), > 1e-6 reach 12 epochs; the
                    features of epochs <= K-1-LOOKAHEAD_EPOCHS['offline'] (13) are unchanged to
                    1e-6. So these are offline features with ~1-2 min of look-ahead; deployable
                    only with that latency.
      causal=True   "causal" (caches litc_<montage>_*): forward-only Butterworth (sosfilt with
                    steady-state initial conditions), all detection thresholds of epoch e from
                    epochs < e only (lag 1; detectors are off in the first epoch), fixations and
                    long closures assigned to the epoch in which they END, the longest closure
                    clipped at the epoch end. Remaining look-ahead is sub-epoch (centred
                    Savitzky-Golay derivative +/-3 samples,
                    Ricker convolution +/-40 samples, blink offsets up to ~2.5 s after the
                    peak): features of epoch e use samples up to the end of epoch e+1
                    (LOOKAHEAD_EPOCHS['causal'] = 1, i.e. <= 8 s after the epoch ends). This is
                    asserted EXACTLY by tests/test_features.py and check_lookahead.py.
  * Blink polarity is FIXED by montage (no data-dependent sign): in SEED-VIG ch3-ch4 has
    negative blinks in 23/23 recordings (skewness of the 0.5-10 Hz band < 0) and ch4-ch7 has
    positive blinks in 23/23, so VEO is used as ch4-ch3 and VEOf as ch4-ch7.
  * Durations/windows are given in seconds and converted with round(t*fs); the SWT/SEM
    window and the 1000-sample epoch are fixed sample counts, so extract_recording and
    sem_features require fs == 125 Hz (checked).

Montages (00_COMMON/seedvig_eog.py data facts):
  trad : H = ch1 - ch2 (HEO), V = ch4 - ch3 (= -VEO, blinks positive)
  fore : H = ch5 - ch6 (HEOf), V = ch4 - ch7 (VEOf, "minus" rule of Zheng & Lu 2017)

Blocks (column prefixes) and their sources:
  F01 blink duration D50 / D_total ...... Caffier 2003 doi:10.1007/s00421-003-0807-5;
                                          Schleicher 2008 doi:10.1080/00140130701817062;
                                          Cori 2019 doi:10.1016/j.smrv.2019.03.004;
                                          Pettersson 2013 doi:10.1186/1475-925X-12-110
  F02 closing / reopening / closed time . Caffier 2003; Schleicher 2008;
                                          Hu & Zheng 2009 doi:10.1016/j.eswa.2008.09.030
  F03 AVR, peak lid velocities .......... Johns 2007 doi:10.1007/s11818-007-0311-y;
                                          Wilkinson 2013 doi:10.5664/jcsm.3278
  F04 amplitude, rate, IBI .............. Schleicher 2008; Morris & Miller 1996
                                          doi:10.1016/0301-0511(95)05166-X
  F05 long closures / closed fraction ... Wilkinson 2013; Akerstedt 2010
      (explicit eye-closure block = PERCLOS doi:10.1111/j.1365-2869.2009.00796.x;
      proxy, near-circular with the label; Zheng & Lu 2017 doi:10.1088/1741-2552/aa5a98
      kept separate, excluded by default. NOTE: the SEED-VIG label is (blink + CLOS)/time,
      so F01-F04 (blink duration x rate, closed time) also re-estimate its blink term; the
      default set is "without the explicit eye-closure block", NOT "PERCLOS-free")
  F07 SEM wavelet energy index .......... Magosso 2006 doi:10.1016/j.medengphy.2006.01.002;
                                          Ma, Shi & Lu 2010 doi:10.1109/IEMBS.2010.5627122
  F08 SEM spectral index / amplitude .... Cona 2013 doi:10.5220/0004551304740481
                                          (Zhu, Zheng & Lu 2014, doi:10.1109/IJCNN.2014.6889642,
                                          is a CNN study on EOG; it is NOT used as a source of
                                          these definitions)
  F09 saccade rate/amp/dur/PV ........... Nystrom & Holmqvist 2010 doi:10.3758/BRM.42.1.188;
                                          Russo 2003 doi:10.1016/S1388-2457(03)00008-7;
                                          Hirvonen 2010 doi:10.1016/j.jneumeth.2010.01.010
  F10 main-sequence PV/A, residual ...... Bahill 1975 doi:10.1016/0025-5564(75)90075-9;
                                          Di Stasi 2013 doi:10.1016/j.neubiorev.2013.03.011
  F11 fixation (I-DT dispersion) ........ Bulling 2011 doi:10.1109/TPAMI.2010.86
  F12 Zheng & Lu 36 eye-movement feats .. Zheng & Lu 2017 doi:10.1088/1741-2552/aa5a98;
                                          Zhang 2015 doi:10.1109/NER.2015.7146721
  F13 narrow-band log-PSD, LF/HF ........ Golz 2007 doi:10.1007/s11265-007-0083-4
  F14 2-Hz band differential entropy .... Shi, Jiao & Lu 2013 doi:10.1109/EMBC.2013.6611075;
                                          Duan 2013 doi:10.1109/NER.2013.6695876; Zheng & Lu 2017
  F15 sample entropy, Hjorth ............ Zhang, Wang & Fu 2014 doi:10.1109/TITS.2013.2275192;
                                          Wang 2019 doi:10.1109/ACCESS.2019.2915533
  F17 causal temporal context ........... Shi & Lu 2010 doi:10.1109/IEMBS.2010.5627125 (LDS); Zheng & Lu 2017
      (utilities `trailing_context`, `causal_moving_average`, `kalman_filter_rw`;
       `rts_smoother_rw` is NON-CAUSAL and only for comparison)

Interpretation choices that the papers leave open are documented at each function.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, find_peaks, savgol_filter, sosfilt, sosfilt_zi, sosfiltfilt, welch
from scipy.signal import windows as sigwin

FS = 125
N_SAMP = 1000                 # samples per 8-s epoch
EPOCH_S = 8.0
SCALE_EPOCHS = 37             # trailing window for robust scales (~5 min)
AGG_EPOCHS = 8                # trailing window for event statistics (64 s)
LF_CTX_EPOCHS = 8             # trailing window (64 s) for the sub-1-Hz multitaper bands of F13
MAD_K = 1.4826

MONTAGES = ("trad", "fore")
# Look-ahead in epochs: features of epoch e depend on samples up to the end of epoch e + L.
# 'offline' is the value measured on real recordings by check_lookahead.py (relative change
# < 1e-6 beyond it; see results/feature_check/lookahead.md); 'causal' is exact (unit test).
LOOKAHEAD_EPOCHS = {"offline": 13, "causal": 1}

# Counts of leading epochs in which a trailing estimate did not exist yet and the detector
# was disabled (reset by extract_recording and reported in its `info`).
FALLBACKS: dict[str, int] = {}


def _note(key: str, n) -> None:
    n = int(n)
    if n:
        FALLBACKS[key] = FALLBACKS.get(key, 0) + n


def _ns(t_s: float, fs: int = FS) -> int:
    """Seconds -> whole samples (round(t*fs)); durations are specified in seconds."""
    return int(round(t_s * fs))


def _odd(n: int) -> int:
    return n if n % 2 else n + 1


def _check_fs(fs: int) -> None:
    if fs != FS:
        raise ValueError(f"fs={fs}: this function uses fixed sample windows that are only valid at {FS} Hz")


# =============================================================================== helpers
def montage_signals(eog: np.ndarray, montage: str = "trad") -> dict:
    """Continuous H and V (blinks positive) from epoched electrodes (n_ep, 7, 1000).

    Also returns `elec`: the raw electrodes used for F14 (forehead: ch4-ch7; trad: H and V).
    """
    if montage not in MONTAGES:
        raise ValueError(f"unknown montage {montage!r}")
    x = np.asarray(eog, np.float64).transpose(1, 0, 2).reshape(eog.shape[1], -1)
    if montage == "trad":
        H, V = x[0] - x[1], x[3] - x[2]
        elec = np.vstack([H, V])
        enames = ["H", "V"]
    else:
        H, V = x[4] - x[5], x[3] - x[6]
        elec = x[3:7].copy()
        enames = ["E4", "E5", "E6", "E7"]
    return {"H": H, "V": V, "elec": elec, "elec_names": enames}


def runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start and stop (exclusive) indices of True runs in a boolean vector."""
    m = np.asarray(mask, bool).astype(np.int8)
    d = np.diff(np.concatenate(([0], m, [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def run_length_segments(labels: np.ndarray) -> list[tuple]:
    """Run-length encoding of a multi-level label vector: [(label, start, end_inclusive)].

    This is the correct replacement of the toolbox's getLabeledEvents.m, which pairs
    diff == +k and diff == -k and therefore breaks on 0/1/2/3 label vectors (audit signal-5).
    """
    lab = np.asarray(labels)
    if lab.size == 0:
        return []
    b = np.flatnonzero(np.diff(lab) != 0) + 1
    st = np.r_[0, b]
    en = np.r_[b, lab.size] - 1
    return [(lab[s].item(), int(s), int(e)) for s, e in zip(st, en)]


def _sos(lo, hi, fs=FS, order=4):
    if lo is None:
        return butter(order, hi, btype="low", fs=fs, output="sos")
    if hi is None:
        return butter(order, lo, btype="high", fs=fs, output="sos")
    return butter(order, [lo, hi], btype="band", fs=fs, output="sos")


def zero_phase(x, lo, hi, fs=FS, order=4):
    """Zero-phase Butterworth (sosfiltfilt) along the last axis. NOT causal."""
    return sosfiltfilt(_sos(lo, hi, fs, order), x, axis=-1)


def causal_filter(x, lo, hi, fs=FS, order=4):
    """Forward-only Butterworth (sosfilt) along the last axis, initial state = steady state
    for a constant input equal to the first sample (no start-up step). y[t] uses x[0..t]
    only, so truncating the input never changes earlier outputs."""
    sos = _sos(lo, hi, fs, order)
    x = np.asarray(x, np.float64)
    zi = sosfilt_zi(sos)
    zi = zi.reshape(zi.shape[0], *([1] * (x.ndim - 1)), 2) * x[..., 0][None, ..., None]
    return sosfilt(sos, x, axis=-1, zi=zi)[0]


def filt(x, lo, hi, fs=FS, order=4, causal=False):
    """Butterworth filter: causal (forward only) or offline zero-phase."""
    return causal_filter(x, lo, hi, fs, order) if causal else zero_phase(x, lo, hi, fs, order)


def _lag(v: np.ndarray, lag: int) -> np.ndarray:
    """Shift per-epoch estimates by `lag` epochs (estimate for epoch e from epochs <= e-lag)."""
    v = np.asarray(v, float)
    return v if lag <= 0 else np.r_[np.full(min(lag, len(v)), np.nan), v[:len(v) - lag]]


def _fill_forward(v: np.ndarray, lead: float, key: str, lag: int = 0) -> np.ndarray:
    """Causal replacement of missing per-epoch estimates: shift by `lag` epochs (lag = 1 in the
    causal variant: thresholds for epoch e come from epochs < e only), carry the last finite
    value forward; leading entries without any past estimate get `lead` (a value that disables
    the detector). Missing entries other than the `lag` expected leading ones are counted in
    FALLBACKS[key]."""
    v = _lag(v, lag).copy()
    bad = ~np.isfinite(v)
    if bad.any():
        _note(key, bad.sum() - min(lag, len(v)))
        v = _ffill(v[:, None])[:, 0]
        v[~np.isfinite(v)] = lead
    return v


def epoch_robust_sd(x: np.ndarray, n_samp: int = N_SAMP, valid: np.ndarray | None = None) -> np.ndarray:
    """1.4826 * MAD of x in each consecutive block of n_samp samples (last block may be short)."""
    n_ep = int(np.ceil(len(x) / n_samp))
    out = np.full(n_ep, np.nan)
    for e in range(n_ep):
        seg = x[e * n_samp:(e + 1) * n_samp]
        if valid is not None:
            seg = seg[valid[e * n_samp:(e + 1) * n_samp]]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 10:
            out[e] = MAD_K * np.median(np.abs(seg - np.median(seg)))
    return out


def trailing_nanmedian(v: np.ndarray, n: int) -> np.ndarray:
    """Causal running median over the current and previous n-1 entries (NaNs ignored)."""
    out = np.full(len(v), np.nan)
    for i in range(len(v)):
        w = v[max(0, i - n + 1):i + 1]
        w = w[np.isfinite(w)]
        if w.size:
            out[i] = np.median(w)
    return out


def trailing_scale(x, n_samp=N_SAMP, n_epochs=SCALE_EPOCHS, valid=None, floor=1e-9, lag=0):
    """Per-sample causal robust scale: trailing median of per-epoch robust SDs (epochs
    e-lag-n_epochs+1 .. e-lag for samples of epoch e). Epochs without any past estimate get
    +inf (thresholds built on it detect nothing)."""
    s = trailing_nanmedian(epoch_robust_sd(x, n_samp, valid), n_epochs)
    s = _fill_forward(s, np.inf, "trailing_scale", lag)
    return np.maximum(np.repeat(s, n_samp)[:len(x)], floor)


def parabolic_peak(y: np.ndarray, i: int) -> tuple[float, float]:
    """3-point parabolic interpolation of a local maximum: (value, position)."""
    if 0 < i < len(y) - 1:
        a, b, c = y[i - 1], y[i], y[i + 1]
        den = a - 2 * b + c
        if den < 0:
            p = 0.5 * (a - c) / den
            return float(b - 0.25 * (a - c) * p), float(i + p)
    return float(y[i]), float(i)


def _cross_before(y, ipk, level, imin):
    """Last upward crossing of `level` before ipk (linear interpolation), NaN if none."""
    imin = max(0, imin)
    below = np.flatnonzero(y[imin:ipk + 1] < level)
    if below.size == 0:
        return np.nan
    i = imin + below[-1]
    if i >= ipk:
        return np.nan
    y0, y1 = y[i], y[i + 1]
    return i + (level - y0) / (y1 - y0)


def _cross_after(y, ipk, level, imax):
    """First downward crossing of `level` after ipk (linear interpolation), NaN if none."""
    imax = min(len(y) - 1, imax)
    below = np.flatnonzero(y[ipk:imax + 1] < level)
    if below.size == 0:
        return np.nan
    i = ipk + below[0]
    if i <= ipk:
        return np.nan
    y0, y1 = y[i - 1], y[i]
    return (i - 1) + (y0 - level) / (y0 - y1)


def _ffill(a: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along axis 0 (leading NaNs stay NaN)."""
    a = a.copy()
    for j in range(a.shape[1]):
        col = a[:, j]
        idx = np.where(np.isfinite(col), np.arange(len(col)), -1)
        np.maximum.accumulate(idx, out=idx)
        ok = idx >= 0
        col[ok] = col[idx[ok]]
        a[:, j] = col
    return a


def _q(x, p):
    return float(np.percentile(x, p)) if len(x) else np.nan


def _med(x):
    return float(np.median(x)) if len(x) else np.nan


def _mean(x):
    return float(np.mean(x)) if len(x) else np.nan


def _sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else (0.0 if len(x) == 1 else np.nan)


# =============================================================================== blinks (F01-F05)
def detect_blinks(V: np.ndarray, H: np.ndarray | None = None, fs: int = FS, n_samp: int = N_SAMP,
                  kc: float = 4.0, ko: float = 0.6, rise_fall: tuple = (0.032, 0.504),
                  dur_range: tuple = (0.05, 1.0), amp_frac: float = 0.3, h_frac: float = 0.5,
                  scale_epochs: int = SCALE_EPOCHS, causal: bool = False) -> dict:
    """Scale-free blink detector with sub-sample kinematics (F01-F03).

    Implements the detector specified for F03 (velocity-threshold blink detection as used
    for JDS/AVR measures, Johns 2007; Wilkinson 2013) with amplitude-crossing timing
    (10 % / 50 % / 90 % levels, Pettersson 2013; Picot 2012 doi:10.1007/s00138-011-0374-4):
      * V (blinks positive) is band-passed 0.05-20 Hz (4th-order Butterworth; zero-phase, or
        forward-only when causal=True).
      * velocity v = Savitzky-Golay derivative (56 ms = 7 samples at 125 Hz, order 2) in uV/s.
      * sigma_v = trailing robust SD of v (~5 min including the current epoch; with
        causal=True from the previous epochs only). A candidate starts where
        v > kc*sigma_v and must be followed within rise_fall seconds (32-504 ms) by
        v < -ko*kc*sigma_v (reopening is slower, so its threshold is lower).
      * baseline b = mean V over the 80 ms before the start of the rise (the rise start is
        searched back at most 200 ms); amplitude A = V(peak) - b; on/off = b + 0.1A
        crossings; D50 = width at b + 0.5A; T_closed = time above b + 0.9A. All crossings
        are linearly interpolated.
      * accepted if dur_range[0] <= D_total <= dur_range[1] s, A >= amp_frac * P95 of the
        candidate amplitudes in the trailing 5 min, and the net horizontal step over the
        event |H(off) - H(on)| <= h_frac * A (rejects oblique gaze shifts).
      * overlapping candidates are merged (larger A kept). Incomplete events are dropped
        (no fabricated events, unlike detectBlinks_custom.m; audit signal-4).
    All durations are specified in seconds and converted with round(t*fs).
    Returns per-blink arrays; times are in samples (float) or seconds as named.
    """
    V = np.asarray(V, np.float64)
    N = len(V)
    Vf = filt(V, 0.05, 20.0, fs, causal=causal)
    v = savgol_filter(Vf, _odd(_ns(0.056, fs)), 2, deriv=1, delta=1.0 / fs)
    sig = trailing_scale(v, n_samp, scale_epochs, lag=int(causal))
    thc = kc * sig
    fall = v < -ko * thc
    Hf = filt(H, None, 10.0, fs, causal=causal) if H is not None else None
    rs, rend = runs(v > thc)
    lo_rf, hi_rf = _ns(rise_fall[0], fs), _ns(rise_fall[1], fs)
    n_back = _ns(0.2, fs)                       # rise-start search / crossing search limit
    n_base = _ns(0.08, fs)                      # baseline window
    cand = []
    last_end = -1
    for s, e in zip(rs, rend):
        if s <= last_end:
            continue
        a = max(e, s + lo_rf)
        b = min(N, s + hi_rf + 1)
        if a >= b:
            continue
        j = np.flatnonzero(fall[a:b])
        if j.size == 0:
            continue
        j = a + j[0]
        fe = j
        while fe + 1 < N and fall[fe + 1]:
            fe += 1
        pk = s + int(np.argmax(Vf[s:fe + 1]))
        i0 = s
        while i0 > 0 and v[i0 - 1] > 0.5 * sig[i0 - 1] and s - i0 < n_back:
            i0 -= 1
        base = float(np.mean(Vf[max(0, i0 - (n_base - 1)):i0 + 1]))
        A = float(Vf[pk] - base)
        last_end = fe
        if A <= 0:
            continue
        lvl = lambda f: base + f * A  # noqa: E731
        imax = fe + int(1.5 * fs)
        on10 = _cross_before(Vf, pk, lvl(0.1), i0 - n_back)
        off10 = _cross_after(Vf, pk, lvl(0.1), imax)
        on50 = _cross_before(Vf, pk, lvl(0.5), i0 - n_back)
        off50 = _cross_after(Vf, pk, lvl(0.5), imax)
        on90 = _cross_before(Vf, pk, lvl(0.9), i0 - n_back)
        off90 = _cross_after(Vf, pk, lvl(0.9), imax)
        if not np.all(np.isfinite([on10, off10, on50, off50, on90, off90])):
            continue
        vpk, ppk = parabolic_peak(Vf, pk)
        ion, ioff = int(np.floor(on10)), int(np.ceil(off10))
        ivc = ion + int(np.argmax(v[ion:pk + 1]))
        ivo = pk + int(np.argmin(v[pk:ioff + 1]))
        pcv = parabolic_peak(v, ivc)[0]
        pov = parabolic_peak(-v, ivo)[0]
        if pcv <= 0 or pov <= 0:
            continue
        dh = abs(Hf[min(N - 1, ioff)] - Hf[max(0, ion)]) if Hf is not None else 0.0
        cand.append((on10, ppk, off10, A, off50 - on50, off10 - on10, ppk - on10, off10 - ppk,
                     off90 - on90, pcv, pov, (on90 - on10), sig[pk], dh))
    keys = ["on", "peak", "off", "A", "D50", "Dtot", "Tc", "To", "Tclosed", "PCV", "POV",
            "rise1090", "sigma_v", "dH"]
    if not cand:
        out = {k: np.zeros(0) for k in keys}
    else:
        arr = np.array(cand, float)
        out = {k: arr[:, i] for i, k in enumerate(keys)}
        for k in ["D50", "Dtot", "Tc", "To", "Tclosed", "rise1090"]:
            out[k] = out[k] / fs                                   # samples -> s
    # acceptance: duration, horizontal step, relative amplitude (causal trailing P95)
    if len(out["A"]):
        ok = (out["Dtot"] >= dur_range[0]) & (out["Dtot"] <= dur_range[1]) & (out["dH"] <= h_frac * out["A"])
        tpk = out["peak"]
        p95 = np.empty(len(tpk))
        win = 300 * fs
        for i in range(len(tpk)):
            lo = np.searchsorted(tpk, tpk[i] - win)
            p95[i] = np.percentile(out["A"][lo:i + 1], 95)
        ok &= out["A"] >= amp_frac * p95
        out = {k: v_[ok] for k, v_ in out.items()}
        # merge overlaps (keep larger amplitude)
        keep = np.ones(len(out["A"]), bool)
        last = -1
        for i in range(len(out["A"])):
            if last >= 0 and out["on"][i] < out["off"][last]:
                if out["A"][i] > out["A"][last]:
                    keep[last] = False
                    last = i
                else:
                    keep[i] = False
            else:
                last = i
        out = {k: v_[keep] for k, v_ in out.items()}
    out["AVRc"] = out["A"] / np.maximum(out["PCV"], 1e-9)
    out["AVRo"] = out["A"] / np.maximum(out["POV"], 1e-9)
    out["Vf"], out["v"], out["sigma"] = Vf, v, sig
    return out


def blink_mask(bl: dict, N: int, pad: int = 0) -> np.ndarray:
    """Boolean mask of samples inside detected blinks (on..off) +/- pad samples."""
    m = np.zeros(N, bool)
    for a, b in zip(bl["on"], bl["off"]):
        m[max(0, int(np.floor(a)) - pad):min(N, int(np.ceil(b)) + 1 + pad)] = True
    return m


def interpolate_masked(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Linear interpolation across masked samples (used to remove blinks before SEM analysis)."""
    if not mask.any() or mask.all():
        return x.copy()
    y = x.copy()
    idx = np.arange(len(x))
    y[mask] = np.interp(idx[mask], idx[~mask], x[~mask])
    return y


def _epoch_of(t_samples, n_samp=N_SAMP):
    return np.floor(np.asarray(t_samples, float) / n_samp).astype(int)


def _window_events(ep_of_event: np.ndarray, e: int, win: int):
    lo = np.searchsorted(ep_of_event, e - win + 1, "left")
    hi = np.searchsorted(ep_of_event, e, "right")
    return lo, hi


def blink_epoch_features(bl: dict, n_ep: int, n_samp: int = N_SAMP, fs: int = FS,
                         win: int = AGG_EPOCHS) -> tuple[np.ndarray, list[str]]:
    """F01-F04 per epoch over a trailing 64-s window + missingness flag.

    F01 (Caffier 2003; Schleicher 2008; Cori 2019): median/mean/SD/P90 of D50 and D_total,
        fraction of blinks with D_total > 0.5 s.
    F02 (Caffier 2003; Hu & Zheng 2009): median/SD/P90 of closing time Tc, reopening time
        To and closed time T_closed; median To/Tc.
    F03 (Johns 2007; Wilkinson 2013): mean/SD of AVRc and AVRo (s), median PCV and POV
        divided by the trailing robust velocity scale, median PCV/A, median 10-90 % rise time
        (robust AVR variant).
    F04 (Schleicher 2008; Morris & Miller 1996): median and P10 of A / trailing-5-min median A,
        blink rate (/min), IBI mean (s) and IBI CV.
    Windows with no blink carry the last value forward and set `blink_missing` = 1 (no 0
    imputation). Leading epochs before the first blink stay NaN (impute on training persons).
    """
    names = (["F01_D50_med", "F01_D50_mean", "F01_D50_sd", "F01_D50_p90",
              "F01_Dtot_med", "F01_Dtot_mean", "F01_Dtot_sd", "F01_Dtot_p90", "F01_frac_long05",
              "F02_Tc_med", "F02_Tc_sd", "F02_Tc_p90", "F02_To_med", "F02_To_sd", "F02_To_p90",
              "F02_Tclosed_med", "F02_Tclosed_sd", "F02_Tclosed_p90", "F02_ToTc_med",
              "F03_AVRc_mean", "F03_AVRc_sd", "F03_AVRo_mean", "F03_AVRo_sd",
              "F03_PCVn_med", "F03_POVn_med", "F03_PCV_over_A_med", "F03_rise1090_med",
              "F04_Anorm_med", "F04_Anorm_p10"])
    rate_names = ["F04_rate_per_min", "F04_IBI_mean", "F04_IBI_cv"]
    X = np.full((n_ep, len(names)), np.nan)
    R = np.zeros((n_ep, len(rate_names)))
    flag = np.ones(n_ep)
    tpk = bl["peak"]
    ep = _epoch_of(tpk, n_samp)
    # amplitude normalised by trailing-5-min median amplitude (causal)
    An = np.full(len(tpk), np.nan)
    for i in range(len(tpk)):
        lo = np.searchsorted(tpk, tpk[i] - 300 * fs)
        An[i] = bl["A"][i] / np.median(bl["A"][lo:i + 1])
    for e in range(n_ep):
        lo, hi = _window_events(ep, e, win)
        n_win_ep = min(win, e + 1)
        if hi - lo == 0:
            R[e] = [0.0, np.nan, np.nan]
            continue
        flag[e] = 0.0
        sl = slice(lo, hi)
        d50, dt = bl["D50"][sl], bl["Dtot"][sl]
        tc, to, tcl = bl["Tc"][sl], bl["To"][sl], bl["Tclosed"][sl]
        avc, avo = bl["AVRc"][sl], bl["AVRo"][sl]
        sv = bl["sigma_v"][sl]
        X[e] = [_med(d50), _mean(d50), _sd(d50), _q(d50, 90),
                _med(dt), _mean(dt), _sd(dt), _q(dt, 90), float(np.mean(dt > 0.5)),
                _med(tc), _sd(tc), _q(tc, 90), _med(to), _sd(to), _q(to, 90),
                _med(tcl), _sd(tcl), _q(tcl, 90), _med(to / np.maximum(tc, 1e-3)),
                _mean(avc), _sd(avc), _mean(avo), _sd(avo),
                _med(bl["PCV"][sl] / sv), _med(bl["POV"][sl] / sv), _med(1.0 / avc),
                _med(bl["rise1090"][sl]), _med(An[sl]), _q(An[sl], 10)]
        ibi = np.diff(tpk[sl]) / fs
        R[e, 0] = (hi - lo) / (n_win_ep * EPOCH_S / 60.0)
        R[e, 1] = _mean(ibi) if len(ibi) else np.nan
        R[e, 2] = (_sd(ibi) / _mean(ibi)) if len(ibi) > 1 else np.nan
    X = _ffill(X)
    R[:, 1:] = _ffill(R[:, 1:])
    return np.hstack([X, R, flag[:, None]]), names + rate_names + ["F04_blink_missing"]


def closure_features(V: np.ndarray, bl: dict, n_ep: int, n_samp: int = N_SAMP, fs: int = FS,
                     win: int = AGG_EPOCHS, causal: bool = False) -> tuple[np.ndarray, list[str]]:
    """F05 eye-closure block (PERCLOS PROXY; Wilkinson 2013; Akerstedt 2010; Zheng & Lu 2017).

    WARNING: the SEED-VIG label is PERCLOS from eye-tracking glasses ((blink + CLOS)/time,
    Zheng & Lu 2017). An EOG closure fraction re-estimates the label itself, so this block
    is reported separately and is NOT part of the default literature set. (The blink blocks
    F01-F04 also re-estimate the 'blink' term of PERCLOS; see README, sensitivity runs.)
    A sample is 'closed' when the 20-Hz low-passed V exceeds a causal baseline (25th
    percentile of the trailing 20 one-second block medians) by more than half the trailing
    5-min median blink amplitude. Features: number of closures >= 0.5 s in the epoch,
    closed fraction of the epoch, longest closure overlapping the epoch (s), and the closed
    fraction over the trailing 64 s. With causal=True: closures are counted in the epoch where
    they end, the longest closure is clipped at the epoch end, and the reference amplitude
    comes from previous epochs only. The closed fraction over the
    trailing 64 s. Epochs before the first blink have no reference amplitude: closure
    detection is disabled there (no whole-recording fallback).
    The sample-level 'closed' mask comes from closure_mask() (refactored 28 Oct 2026 for
    eog_events.py; outputs unchanged, checked against the litc/lit caches by tests/test_deep.py).
    """
    closed, st, en = closure_mask(V, bl, n_ep, n_samp, fs, causal)
    long_ = (en - st) >= int(0.5 * fs)
    X = np.zeros((n_ep, 4))
    cfrac = np.array([closed[e * n_samp:(e + 1) * n_samp].mean() for e in range(n_ep)])
    for e in range(n_ep):
        a, b = e * n_samp, (e + 1) * n_samp
        ov = (st < b) & (en > a)
        # offline: long closures that START in the epoch; causal: long closures that END in it
        X[e, 0] = np.sum(ov & long_ & ((en <= b) if causal else (st >= a)))
        X[e, 1] = cfrac[e]
        stop = np.minimum(en[ov], b) if causal else en[ov]
        X[e, 2] = ((stop - st[ov]).max() / fs) if ov.any() else 0.0
        X[e, 3] = cfrac[max(0, e - win + 1):e + 1].mean()
    return X, ["F05_n_long", "F05_closed_frac", "F05_max_closure_s", "F05_closed_frac_64s"]


def closure_mask(V: np.ndarray, bl: dict, n_ep: int, n_samp: int = N_SAMP, fs: int = FS,
                 causal: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample-level eye-closure mask of closure_features (F05; see its docstring for the rule).

    Returns (closed, st, en): boolean mask of 'closed' samples and the start / stop
    (exclusive) sample indices of its runs. Closures >= 0.5 s are the 'long closures' of F05.
    """
    N = len(V)
    Vl = filt(V, None, 20.0, fs, causal=causal)
    nb = int(np.ceil(N / fs))
    bmed = np.array([np.median(Vl[i * fs:(i + 1) * fs]) for i in range(nb)])
    base = np.array([np.percentile(bmed[max(0, i - 19):i + 1], 25) for i in range(nb)])
    base = np.repeat(base, fs)[:N]
    tpk = bl["peak"]
    aref_ep = np.full(n_ep, np.nan)
    ep = _epoch_of(tpk, n_samp)
    for e in range(n_ep):
        lo, hi = _window_events(ep, e, SCALE_EPOCHS)
        if hi > lo:
            aref_ep[e] = np.median(bl["A"][lo:hi])
    aref_ep = _fill_forward(aref_ep, np.inf, "closure_aref", lag=int(causal))
    aref = np.repeat(aref_ep, n_samp)[:N]
    closed = (Vl - base) > 0.5 * aref
    st, en = runs(closed)
    return closed, st, en


# =============================================================================== saccades (F09-F11)
def adaptive_velocity_thresholds(av: np.ndarray, valid: np.ndarray, n_samp: int = N_SAMP,
                                 scale_epochs: int = SCALE_EPOCHS, pt_k: float = 6.0,
                                 ot_k: float = 3.0, step: int = 2, lag: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Nystrom & Holmqvist (2010) data-driven peak/onset velocity thresholds, made causal.

    For each epoch the pool is |v| over the trailing ~5 min (valid samples only; with lag = 1
    the pool ends with the previous epoch). Start with
    PT = P99 (scale-free instead of a deg/s value), iterate PT = mean + pt_k*SD of samples
    below PT until the change is < 1 %; onset threshold OT = mean + ot_k*SD below PT.
    Epochs whose pool is too small carry the last thresholds forward; leading epochs without
    any estimate get +inf (no saccade detected), counted in FALLBACKS.
    Returns per-epoch (PT, OT).
    """
    N = len(av)
    n_ep = int(np.ceil(N / n_samp))
    PT, OT = np.full(n_ep, np.nan), np.full(n_ep, np.nan)
    for e in range(n_ep):
        a, b = max(0, e - scale_epochs + 1) * n_samp, min(N, (e + 1) * n_samp)
        pool = av[a:b][valid[a:b]][::step]
        if pool.size < 50:
            continue
        pt = np.percentile(pool, 99)
        for _ in range(100):
            below = pool[pool < pt]
            new = below.mean() + pt_k * below.std()
            done = abs(new - pt) < 0.01 * pt
            pt = new
            if done:
                break
        below = pool[pool < pt]
        PT[e], OT[e] = pt, below.mean() + ot_k * below.std()
    PT = _fill_forward(PT, np.inf, "saccade_PT", lag)
    OT = _fill_forward(OT, np.inf, "saccade_OT", lag)
    return PT, OT


def detect_saccades(X: np.ndarray, exclude: np.ndarray | None = None, fs: int = FS,
                    n_samp: int = N_SAMP, lp: float = 30.0, dur_range: tuple = (0.024, 0.2),
                    scale_epochs: int = SCALE_EPOCHS, causal: bool = False) -> dict:
    """Velocity-threshold saccade detector (F09; Nystrom & Holmqvist 2010; Bulling 2011).

    X is low-passed at 30 Hz (4th-order Butterworth; zero-phase, or forward-only when
    causal=True); velocity = Savitzky-Golay derivative (40 ms = 5 samples at 125 Hz, order 2).
    Samples in `exclude` (blinks +/- 50 ms) are ignored. Peaks of |v| above the adaptive PT
    are grown to onset/offset where |v| falls below OT (same velocity sign). Durations
    dur_range in seconds (24-200 ms = 3-25 samples at 125 Hz, converted with round(t*fs)).
    Amplitude A_s = |X(off) - X(on)|; PV = parabolic-interpolated peak |v| (uV/s).
    """
    X = np.asarray(X, np.float64)
    N = len(X)
    Xf = filt(X, None, lp, fs, causal=causal)
    v = savgol_filter(Xf, _odd(_ns(0.04, fs)), 2, deriv=1, delta=1.0 / fs)
    dmin, dmax = _ns(dur_range[0], fs), _ns(dur_range[1], fs)
    av = np.abs(v)
    valid = np.ones(N, bool) if exclude is None else ~np.asarray(exclude, bool)
    PT, OT = adaptive_velocity_thresholds(av, valid, n_samp, scale_epochs, lag=int(causal))
    pts = np.repeat(PT, n_samp)[:N]
    ots = np.repeat(OT, n_samp)[:N]
    pk, _ = find_peaks(np.where(valid, av, 0.0), height=pts)
    taken = np.zeros(N, bool)
    ev = []
    for p in pk[np.argsort(-av[pk])]:
        if taken[p]:
            continue
        sg = np.sign(v[p])
        on = p
        while on > 0 and av[on - 1] > ots[on - 1] and np.sign(v[on - 1]) == sg and valid[on - 1] and not taken[on - 1]:
            on -= 1
        off = p
        while off < N - 1 and av[off + 1] > ots[off + 1] and np.sign(v[off + 1]) == sg and valid[off + 1] and not taken[off + 1]:
            off += 1
        taken[on:off + 1] = True
        dur = off - on + 1
        if not (dmin <= dur <= dmax):
            continue
        a0, a1 = max(0, on - 1), min(N - 1, off + 1)
        amp = abs(Xf[a1] - Xf[a0])
        pv = parabolic_peak(av, p)[0]
        ev.append((on, off, p, amp, dur / fs, pv, sg))
    keys = ["on", "off", "peak", "amp", "dur", "pv", "sign"]
    if ev:
        arr = np.array(sorted(ev), float)
        out = {k: arr[:, i] for i, k in enumerate(keys)}
    else:
        out = {k: np.zeros(0) for k in keys}
    out["Xf"], out["v"], out["PT"], out["OT"] = Xf, v, PT, OT
    return out


def saccade_epoch_features(sh: dict, sv: dict, n_ep: int, n_samp: int = N_SAMP, fs: int = FS,
                           win: int = AGG_EPOCHS) -> tuple[np.ndarray, list[str]]:
    """F09 + F10 per epoch over the trailing 64 s.

    F09 (Russo 2003; Hirvonen 2010; Nystrom & Holmqvist 2010): horizontal saccade rate (/min),
        median A_s / trailing-5-min P90(A_s), median duration (s), median PV / trailing-5-min
        P90(PV), fraction of large saccades (A_s > 0.5*P90); vertical saccade rate and
        normalised median amplitude.
    F10 (Bahill 1975; Di Stasi 2013): median q = PV/A_s (1/s, gain-free) and median
        main-sequence residual r = log PV - (alpha + beta log A_s). alpha/beta are fitted on
        the recording's saccades up to the current epoch with an expanding window frozen
        after 300 s (causal; one trimming pass |res| > 2.5 MAD). Missing -> carried forward.
    """
    names = ["F09_H_rate", "F09_H_amp_norm_med", "F09_H_dur_med", "F09_H_pv_norm_med",
             "F09_H_frac_large", "F09_V_rate", "F09_V_amp_norm_med", "F10_q_med", "F10_ms_resid_med",
             "F09_sacc_missing"]
    Xo = np.full((n_ep, len(names)), np.nan)

    def norm_by_trailing_p90(t, val):
        out = np.full(len(t), np.nan)
        for i in range(len(t)):
            lo = np.searchsorted(t, t[i] - 300 * fs)
            out[i] = val[i] / max(np.percentile(val[lo:i + 1], 90), 1e-9)
        return out

    th, tv = sh["on"], sv["on"]
    an_h = norm_by_trailing_p90(th, sh["amp"])
    pn_h = norm_by_trailing_p90(th, sh["pv"])
    an_v = norm_by_trailing_p90(tv, sv["amp"])
    eh, evv = _epoch_of(th, n_samp), _epoch_of(tv, n_samp)
    la, lp = np.log(np.maximum(sh["amp"], 1e-6)), np.log(np.maximum(sh["pv"], 1e-6))
    fit = None
    for e in range(n_ep):
        n_win_ep = min(win, e + 1)
        mins = n_win_ep * EPOCH_S / 60.0
        lo, hi = _window_events(eh, e, win)
        # main-sequence fit: expanding window until 300 s, then frozen
        if (e + 1) * EPOCH_S <= 300 or fit is None:
            m = th < (e + 1) * n_samp
            if m.sum() >= 10:
                A_ = np.c_[np.ones(m.sum()), la[m]]
                coef = np.linalg.lstsq(A_, lp[m], rcond=None)[0]
                res = lp[m] - A_ @ coef
                keep = np.abs(res - np.median(res)) <= 2.5 * MAD_K * np.median(np.abs(res - np.median(res))) + 1e-12
                if keep.sum() >= 5:
                    coef = np.linalg.lstsq(A_[keep], lp[m][keep], rcond=None)[0]
                fit = coef
        Xo[e, 0] = (hi - lo) / mins
        Xo[e, 9] = float(hi - lo == 0)
        if hi > lo:
            sl = slice(lo, hi)
            Xo[e, 1] = _med(an_h[sl])
            Xo[e, 2] = _med(sh["dur"][sl])
            Xo[e, 3] = _med(pn_h[sl])
            Xo[e, 4] = float(np.mean(an_h[sl] > 0.5))
            Xo[e, 7] = _med(sh["pv"][sl] / np.maximum(sh["amp"][sl], 1e-9))
            if fit is not None:
                Xo[e, 8] = _med(lp[sl] - (fit[0] + fit[1] * la[sl]))
        lo, hi = _window_events(evv, e, win)
        Xo[e, 5] = (hi - lo) / mins
        if hi > lo:
            Xo[e, 6] = _med(an_v[lo:hi])
    cols = [1, 2, 3, 4, 6, 7, 8]
    Xo[:, cols] = _ffill(Xo[:, cols])
    return Xo, names


def detect_fixations(Hs: np.ndarray, Vs: np.ndarray, event_mask: np.ndarray, fs: int = FS,
                     n_samp: int = N_SAMP, min_dur: float = 0.2, k: float = 3.0, lag: int = 0) -> dict:
    """I-DT dispersion fixation detector (F11; Bulling 2011), scale-free threshold.

    Hs/Vs are 10-Hz LOW-passed H/V (no high-pass: a 0.5-Hz high-pass turns every saccade step
    into a decaying transient and splits fixations; found by the unit test).
    Dispersion of a window = (max-min H) + (max-min V).
    Threshold per epoch = k * trailing-5-min median dispersion of consecutive 200-ms windows
    of event-free samples (causal; carried forward when an epoch has none, and -inf =
    detector disabled before the first estimate). Inside every gap between blinks/saccades a window
    of >= 200 ms is grown while its dispersion stays below threshold; each grown window is
    one fixation. Returns onsets/offsets (samples, offset exclusive) and durations (s).
    """
    N = len(Hs)
    w = _ns(min_dur, fs)
    n_ep = int(np.ceil(N / n_samp))
    disp_ep = np.full(n_ep, np.nan)
    for e in range(n_ep):
        a, b = e * n_samp, min(N, (e + 1) * n_samp)
        ds = []
        for s in range(a, b - w + 1, w):
            if not event_mask[s:s + w].any():
                ds.append(np.ptp(Hs[s:s + w]) + np.ptp(Vs[s:s + w]))
        if ds:
            disp_ep[e] = np.median(ds)
    thr = _fill_forward(k * trailing_nanmedian(disp_ep, SCALE_EPOCHS), -np.inf, "fixation_thr", lag)
    st, en = runs(~event_mask)
    on, off = [], []
    for a, b in zip(st, en):
        i = a
        while i + w <= b:
            t = thr[min(i // n_samp, n_ep - 1)]
            hmax, hmin = Hs[i:i + w].max(), Hs[i:i + w].min()
            vmax, vmin = Vs[i:i + w].max(), Vs[i:i + w].min()
            if (hmax - hmin) + (vmax - vmin) > t:
                i += 1
                continue
            j = i + w
            while j < b:
                hmax, hmin = max(hmax, Hs[j]), min(hmin, Hs[j])
                vmax, vmin = max(vmax, Vs[j]), min(vmin, Vs[j])
                if (hmax - hmin) + (vmax - vmin) > t:
                    break
                j += 1
            on.append(i)
            off.append(j)
            i = j
    on, off = np.array(on, float), np.array(off, float)
    return {"on": on, "off": off, "dur": (off - on) / fs}


def fixation_epoch_features(fx: dict, n_ep: int, n_samp: int = N_SAMP, win: int = AGG_EPOCHS,
                            assign: str = "onset"):
    """F11 per epoch over the trailing 64 s: mean, SD, P90 of fixation duration and the
    fraction of time in fixation (Bulling 2011; Schleicher 2008).
    assign='onset' puts a fixation in the epoch where it starts (offline: a long fixation is
    then known only later); assign='offset' puts it in the epoch where it ENDS (causal: a
    fixation is counted once it is complete)."""
    X = np.full((n_ep, 4), np.nan)
    if assign == "onset":
        t_ev = fx["on"]
    elif assign == "offset":
        t_ev = fx["off"] - 1                       # last sample of the fixation
    else:
        raise ValueError(assign)
    order = np.argsort(t_ev, kind="stable")
    fx = {k: np.asarray(v)[order] for k, v in fx.items()}
    ep = _epoch_of(np.asarray(t_ev)[order], n_samp)
    for e in range(n_ep):
        lo, hi = _window_events(ep, e, win)
        n_win_ep = min(win, e + 1)
        d = fx["dur"][lo:hi]
        X[e] = [_mean(d), _sd(d), _q(d, 90), float(d.sum() / (n_win_ep * EPOCH_S)) if len(d) else 0.0]
    X[:, :3] = _ffill(X[:, :3])
    return X, ["F11_fix_dur_mean", "F11_fix_dur_sd", "F11_fix_dur_p90", "F11_fix_time_frac"]


# =============================================================================== SEM (F07-F08)
SWT_LEVELS = 10
SEM_WIN = 4096


def _swt_detail_energy(seg: np.ndarray, tail: int) -> np.ndarray:
    """Energy of SWT (db4, 10 levels, norm=True) detail levels 1..10 over the last `tail` samples."""
    import pywt

    c = pywt.swt(seg, "db4", level=SWT_LEVELS, trim_approx=True, norm=True)
    # c = [cA10, cD10, cD9, ..., cD1]
    det = np.array(c[1:])[::-1]            # rows: level 1 .. 10
    return det[:, -tail:]


def sem_index(det_tail: np.ndarray) -> float:
    """Magosso 2006 / Ma 2010 SEM energy ratio with unit weights:
    f = E(levels 7-10) / (E(levels 3-5) + E(levels 7-10)); levels 1, 2, 6 excluded.
    At 125 Hz: levels 7-10 = 0.06-0.98 Hz (slow), levels 3-5 = 1.95-15.6 Hz (fast)."""
    e = np.sum(det_tail ** 2, axis=-1)
    slow, fast = e[6:10].sum(), e[2:5].sum()
    return float(slow / (slow + fast + 1e-30))


def sem_features(H: np.ndarray, V: np.ndarray, mask: np.ndarray | None, n_ep: int,
                 n_samp: int = N_SAMP, fs: int = FS, causal: bool = False) -> tuple[np.ndarray, list[str]]:
    """F07 + F08 per epoch from the RAW (no 0.5-Hz high-pass) H and V.

    F07 (Magosso 2006; Ma, Shi & Lu 2010): stationary wavelet transform (db4, 10 levels) of
        the 4096-sample (32.8-s) window ending at the end of the current epoch (trailing; the
        first epochs are left-padded by symmetric reflection of the available past), linear
        trend removed. f = SEM energy ratio over the last 1000 coefficients (unit weights
        W_j = 1, because Magosso's trained weights are not published in usable form),
        logit(f), and the SEM proportion = fraction of 16 sub-windows (~0.5 s) with f > 0.5
        and f > 0.8. LIMITATION: pywt.swt uses periodic extension and its coefficients are
        not phase-aligned; the db4 filters of levels 8-10 (~900-3600 samples) are as long as
        the window, so those levels mix in wrapped-around samples from the window start and
        the jump at the window edge. F07 is therefore an index of the trailing 32-s window,
        NOT of the 8-s epoch. There is no future leakage.
    F08 (Cona 2013): Welch PSD (Hann, 2048, 50 %) of the same window:
        SEMratio = log10 P[0.1-0.6] / P[1-8], log10 P[0.1-0.6], log10 P[0.6-1]; for H also
        the epoch peak-to-peak of 0.1-0.6 Hz H divided by its trailing-5-min median (gain-free)
        and the slow-velocity fraction (0.1-1 Hz velocity between 1x and 5x its trailing
        robust SD, outside blinks/saccades). The 0.1-0.6 / 0.1-1 Hz band-passes are
        zero-phase (impulse support +/-13-24 s) or forward-only when causal=True.
    Blinks (+/- 100 ms) are removed by linear interpolation before the analysis (`mask`).
    Requires fs = 125 Hz (window and SWT levels are fixed sample counts).
    """
    from scipy.signal import detrend

    _check_fs(fs)
    sigs = {"H": H, "V": V}
    if mask is not None:
        sigs = {k: interpolate_masked(x, mask) for k, x in sigs.items()}
    N = len(H)
    names, cols = [], []
    sub = np.array_split(np.arange(n_samp), 16)
    for nm, x in sigs.items():
        F = np.full((n_ep, 7), np.nan)
        for e in range(n_ep):
            end = min(N, (e + 1) * n_samp)
            if end >= SEM_WIN:
                seg = x[end - SEM_WIN:end]
            else:
                past = x[:end]
                reps = int(np.ceil(SEM_WIN / len(past))) + 1
                ext = np.concatenate([past[::-1] if r % 2 else past for r in range(reps)][::-1])
                seg = ext[-SEM_WIN:]
            seg = detrend(seg, type="linear")
            tail = end - e * n_samp
            det = _swt_detail_energy(seg, tail)
            f = sem_index(det)
            fs_sub = np.array([sem_index(det[:, s_]) for s_ in sub if s_[-1] < tail])
            fr, P = welch(seg, fs=fs, window="hann", nperseg=2048, noverlap=1024, detrend="linear")
            band = lambda lo, hi: P[(fr >= lo) & (fr < hi)].sum() * (fr[1] - fr[0]) + 1e-12  # noqa: E731
            F[e] = [f, np.log(np.clip(f, 1e-4, 1 - 1e-4) / (1 - np.clip(f, 1e-4, 1 - 1e-4))),
                    float(np.mean(fs_sub > 0.5)), float(np.mean(fs_sub > 0.8)),
                    np.log10(band(0.1, 0.6) / band(1.0, 8.0)), np.log10(band(0.1, 0.6)), np.log10(band(0.6, 1.0))]
        cols.append(F)
        names += [f"F07_{nm}_sem_f", f"F07_{nm}_sem_logit", f"F07_{nm}_semprop050", f"F07_{nm}_semprop080",
                  f"F08_{nm}_semratio", f"F08_{nm}_logP_0.1-0.6", f"F08_{nm}_logP_0.6-1"]
    # SEM amplitude and slow-velocity fraction (H)
    hs = filt(sigs["H"], 0.1, 0.6, fs, causal=causal)
    p2p = np.array([np.ptp(hs[e * n_samp:(e + 1) * n_samp]) for e in range(n_ep)])
    ref = trailing_nanmedian(p2p, SCALE_EPOCHS)
    hv = np.gradient(filt(sigs["H"], 0.1, 1.0, fs, causal=causal)) * fs
    sc = trailing_scale(hv, n_samp, lag=int(causal))
    slow = (np.abs(hv) > sc) & (np.abs(hv) < 5 * sc)
    if mask is not None:
        slow &= ~mask
    sfrac = np.array([slow[e * n_samp:(e + 1) * n_samp].mean() for e in range(n_ep)])
    cols.append(np.c_[p2p / np.maximum(ref, 1e-9), sfrac])
    names += ["F08_H_sem_amp_norm", "F08_H_slowvel_frac"]
    return np.hstack(cols), names


# =============================================================================== F12 Zheng & Lu
def ricker(sigma: float, half_width: float = 5.0) -> np.ndarray:
    """Mexican-hat (Ricker) wavelet sampled on +/- half_width*sigma samples.
    psi(t) = 2/(sqrt(3 sigma) pi^0.25) (1 - t^2/sigma^2) exp(-t^2/(2 sigma^2)).
    Implemented by hand because scipy >= 1.15 removed signal.ricker/cwt."""
    hw = int(np.ceil(half_width * sigma))
    t = np.arange(-hw, hw + 1, dtype=float)
    return 2 / (np.sqrt(3 * sigma) * np.pi ** 0.25) * (1 - t ** 2 / sigma ** 2) * np.exp(-t ** 2 / (2 * sigma ** 2))


def cwt_events(x: np.ndarray, kind: str, scale: float = 8.0, k: float = 3.5, fs: int = FS,
               n_samp: int = N_SAMP, max_span: float | None = None, lag: int = 0) -> dict:
    """Zheng & Lu (2017) / Zhang (2015) CWT peak-pattern event detector.

    Mexican-hat CWT at scale 8 (pseudo-frequency 0.25*125/8 = 3.9 Hz). Positive coefficient
    peaks above theta are coded '1', negative peaks below -theta '0'. theta = k * causal
    trailing robust SD of the coefficients (the paper's theta_h/theta_l are not published).
    kind='blink': '010' on V within max_span (0.5 s); amplitude = x(1) - mean(x(0), x(0')),
                  duration = time between the two '0' peaks.
    kind='saccade': adjacent '01' or '10' on H within max_span (0.4 s); amplitude =
                  |x(second) - x(first)|, duration = time between the peaks.
    The line-fit slope/correlation constraints of the original are not reproduced.
    """
    c = np.convolve(x, ricker(scale), mode="same")
    th = k * trailing_scale(c, n_samp, lag=lag)
    pp, _ = find_peaks(c, height=th)
    pn, _ = find_peaks(-c, height=th)
    loc = np.r_[pp, pn]
    sym = np.r_[np.ones(len(pp), int), np.zeros(len(pn), int)]
    o = np.argsort(loc)
    loc, sym = loc[o], sym[o]
    ev = []
    i = 0
    if kind == "blink":
        span = int((max_span or 0.5) * fs)
        while i + 2 < len(loc):
            if sym[i] == 0 and sym[i + 1] == 1 and sym[i + 2] == 0 and loc[i + 2] - loc[i] <= span:
                amp = x[loc[i + 1]] - 0.5 * (x[loc[i]] + x[loc[i + 2]])
                ev.append((loc[i + 1], amp, (loc[i + 2] - loc[i]) / fs))
                i += 3
            else:
                i += 1
    else:
        span = int((max_span or 0.4) * fs)
        while i + 1 < len(loc):
            if sym[i] != sym[i + 1] and loc[i + 1] - loc[i] <= span:
                ev.append(((loc[i] + loc[i + 1]) // 2, abs(x[loc[i + 1]] - x[loc[i]]), (loc[i + 1] - loc[i]) / fs))
                i += 2
            else:
                i += 1
    arr = np.array(ev, float).reshape(-1, 3)
    return {"t": arr[:, 0], "amp": arr[:, 1], "dur": arr[:, 2]}


def zheng_lu_36(bev: dict, sev: dict, n_ep: int, n_samp: int = N_SAMP, fs: int = FS,
                var_epochs: int = 4) -> tuple[np.ndarray, list[str]]:
    """The 36 eye-movement features of Zheng & Lu 2017 (Table in J Neural Eng 14:026017).

    Per 8-s epoch, for blinks and for saccades (13 each): max/mean/sum of the instantaneous
    rate (1/inter-event interval, Hz - the paper's 'rate' is undefined, this choice is ours),
    max/min/mean amplitude, mean/max of the rate variance and amplitude variance (variance
    per epoch over the current and previous 3 epochs = 32 s), power and mean power of the
    amplitude, number. 'Fixation' group (10): mean/max of blink-duration variance and of
    saccade-duration variance, max/min/mean of blink duration and saccade duration.
    Values are 0 where no event exists (as in the SEED-VIG feature files).
    """
    def per_epoch(ev):
        t = ev["t"]
        ep = _epoch_of(t, n_samp)
        rate = np.r_[np.nan, fs / np.maximum(np.diff(t), 1)] if len(t) else np.zeros(0)
        out = np.zeros((n_ep, 13))
        vr, va, vd = np.zeros(n_ep), np.zeros(n_ep), np.zeros(n_ep)
        dstat = np.zeros((n_ep, 3))
        for e in range(n_ep):
            lo, hi = np.searchsorted(ep, e, "left"), np.searchsorted(ep, e, "right")
            if hi == lo:
                continue
            r = rate[lo:hi]
            r = r[np.isfinite(r)]
            a, d = ev["amp"][lo:hi], ev["dur"][lo:hi]
            out[e, 0:3] = [r.max(), r.mean(), r.sum()] if len(r) else [0, 0, 0]
            out[e, 3:6] = [a.max(), a.min(), a.mean()]
            out[e, 10:13] = [np.sum(a ** 2), np.mean(a ** 2), hi - lo]
            vr[e] = r.var(ddof=1) if len(r) > 1 else 0.0
            va[e] = a.var(ddof=1) if len(a) > 1 else 0.0
            vd[e] = d.var(ddof=1) if len(d) > 1 else 0.0
            dstat[e] = [d.max(), d.min(), d.mean()]
        for e in range(n_ep):
            s = slice(max(0, e - var_epochs + 1), e + 1)
            out[e, 6:10] = [vr[s].mean(), vr[s].max(), va[s].mean(), va[s].max()]
        vdm = np.array([[vd[max(0, e - var_epochs + 1):e + 1].mean(), vd[max(0, e - var_epochs + 1):e + 1].max()]
                        for e in range(n_ep)])
        return out, vdm, dstat

    gnames = ["rate_max", "rate_mean", "rate_sum", "amp_max", "amp_min", "amp_mean", "ratevar_mean",
              "ratevar_max", "ampvar_mean", "ampvar_max", "amp_power", "amp_meanpower", "number"]
    b, bvd, bd = per_epoch(bev)
    s, svd, sd = per_epoch(sev)
    X = np.hstack([b, s, bvd, svd, bd, sd])
    names = ([f"F12_blink_{g}" for g in gnames] + [f"F12_sacc_{g}" for g in gnames]
             + ["F12_fix_blinkdurvar_mean", "F12_fix_blinkdurvar_max", "F12_fix_saccdurvar_mean",
                "F12_fix_saccdurvar_max", "F12_fix_blinkdur_max", "F12_fix_blinkdur_min",
                "F12_fix_blinkdur_mean", "F12_fix_saccdur_max", "F12_fix_saccdur_min", "F12_fix_saccdur_mean"])
    return X, names


# =============================================================================== spectral (F13-F14)
def _dpss_psd(x: np.ndarray, fs: int = FS, nw: float = 2.0, k: int = 3):
    """Multitaper PSD (DPSS, NW=2, 3 tapers) of the rows of x (..., T); resolution 2W = 2*NW*fs/T."""
    T = x.shape[-1]
    tap = sigwin.dpss(T, nw, k)                                   # (k, T)
    X = np.fft.rfft(x[..., None, :] * tap, axis=-1)
    P = (np.abs(X) ** 2).mean(-2) / fs
    P[..., 1:-1] *= 2
    return np.fft.rfftfreq(T, 1 / fs), P


def _trailing_windows(sig2: np.ndarray, e0: int, e1: int, k: int) -> np.ndarray:
    """Concatenate epochs e-k+1..e for e in [e0, e1) -> (e1-e0, S, k*T); requires e0 >= k-1."""
    return np.concatenate([sig2[e0 - k + 1 + j:e1 - k + 1 + j] for j in range(k)], axis=-1)


def spectral_features(sig2: np.ndarray, snames: list[str], fs: int = FS,
                      ctx_epochs: int = LF_CTX_EPOCHS) -> tuple[np.ndarray, list[str]]:
    """F13 (Golz 2007) per epoch for each signal row of sig2 (n_ep, S, T), epochs contiguous.

    (a) Welch (Hann, 2-s segments, 50 % overlap) power in 22 one-Hz bands from 0.5 to
        22.5 Hz of the current 8-s epoch, expressed as log10 RELATIVE power (band / total
        0.5-22.5 Hz; gain-free) plus log10 total power (carries the gain).
    (b) multitaper (DPSS, NW = 2, 3 tapers) over the TRAILING ctx_epochs epochs (64 s; fewer at
        the start of the recording): log10 relative power in 0.125-0.25, 0.25-0.5 and 0.5-1 Hz
        (sub-1-Hz bands that Publication 2 removed), relative to 0.5-22.5 Hz of the same
        window. Resolution 2W = 2*NW/64 s = 0.0625 Hz, so the narrowest band holds two
        resolution cells. (An 8-s multitaper, 2W = 0.5 Hz, cannot resolve these bands.)
    (c) LF/HF = log10 P[0.1-1] / P[1-10] from the same trailing multitaper window.
    Each epoch / window is linearly detrended first. Only past and current samples are used.
    """
    from scipy.signal import detrend

    x = detrend(sig2, axis=-1, type="linear")
    nper = _ns(2.0, fs)
    f, P = welch(x, fs=fs, window="hann", nperseg=nper, noverlap=nper // 2, axis=-1, detrend=False)
    df = f[1] - f[0]
    edges = np.arange(0.5, 23.0, 1.0)
    bands = np.stack([P[..., (f >= lo) & (f < lo + 1)].sum(-1) * df for lo in edges[:-1]], -1)  # (n, S, 22)
    tot = bands.sum(-1, keepdims=True) + 1e-12
    a_rel = np.log10(bands / tot + 1e-12)
    a_tot = np.log10(tot)

    n, S, _ = sig2.shape
    b_rel = np.full((n, S, 4), np.nan)

    def lf_block(w):
        w = detrend(w, axis=-1, type="linear")
        fm, Pm = _dpss_psd(w, fs)
        dfm = fm[1] - fm[0]
        bm = lambda lo, hi: Pm[..., (fm >= lo) & (fm < hi)].sum(-1) * dfm + 1e-12  # noqa: E731
        totm = bm(0.5, 22.5)
        return np.stack([np.log10(bm(0.125, 0.25) / totm), np.log10(bm(0.25, 0.5) / totm),
                         np.log10(bm(0.5, 1.0) / totm), np.log10(bm(0.1, 1.0) / bm(1.0, 10.0))], -1)

    k = max(1, int(ctx_epochs))
    for e in range(min(k - 1, n)):                                    # short windows at the start
        b_rel[e] = lf_block(_trailing_windows(sig2, e, e + 1, e + 1))
    for e0 in range(k - 1, n, 64):                                    # full windows, in chunks
        e1 = min(n, e0 + 64)
        b_rel[e0:e1] = lf_block(_trailing_windows(sig2, e0, e1, k))
    feats = np.concatenate([a_rel, a_tot, b_rel], -1)               # (n, S, 27)
    per = ([f"F13_logrel_{lo:.1f}-{lo + 1:.1f}" for lo in edges[:-1]] + ["F13_logtotal_0.5-22.5",
           "F13_mt_logrel_0.125-0.25", "F13_mt_logrel_0.25-0.5", "F13_mt_logrel_0.5-1", "F13_lfhf"])
    names = [f"{p.split('_', 1)[0]}_{s}_{p.split('_', 1)[1]}" for s in snames for p in per]
    return feats.reshape(feats.shape[0], -1), names


DE_BANDS = [(1, 2)] + [(lo, lo + 2) for lo in range(2, 50, 2)]    # 25 bands, 1-50 Hz


def de_2hz_features(sig: np.ndarray, snames: list[str], fs: int = FS) -> tuple[np.ndarray, list[str]]:
    """F14 Gaussian differential entropy in 25 bands of 2 Hz from 1 to 50 Hz (SEED-VIG standard).

    DE_k = 0.5 ln(2 pi e P_k) with P_k = Welch band power (Hann, 1-s segments, 50 % overlap,
    linear detrend per segment) (Shi, Jiao & Lu 2013; Duan 2013; Zheng & Lu 2017).
    """
    f, P = welch(sig, fs=fs, window="hann", nperseg=fs, noverlap=fs // 2, axis=-1, detrend="linear")
    df = f[1] - f[0]
    Pk = np.stack([P[..., (f >= lo) & (f < hi)].sum(-1) * df for lo, hi in DE_BANDS], -1) + 1e-12
    de = 0.5 * np.log(2 * np.pi * np.e * Pk)
    names = [f"F14_{s}_DE_{lo}-{hi}Hz" for s in snames for lo, hi in DE_BANDS]
    return de.reshape(de.shape[0], -1), names


# =============================================================================== nonlinear (F15)
def sample_entropy(x: np.ndarray, m: int = 2, r: float = 0.2) -> float:
    """Sample entropy (Richman & Moorman 2000, doi:10.1152/ajpheart.2000.278.6.H2039), tolerance r*SD (scale-free), Chebyshev distance.
    Returns NaN when no (m+1)-matches exist (impute on training persons)."""
    x = np.asarray(x, float)
    N = len(x)
    tol = r * x.std()
    D = np.abs(x[:, None] - x[None, :]) <= tol
    Mm = D[:N - m, :N - m].copy()
    for k in range(1, m):
        Mm &= D[k:N - m + k, k:N - m + k]
    Mm1 = Mm & D[m:N, m:N]
    B = (Mm.sum() - (N - m)) / 2.0
    A = (Mm1.sum() - (N - m)) / 2.0
    return float(-np.log(A / B)) if A > 0 and B > 0 else np.nan


def hjorth(x: np.ndarray) -> tuple[float, float]:
    """Hjorth mobility and complexity."""
    d1 = np.diff(x)
    d2 = np.diff(d1)
    mob = np.sqrt(d1.var() / (x.var() + 1e-30))
    mob1 = np.sqrt(d2.var() / (d1.var() + 1e-30))
    return float(mob), float(mob1 / (mob + 1e-30))


def nonlinear_features(sig2: np.ndarray, snames: list[str]) -> tuple[np.ndarray, list[str]]:
    """F15 (Zhang, Wang & Fu 2014; Wang 2019): SampEn(m=2, r=0.2 SD), Hjorth mobility and
    complexity of each epoch of the 0.5-20 Hz band-passed signals (filtered continuously)."""
    n, S, _ = sig2.shape
    X = np.full((n, S, 3), np.nan)
    for i in range(n):
        for s in range(S):
            X[i, s, 0] = sample_entropy(sig2[i, s])
            X[i, s, 1:] = hjorth(sig2[i, s])
    names = [f"F15_{s}_{p}" for s in snames for p in ("sampen", "hjorth_mob", "hjorth_comp")]
    return X.reshape(n, -1), names


# =============================================================================== F17 temporal context
def _segments_by_rec(rec: np.ndarray):
    rec = np.asarray(rec)
    b = np.flatnonzero(np.diff(rec) != 0) + 1
    return list(zip(np.r_[0, b], np.r_[b, len(rec)]))


def causal_moving_average(X: np.ndarray, rec: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over the current and previous w-1 epochs, within each recording
    (NaN-aware). Causal: output at t uses epochs <= t only (Zheng & Lu 2017 used a moving
    average; here it is made causal). Rows must be in time order within each recording."""
    X = np.asarray(X, float)
    out = np.empty_like(X)
    for a, b in _segments_by_rec(rec):
        Z = X[a:b]
        ok = np.isfinite(Z)
        cs = np.cumsum(np.where(ok, Z, 0.0), 0)
        cn = np.cumsum(ok, 0)
        cs = np.vstack([np.zeros((1, Z.shape[1])), cs])
        cn = np.vstack([np.zeros((1, Z.shape[1])), cn])
        idx = np.arange(1, b - a + 1)
        lo = np.maximum(0, idx - w)
        s, c = cs[idx] - cs[lo], cn[idx] - cn[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[a:b] = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out


def trailing_context(X: np.ndarray, rec: np.ndarray, names: list[str] | None = None):
    """F17 causal context features for each column: trailing mean and SD over 2 min (15 epochs),
    trailing mean over 5 min (37 epochs), least-squares slope over the last 5 min (per epoch)
    and delta = x_t - trailing 5-min mean. Windows reset at recording boundaries."""
    X = np.asarray(X, float)
    m15 = causal_moving_average(X, rec, 15)
    m37 = causal_moving_average(X, rec, 37)
    sd15 = np.sqrt(np.maximum(causal_moving_average(X ** 2, rec, 15) - m15 ** 2, 0))
    slope = np.full_like(X, np.nan)
    for a, b in _segments_by_rec(rec):
        Z = X[a:b]
        for t in range(b - a):
            lo = max(0, t - 36)
            W = Z[lo:t + 1]
            if len(W) < 3:
                continue
            tt = np.arange(len(W), dtype=float)
            tt -= tt.mean()
            Wc = np.where(np.isfinite(W), W - np.nanmean(W, 0), 0.0)
            slope[a + t] = (tt @ Wc) / (tt @ tt)
    out = np.hstack([m15, sd15, m37, slope, X - m37])
    if names is None:
        return out
    nn = [f"{n}|{s}" for s in ("tmean2m", "tsd2m", "tmean5m", "slope5m", "delta5m") for n in names]
    return out, nn


def kalman_filter_rw(X: np.ndarray, rec: np.ndarray, q_over_r: float = 0.01) -> np.ndarray:
    """CAUSAL forward Kalman filter of a random-walk LDS per feature (z_t = z_{t-1} + w,
    x_t = z_t + v), run separately inside each recording. Shi & Lu (2010, EMBC,
    doi:10.1109/IEMBS.2010.5627125) introduced LDS smoothing for vigilance; this is its
    forward-only (deployable) part. Only the ratio q/r matters (scale-free); with
    q/r = 0.01 the steady-state gain is ~0.095 (time constant ~10 epochs = 80 s).
    NaN observations skip the update step."""
    X = np.asarray(X, float)
    out = np.full_like(X, np.nan)
    q, r = q_over_r, 1.0
    for a, b in _segments_by_rec(rec):
        z = np.full(X.shape[1], np.nan)
        P = np.full(X.shape[1], np.nan)
        for t in range(a, b):
            x = X[t]
            ok = np.isfinite(x)
            init = ok & ~np.isfinite(z)
            z[init], P[init] = x[init], r
            upd = ok & ~init
            Pp = P + q
            K = Pp / (Pp + r)
            z = np.where(upd, z + K * (x - z), z)
            P = np.where(upd, (1 - K) * Pp, np.where(np.isfinite(P) & ~init, Pp, P))
            out[t] = z
    return out


def rts_smoother_rw(X: np.ndarray, rec: np.ndarray, q_over_r: float = 0.01) -> np.ndarray:
    """NON-CAUSAL Rauch-Tung-Striebel smoother of the same random-walk LDS (uses FUTURE
    epochs of the recording). This is the 'off-line LDS' of the SEED-VIG literature (Shi &
    Lu 2010; Zheng & Lu 2017). Use only as a labelled comparison, never as a deployable
    result."""
    X = np.asarray(X, float)
    out = np.full_like(X, np.nan)
    q, r = q_over_r, 1.0
    for a, b in _segments_by_rec(rec):
        n = b - a
        Z = np.full((n, X.shape[1]), np.nan)
        Pf = np.full((n, X.shape[1]), np.nan)
        z = np.full(X.shape[1], np.nan)
        P = np.full(X.shape[1], np.nan)
        for t in range(n):
            x = X[a + t]
            ok = np.isfinite(x)
            init = ok & ~np.isfinite(z)
            z[init], P[init] = x[init], r
            upd = ok & ~init
            Pp = P + q
            K = Pp / (Pp + r)
            z = np.where(upd, z + K * (x - z), z)
            P = np.where(upd, (1 - K) * Pp, np.where(np.isfinite(P) & ~init, Pp, P))
            Z[t], Pf[t] = z, P
        S = Z.copy()
        for t in range(n - 2, -1, -1):
            Pp = Pf[t] + q
            G = Pf[t] / Pp
            ok = np.isfinite(S[t + 1]) & np.isfinite(Z[t])
            S[t] = np.where(ok, Z[t] + G * (S[t + 1] - Z[t]), Z[t])
        out[a:b] = S
    return out


# =============================================================================== driver
DEFAULT_EXCLUDE_BLOCKS = ("F05",)   # explicit eye-closure block (PERCLOS proxy): report separately
BLINK_BLOCKS = ("F01", "F02", "F03", "F04")   # blink duration / closing / AVR / rate blocks


def extract_recording(eog: np.ndarray, montage: str = "trad", causal: bool = False) -> dict:
    """All literature features of one recording (n_ep, 7, 1000) -> dict(X, names, block, info).

    causal=False: offline zero-phase variant (cache lit_<montage>_*; look-ahead about
    LOOKAHEAD_EPOCHS['offline'] epochs). causal=True: forward-only filters and offset-assigned
    fixations (cache litc_<montage>_*; look-ahead LOOKAHEAD_EPOCHS['causal'] = 1 epoch).
    `block` holds the F-code of each column; F05 (explicit eye-closure block) is included in X
    but dropped from the default set (DEFAULT_EXCLUDE_BLOCKS). Note that F01-F04 (blink
    duration, closed time, rate) also re-estimate the blink term of the PERCLOS label, so the
    default set is "without the explicit eye-closure block", not "PERCLOS-free".
    """
    if eog.shape[-1] != N_SAMP:
        raise ValueError(f"expected {N_SAMP} samples per epoch (8 s at {FS} Hz), got {eog.shape[-1]}")
    FALLBACKS.clear()
    n_ep = eog.shape[0]
    sig = montage_signals(eog, montage)
    H, V = sig["H"], sig["V"]
    N = len(H)
    bl = detect_blinks(V, H, causal=causal)
    bmask50 = blink_mask(bl, N, pad=_ns(0.05))
    bmask100 = blink_mask(bl, N, pad=_ns(0.1))
    Xb, nb = blink_epoch_features(bl, n_ep)
    Xc, nc = closure_features(V, bl, n_ep, causal=causal)
    sh = detect_saccades(H, bmask50, causal=causal)
    sv = detect_saccades(V, bmask50, causal=causal)
    Xs, ns = saccade_epoch_features(sh, sv, n_ep)
    ev_mask = bmask50.copy()
    for s in (sh, sv):
        for a, b in zip(s["on"].astype(int), s["off"].astype(int)):
            ev_mask[a:b + 1] = True
    Hs, Vs = filt(H, None, 10.0, causal=causal), filt(V, None, 10.0, causal=causal)
    fx = detect_fixations(Hs, Vs, ev_mask, lag=int(causal))
    Xf, nf = fixation_epoch_features(fx, n_ep, assign="offset" if causal else "onset")
    Xsem, nsem = sem_features(H, V, bmask100, n_ep, causal=causal)
    bcwt = cwt_events(bl["Vf"], "blink", lag=int(causal))
    scwt = cwt_events(filt(H, 0.05, 20.0, causal=causal), "saccade", lag=int(causal))
    Xz, nz = zheng_lu_36(bcwt, scwt, n_ep)
    HV = np.stack([H, V]).reshape(2, n_ep, N_SAMP).transpose(1, 0, 2)
    Xsp, nsp = spectral_features(HV, ["H", "V"])
    E = sig["elec"].reshape(sig["elec"].shape[0], n_ep, N_SAMP).transpose(1, 0, 2)
    Xde, nde = de_2hz_features(E, sig["elec_names"])
    HVb = filt(np.stack([H, V]), 0.5, 20.0, causal=causal).reshape(2, n_ep, N_SAMP).transpose(1, 0, 2)
    Xnl, nnl = nonlinear_features(HVb, ["H", "V"])
    parts = [(Xb, nb), (Xc, nc), (Xs, ns), (Xf, nf), (Xsem, nsem), (Xz, nz), (Xsp, nsp), (Xde, nde), (Xnl, nnl)]
    X = np.hstack([p[0] for p in parts])
    names = [n for p in parts for n in p[1]]
    block = [n.split("_", 1)[0] for n in names]
    info = {"variant": "causal" if causal else "offline",
            "n_blinks": len(bl["A"]), "blink_rate_per_min": len(bl["A"]) / (N / FS / 60),
            "blink_D50_median_s": float(np.median(bl["D50"])) if len(bl["D50"]) else np.nan,
            "blink_Dtot_median_s": float(np.median(bl["Dtot"])) if len(bl["Dtot"]) else np.nan,
            "n_sacc_H": len(sh["amp"]), "n_sacc_V": len(sv["amp"]), "n_fix": len(fx["dur"]),
            "n_cwt_blinks": len(bcwt["t"]), "n_cwt_sacc": len(scwt["t"]),
            "fallback_epochs": sum(FALLBACKS.values()),
            "fallbacks": ";".join(f"{k}:{v}" for k, v in sorted(FALLBACKS.items())) or "none"}
    return {"X": X, "names": names, "block": block, "info": info}
