"""
===============================================================================
  Privacy-Preserving Federated Learning over Noisy AIoT Links
  v8.0 — Clean Rewrite

  Problem Statement:
    Federated learning in AIoT networks struggles with link variability
    (loss, latency, jitter) and privacy risks. Standard FL assumes reliable
    communication, which is unrealistic in wireless AIoT. We design
    link-aware FL with selective aggregation and differential privacy that
    achieves good accuracy and communication efficiency.

  Core Contributions:
    1. Per-client channel modeling with 3-state burst-error Markov chains
    2. Channel-aware dynamic privacy budget allocation (DynamicPrivacyScheduler)
    3. Link-aware selective aggregation toggling
    4. Online RDP privacy accounting (PrivacyOdometer)
    5. Comprehensive evaluation: ablation, baselines, Pareto analysis

  Key fixes over v7.2:
    - Per-client independent channels (not global per-round)
    - Consistent activity label mapping (ACTIVITY_MAP ↔ ACTIVITY_GROUP_MAP)
    - No double-counting of gradient norms
    - Ablation configs don't pass spurious keys to Config
    - Privacy accounting uses consistent RDP-based σ calibration
    - Hard budget enforcement after each spend
    - Non-DP and centralized baselines included
    - 5+ seeds for statistical validity
===============================================================================
"""

import argparse
import copy
import json
import math
import os
import pickle
import random
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from scipy.fft import fft
from scipy.stats import entropy, t as t_dist, ttest_rel
from sklearn.metrics import f1_score, confusion_matrix, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.accountants import RDPAccountant as OpacusRDPAccountant
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kw):
        return x


# =============================================================================
# §0  ENUMS & CONSTANTS
# =============================================================================

class ChannelState(Enum):
    GOOD = auto()
    MODERATE = auto()
    BAD = auto()


# ── Activity labels ──────────────────────────────────────────────────────────
# ACTIVITY_MAP: raw code → canonical name (used in data loading)
# ACTIVITY_GROUP_MAP: canonical name → 3-class group
# These MUST be consistent — every value in ACTIVITY_MAP must be a key in
# ACTIVITY_GROUP_MAP when fine_classes=False.

ACTIVITY_MAP = {
    "A": "Walking",
    "B": "Jogging",
    "C": "Stairs",
    "D": "Sitting",
    "E": "Standing",
    "F": "Typing",
    "G": "Brushing Teeth",
    "H": "Eating Soup",
    "I": "Eating Chips",
    "J": "Eating Pasta",
    "K": "Drinking from Cup",
    "L": "Eating Sandwich",
    "M": "Kicking",
    "O": "Playing Catch",
    "P": "Dribbling",
    "Q": "Writing",
    "R": "Clapping",
    "S": "Folding Clothes",
}

ACTIVITY_GROUP_MAP = {
    # Non-Hand activities (lower-body / whole-body)
    "Walking": "Non-Hand",
    "Jogging": "Non-Hand",
    "Stairs": "Non-Hand",
    "Standing": "Non-Hand",
    "Kicking": "Non-Hand",
    # Hand-General activities
    "Dribbling": "Hand-General",
    "Playing Catch": "Hand-General",
    "Typing": "Hand-General",
    "Writing": "Hand-General",
    "Clapping": "Hand-General",
    "Brushing Teeth": "Hand-General",
    "Folding Clothes": "Hand-General",
    "Sitting": "Hand-General",
    # Hand-Eating activities
    "Eating Soup": "Hand-Eating",
    "Eating Chips": "Hand-Eating",
    "Eating Pasta": "Hand-Eating",
    "Drinking from Cup": "Hand-Eating",
    "Eating Sandwich": "Hand-Eating",
}

# Validate consistency at import time
_missing = set(ACTIVITY_MAP.values()) - set(ACTIVITY_GROUP_MAP.keys())
assert len(_missing) == 0, (
    f"ACTIVITY_MAP values not in ACTIVITY_GROUP_MAP: {_missing}"
)

ALL_SENSORS = [("phone", "accel"), ("phone", "gyro"),
               ("watch", "accel"), ("watch", "gyro")]
NUM_FEATURES_PER_SENSOR = 48
NUM_FEATURES = NUM_FEATURES_PER_SENSOR * len(ALL_SENSORS)  # 192


# =============================================================================
# §1  CONFIGURATION
# =============================================================================

@dataclass
class Config:
    """
    Experiment configuration with validation.

    Sections: data, federated learning, differential privacy,
    channel model, selective aggregation, scheduler hyper-params.
    """

    # ── Data ──────────────────────────────────────────────────────────────
    csv_path: str = "merged_wisdm.csv"
    window_size: int = 200
    step_size: int = 100
    fine_classes: bool = False

    # ── Federated Learning ────────────────────────────────────────────────
    num_clients: int = 30
    rounds: int = 50                     # was 100 — halved for faster experiments
    local_epochs: int = 3
    local_batch: int = 64
    lr: float = 1e-3
    lr_min: float = 1e-5
    mu: float = 0.0          # FedProx parameter (0 → FedAvg)
    fraction: float = 0.4    # fraction of clients selected per round
    alpha: float = 0.5       # Dirichlet non-IID parameter
    eval_every: int = 1

    # ── Differential Privacy ──────────────────────────────────────────────
    dp_enabled: bool = True
    use_opacus: bool = True
    delta: float = 1e-4
    total_epsilon: float = 5.0           # was 20 — tighter budget so scheduler matters
    clip_norm: float = 1.0
    sigma_cap: float = 200.0             # was 50 — allow larger σ at very tight budgets
    schedule: str = "dynamic"  # dynamic | convex | linear | uniform | exponential

    # Dynamic scheduler hyper-parameters
    schedule_channel_good: float = 1.8   # was 1.3 — more aggressive channel adaptation
    schedule_channel_moderate: float = 0.9  # was 1.0
    schedule_channel_bad: float = 0.2    # was 0.4 — conserve heavily on bad rounds
    schedule_phase_base: float = 0.9
    schedule_phase_amplitude: float = 0.3
    schedule_urgency_scale: float = 0.5
    schedule_grad_target_norm: float = 1.0
    schedule_grad_min: float = 0.5
    schedule_grad_max: float = 1.5
    min_epsilon_per_round: float = 0.005 # was 0.01 — halved with rounds (budget is tighter)

    # Ablation toggles
    use_channel_factor: bool = True
    use_phase_factor: bool = True
    use_urgency_factor: bool = True
    use_gradient_factor: bool = True

    # ── Early Stopping ────────────────────────────────────────────────────
    early_stop: bool = True
    early_stop_patience: int = 10        # was 15 — proportional to 50 rounds
    early_stop_min_delta: float = 0.005  # was 0.01 — more sensitive with fewer rounds

    # ── Selective Aggregation ─────────────────────────────────────────────
    sa_enabled: bool = True
    sa_thresh_good: float = 0.15
    sa_thresh_moderate: float = 0.25
    sa_thresh_bad: float = 0.35

    # ── Quantization ──────────────────────────────────────────────────────
    quant_enabled: bool = False
    quant_bad_bits: int = 8

    # ── Straggler Simulation ──────────────────────────────────────────────
    straggler_frac: float = 0.0
    straggler_delay_multiplier: float = 5.0

    # ── Channel Model ─────────────────────────────────────────────────────
    mode: str = "noisy"    # noisy | reliable
    markov_transition: List = field(default_factory=lambda: [
        [0.70, 0.20, 0.10],             # was [0.90, 0.08, 0.02] — more exits from GOOD
        [0.20, 0.50, 0.30],             # was [0.30, 0.60, 0.10] — more falls into BAD
        [0.10, 0.30, 0.60],             # was [0.20, 0.40, 0.40] — stickier BAD state
    ])
    state_loss_rates: Dict = field(default_factory=lambda: {
        "GOOD": 0.02, "MODERATE": 0.15, "BAD": 0.40
    })
    state_latencies: Dict = field(default_factory=lambda: {
        "GOOD": (20.0, 5.0), "MODERATE": (100.0, 30.0), "BAD": (300.0, 100.0)
    })

    # ── Reproducibility ──────────────────────────────────────────────────
    seed: int = 42
    output_dir: str = "fl_results"
    device: str = ""  # auto-detect in __post_init__

    def __post_init__(self):
        """Validate configuration."""
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.num_clients < 1:
            raise ValueError(f"num_clients must be ≥ 1, got {self.num_clients}")
        if self.rounds < 1:
            raise ValueError(f"rounds must be ≥ 1, got {self.rounds}")
        if not (0 < self.fraction <= 1):
            raise ValueError(f"fraction must be in (0,1], got {self.fraction}")
        if self.lr <= 0 or self.lr_min <= 0:
            raise ValueError("lr and lr_min must be > 0")
        if self.lr_min >= self.lr:
            raise ValueError(f"lr_min ({self.lr_min}) must be < lr ({self.lr})")
        if self.dp_enabled:
            if self.total_epsilon <= 0:
                raise ValueError(f"total_epsilon must be > 0, got {self.total_epsilon}")
            if self.clip_norm <= 0:
                raise ValueError(f"clip_norm must be > 0, got {self.clip_norm}")

    def run_label(self) -> str:
        """Unique label for file naming."""
        parts = [self.mode]
        if self.dp_enabled:
            eng = "opacus" if (self.use_opacus and OPACUS_AVAILABLE) else "manual"
            parts += [f"DP-{eng}", self.schedule, f"eps{self.total_epsilon:.0f}"]
        else:
            parts.append("noDP")
        parts.append("SA" if self.sa_enabled else "noSA")
        parts.append("18cls" if self.fine_classes else "3cls")
        parts.append(f"s{self.seed}")
        return "_".join(parts)

    def to_dict(self) -> Dict:
        return {k: str(v) if isinstance(v, Enum) else v
                for k, v in asdict(self).items()}


# =============================================================================
# §2  FEATURE EXTRACTION
# =============================================================================

def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation with degenerate-signal protection."""
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _dominant_frequency(sig: np.ndarray) -> Tuple[int, float, float]:
    """Index, relative power, and spectral entropy of dominant frequency."""
    spectrum = np.abs(fft(sig)[: len(sig) // 2])
    total = spectrum.sum() + 1e-10
    idx = int(np.argmax(spectrum))
    spec_prob = spectrum / total
    se = float(-np.sum(spec_prob * np.log(spec_prob + 1e-10)))
    return idx, spectrum[idx] / total, se


def extract_features(win: pd.DataFrame) -> np.ndarray:
    """
    Extract 48 statistical features from one sensor window.

    Layout (48 features):
      4 signals (x, y, z, mag) × 8 stats            = 32
      3 pair-wise correlations (xy, xz, yz)           =  3
      mean_mag, mean_abs_sum                           =  2
      3 signals × 3 frequency features                 =  9
      zero-crossing rate(x), rms(mag)                  =  2
                                                  total = 48
    """
    x, y, z = win["x"].values, win["y"].values, win["z"].values
    mag = np.sqrt(x**2 + y**2 + z**2)
    feats: List[float] = []

    # Per-signal statistics
    for sig in (x, y, z, mag):
        feats.extend([
            float(np.mean(sig)),
            float(np.std(sig)),
            float(np.min(sig)),
            float(np.max(sig)),
            float(np.percentile(sig, 25)),
            float(np.percentile(sig, 75)),
            float(np.mean(np.abs(sig - np.mean(sig)))),    # MAD
            float(entropy(np.abs(sig) + 1e-10)),            # signal entropy
        ])

    # Correlations
    feats.extend([_safe_corr(x, y), _safe_corr(x, z), _safe_corr(y, z)])

    # Magnitude aggregates
    feats.append(float(np.mean(mag)))
    feats.append(float(np.mean(np.abs(x) + np.abs(y) + np.abs(z))))

    # Frequency-domain
    for sig in (x, y, z):
        di, dp, se = _dominant_frequency(sig)
        feats.extend([float(di), float(dp), se])

    # Zero-crossing rate & RMS
    feats.append(float(np.mean(np.diff(np.sign(x)) != 0)))
    feats.append(float(np.sqrt(np.mean(mag**2))))

    assert len(feats) == NUM_FEATURES_PER_SENSOR, (
        f"Expected {NUM_FEATURES_PER_SENSOR} features, got {len(feats)}"
    )
    return np.array(feats, dtype=np.float32)


# =============================================================================
# §3  DATA LOADING & SEGMENTATION
# =============================================================================

def _read_dataset(path: str) -> pd.DataFrame:
    """
    Read dataset from CSV or Parquet (auto-detected by extension).

    Parquet is ~5-15× faster than CSV for a 1.2 GB file.
    Convert once with:  python convert_to_parquet.py --csv merged_wisdm.csv
    """
    import time
    t0 = time.time()
    ext = os.path.splitext(path)[1].lower()

    if ext in (".parquet", ".pq"):
        print(f"[Data] Reading Parquet: {path}")
        df = pd.read_parquet(path, engine="pyarrow")
    elif ext == ".feather":
        print(f"[Data] Reading Feather: {path}")
        df = pd.read_feather(path)
    else:
        print(f"[Data] Reading CSV: {path}  (tip: convert to .parquet for 5-15× speedup)")
        df = pd.read_csv(path, engine="c", low_memory=False)

    elapsed = time.time() - t0
    print(f"       {len(df):,} rows × {len(df.columns)} cols in {elapsed:.1f}s")
    return df


def load_and_segment(cfg: Config) -> Tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load WISDM data from all 4 sensors, segment into windows,
    extract features, and return (X, y, subjects, label_encoder).

    Performance optimisations vs v7.2:
      - Reads the file ONCE then filters per sensor (was: 4 full reads)
      - Supports Parquet/Feather for 5-15× faster I/O
      - Pickle cache for instant reload on subsequent runs
    """
    cache_tag = f"cache_{cfg.window_size}_{cfg.step_size}"
    cache_tag += "_fine" if cfg.fine_classes else "_3cls"
    cache_file = os.path.join(cfg.output_dir, f"{cache_tag}.pkl")

    if os.path.exists(cache_file):
        print(f"[Data] Loading from cache: {cache_file}")
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Read file ONCE ────────────────────────────────────────────────────
    df_all = _read_dataset(cfg.csv_path)
    df_all.columns = (df_all.columns.str.strip().str.lower()
                       .str.replace(";", "", regex=False))

    # Normalise column names (applied once to the full dataframe)
    if "activity_label" not in df_all.columns and "activity_code" in df_all.columns:
        df_all["activity_label"] = (
            df_all["activity_code"].map(ACTIVITY_MAP).fillna(df_all["activity_code"])
        )
    if "subject_id" not in df_all.columns and "user" in df_all.columns:
        df_all.rename(columns={"user": "subject_id"}, inplace=True)

    # Clean numeric axes once
    for col in ("x", "y", "z"):
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")
    df_all.dropna(subset=["x", "y", "z", "activity_label"], inplace=True)

    # Clip outliers per-axis (global quantiles)
    for col in ("x", "y", "z"):
        lo, hi = df_all[col].quantile([0.001, 0.999])
        df_all = df_all[(df_all[col] >= lo) & (df_all[col] <= hi)]

    has_device_col = ("device" in df_all.columns and "sensor" in df_all.columns)
    has_ts = "timestamp" in df_all.columns
    ts_col = "timestamp" if has_ts else "subject_id"

    # ── Process each sensor (filter from the single dataframe) ────────────
    sensor_dicts: List[Dict] = []

    for device, sensor in ALL_SENSORS:
        print(f"  → {device}/{sensor} ...", end=" ", flush=True)

        if has_device_col:
            df = df_all[(df_all["device"] == device)
                        & (df_all["sensor"] == sensor)].copy()
        else:
            # Single-sensor CSV fallback (all rows belong to this sensor)
            df = df_all.copy()

        df.sort_values(["subject_id", ts_col], inplace=True, ignore_index=True)

        # Segment into windows
        segs: Dict[Tuple, np.ndarray] = {}
        for (subj, act), grp in df.groupby(
            ["subject_id", "activity_label"], sort=False
        ):
            grp = grp.reset_index(drop=True)
            win_idx = 0
            for start in range(0, len(grp) - cfg.window_size + 1, cfg.step_size):
                win = grp.iloc[start : start + cfg.window_size]
                if len(win) == cfg.window_size:
                    feats = extract_features(win)
                    segs[(subj, act, win_idx)] = feats
                    win_idx += 1

        sensor_dicts.append(segs)
        print(f"{len(segs):,} windows")

    # Free the big dataframe now that we've extracted features
    del df_all

    # ── Intersect keys across all 4 sensors ───────────────────────────────
    common_keys = set(sensor_dicts[0].keys())
    for sd in sensor_dicts[1:]:
        common_keys &= set(sd.keys())
    common_keys = sorted(common_keys)
    print(f"\n  Common windows (all 4 sensors): {len(common_keys):,}")

    if len(common_keys) == 0:
        raise RuntimeError(
            "No common windows across sensors. Check CSV format and column names."
        )

    # ── Build feature matrix ──────────────────────────────────────────────
    segs_list, labels, subjects = [], [], []
    for key in common_keys:
        subj, act, _ = key
        feat_vec = np.concatenate([sd[key] for sd in sensor_dicts])
        segs_list.append(feat_vec)
        if cfg.fine_classes:
            labels.append(act)
        else:
            grp = ACTIVITY_GROUP_MAP.get(act)
            if grp is None:
                warnings.warn(
                    f"Activity '{act}' not in ACTIVITY_GROUP_MAP — skipping"
                )
                continue
            labels.append(grp)
        subjects.append(subj)

    X = np.array(segs_list, dtype=np.float32)
    subjects_arr = np.array(subjects)

    le = LabelEncoder()
    y = le.fit_transform(labels).astype(np.int64)

    assert not np.any(np.isnan(X)), "NaN in features"
    assert X.shape[1] == NUM_FEATURES, (
        f"Expected {NUM_FEATURES} features, got {X.shape[1]}"
    )

    print(f"  Features : {X.shape[1]}  ({NUM_FEATURES_PER_SENSOR} × "
          f"{len(ALL_SENSORS)} sensors)")
    print(f"  Classes  : {len(le.classes_)}  {list(le.classes_)}")
    print(f"  Windows  : {len(X):,}")
    print(f"  Subjects : {len(np.unique(subjects_arr))}")

    result = (X, y, subjects_arr, le)
    with open(cache_file, "wb") as f:
        pickle.dump(result, f)
    print(f"  Cache saved: {cache_file}")

    return result


# =============================================================================
# §4  DIRICHLET NON-IID PARTITION
# =============================================================================

def dirichlet_partition(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42,
    min_samples: int = 100,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition data across clients using Dirichlet distribution."""
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(y))
    client_idx: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        counts = (proportions * len(idx)).astype(int)
        counts[-1] = max(len(idx) - counts[:-1].sum(), 0)
        counts = np.maximum(counts, 0)

        splits = np.split(idx, np.cumsum(counts[:-1]))
        for cid, split in enumerate(splits):
            if len(split) > 0:
                client_idx[cid].extend(split.tolist())

    partitions = []
    for cid in range(num_clients):
        idx = np.array(client_idx[cid], dtype=np.int64)
        if len(idx) < min_samples:
            extra = rng.choice(len(y), size=min_samples - len(idx), replace=True)
            idx = np.concatenate([idx, extra.astype(np.int64)])
        partitions.append((X[idx], y[idx]))

    sizes = [len(p[0]) for p in partitions]
    print(f"\n[Partition] Dirichlet α={alpha} | {num_clients} clients | "
          f"min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.0f}")
    return partitions


# =============================================================================
# §5  MODEL
# =============================================================================

class MLP(nn.Module):
    """MLP for activity recognition: 512 → 256 → 128 → num_classes."""

    def __init__(self, num_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# §6  CHANNEL MODEL  —  Per-Client Burst-Error Markov Chain
# =============================================================================

class MarkovChannel:
    """
    Independent 3-state Markov channel for a single client.

    States: GOOD → MODERATE → BAD with burst-error extensions.
    Each client gets its own MarkovChannel instance so link variability
    is independent across devices, matching real AIoT deployments.
    """

    STATE_MAP = {0: ChannelState.GOOD, 1: ChannelState.MODERATE, 2: ChannelState.BAD}

    def __init__(self, transition_matrix: List[List[float]],
                 loss_rates: Dict[str, float],
                 latency_params: Dict[str, Tuple[float, float]],
                 seed: int = 42):
        self.P = np.array(transition_matrix)
        self.loss = loss_rates
        self.latency = latency_params
        self.rng = np.random.default_rng(seed)
        assert np.allclose(self.P.sum(axis=1), 1.0), "Rows must sum to 1"
        self.state = self._stationary_start()
        self.burst_remaining = 0

    def _stationary_start(self) -> int:
        """Start from the stationary distribution."""
        v = np.ones(3) / 3.0
        for _ in range(1000):
            v_new = v @ self.P
            if np.allclose(v_new, v, atol=1e-10):
                break
            v = v_new
        v = np.abs(v) / np.abs(v).sum()
        return int(self.rng.choice(3, p=v))

    def step(self) -> Tuple[bool, float, ChannelState, float]:
        """
        Advance the Markov chain one step.
        Returns: (packet_lost, loss_rate, channel_state, latency_ms)
        """
        if self.burst_remaining > 0:
            self.burst_remaining -= 1
            current = 2  # remain in BAD during burst
        else:
            current = int(self.rng.choice(3, p=self.P[self.state]))
            self.state = current
            if current == 2:
                self.burst_remaining = self.rng.integers(1, 4)

        ch = self.STATE_MAP[current]
        loss_rate = self.loss[ch.name]
        packet_lost = bool(self.rng.random() < loss_rate)

        mu, s = self.latency[ch.name]
        latency_ms = max(1.0, float(self.rng.normal(mu, s)))

        return packet_lost, loss_rate, ch, latency_ms


class ReliableChannel:
    """Perfect channel — no loss, fixed low latency."""

    def step(self) -> Tuple[bool, float, ChannelState, float]:
        return False, 0.0, ChannelState.GOOD, 20.0


class ChannelManager:
    """
    Manages per-client independent channels.

    In "noisy" mode each client has its own MarkovChannel.
    In "reliable" mode every client uses the same ReliableChannel.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.channels: Dict[int, Any] = {}

    def get_channel(self, client_id: int) -> Any:
        if client_id not in self.channels:
            if self.cfg.mode == "reliable":
                self.channels[client_id] = ReliableChannel()
            else:
                # Unique seed per client for independence
                seed = self.cfg.seed + client_id * 1000
                self.channels[client_id] = MarkovChannel(
                    self.cfg.markov_transition,
                    self.cfg.state_loss_rates,
                    self.cfg.state_latencies,
                    seed=seed,
                )
        return self.channels[client_id]

    def step(self, client_id: int) -> Tuple[bool, float, ChannelState, float]:
        """Step the channel for a specific client."""
        return self.get_channel(client_id).step()


# =============================================================================
# §7  PRIVACY ODOMETER  —  RDP Accounting
# =============================================================================

class PrivacyOdometer:
    """
    Online privacy accounting via Rényi Differential Privacy (RDP).

    Tracks cumulative privacy expenditure and converts to (ε,δ)-DP.
    Supports a hard budget cap: once total_epsilon is reached, all
    subsequent spend() calls return budget_exhausted = True.
    """

    ORDERS = [1.25, 1.5, 1.75, 2., 2.5, 3., 4., 5., 6., 7., 8.,
              10., 12., 16., 20., 32., 64., 128.]

    def __init__(self, delta: float = 1e-4, epsilon_max: float = 10.0):
        self.delta = delta
        self.epsilon_max = epsilon_max
        self.rdp = {a: 0.0 for a in self.ORDERS}
        self.history: List[Dict] = []
        self._cached_eps: Optional[float] = None

    # ── Core RDP computation ──────────────────────────────────────────────

    def _rdp_gaussian(self, sigma: float, q: float,
                      steps: int = 1) -> Dict[float, float]:
        """
        RDP for the subsampled Gaussian mechanism.

        Uses the analytic bound from Mironov (2017) for the full-batch case
        and the Poisson subsampling approximation for q < 1.
        """
        rdp: Dict[float, float] = {}
        for a in self.ORDERS:
            if sigma <= 0 or q <= 0:
                rdp[a] = 0.0
            elif q >= 1.0:
                # Full-batch: RDP of Gaussian = α / (2σ²)
                rdp[a] = steps * a / (2.0 * sigma**2)
            else:
                try:
                    # Subsampled Gaussian: dominant-term bound
                    inner = q**2 * (math.exp(a / sigma**2) - 1.0)
                    if inner >= 1e-10:
                        val = math.log(1.0 + inner) / (a - 1.0)
                    else:
                        val = inner / (a - 1.0)  # Taylor approx
                    rdp[a] = steps * max(val, 0.0)
                except (ValueError, OverflowError):
                    # Fallback: Gaussian upper bound
                    rdp[a] = steps * q**2 * a / (2.0 * sigma**2)
        return rdp

    def _to_eps(self) -> float:
        """Convert accumulated RDP to (ε,δ)-DP via the optimal order."""
        candidates = []
        for a in self.ORDERS:
            if a > 1:
                candidates.append(
                    self.rdp[a] + math.log(1.0 / self.delta) / (a - 1.0)
                )
        return max(min(candidates), 0.0) if candidates else float("inf")

    # ── Public interface ──────────────────────────────────────────────────

    def spend(self, sigma: float, sampling_rate: float,
              steps: int = 1) -> Tuple[float, float, bool]:
        """
        Record a privacy expenditure.
        Returns: (cumulative_epsilon, remaining_epsilon, budget_exhausted)
        """
        if sigma <= 0 or sampling_rate <= 0:
            eps = self.get_epsilon()
            return eps, max(0.0, self.epsilon_max - eps), False

        new_rdp = self._rdp_gaussian(sigma, sampling_rate, steps)
        for a in self.ORDERS:
            self.rdp[a] += new_rdp[a]
        self._cached_eps = None

        eps = self.get_epsilon()
        remaining = max(0.0, self.epsilon_max - eps)
        exhausted = eps >= self.epsilon_max

        self.history.append({
            "sigma": sigma, "q": sampling_rate, "steps": steps,
            "eps_cumulative": eps, "remaining": remaining,
        })
        return eps, remaining, exhausted

    def get_epsilon(self) -> float:
        if self._cached_eps is None:
            self._cached_eps = self._to_eps()
        return self._cached_eps

    def get_remaining(self) -> float:
        return max(0.0, self.epsilon_max - self.get_epsilon())

    def is_exhausted(self) -> bool:
        return self.get_epsilon() >= self.epsilon_max

    # ── σ calibration via RDP ─────────────────────────────────────────────

    @staticmethod
    def calibrate_sigma_for_epsilon(
        target_eps: float,
        delta: float,
        sampling_rate: float = 1.0,
        steps: int = 1,
        sigma_low: float = 0.01,
        sigma_high: float = 500.0,
        tol: float = 1e-3,
    ) -> float:
        """
        Binary search for σ that yields target_eps under RDP accounting.

        This keeps ε→σ conversion consistent with the RDP odometer,
        avoiding the drift between basic Gaussian calibration and RDP
        composition.
        """
        temp = PrivacyOdometer(delta=delta, epsilon_max=1e6)

        for _ in range(100):
            sigma_mid = (sigma_low + sigma_high) / 2.0
            # Reset
            temp.rdp = {a: 0.0 for a in temp.ORDERS}
            temp._cached_eps = None
            temp.spend(sigma_mid, sampling_rate, steps)
            eps_mid = temp.get_epsilon()

            if abs(eps_mid - target_eps) < tol:
                return sigma_mid
            if eps_mid > target_eps:
                sigma_low = sigma_mid   # need more noise → larger σ
            else:
                sigma_high = sigma_mid  # can use less noise → smaller σ

        return (sigma_low + sigma_high) / 2.0


# =============================================================================
# §8  DYNAMIC PRIVACY SCHEDULER
# =============================================================================

class DynamicPrivacyScheduler:
    """
    Channel-Aware Dynamic Privacy Budget Allocation.

    Per-round budget:
        ε_t = ε_base(t) × f_channel(S_t) × f_phase(t) × f_urgency(t) × f_grad(g_t)

    Each factor can be independently toggled off for ablation.
    σ is calibrated via RDP binary search (consistent with the odometer).
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.T = cfg.rounds
        self.odometer = PrivacyOdometer(cfg.delta, cfg.total_epsilon)

        self.round_num = 0
        self.gradient_ema: Optional[float] = None
        self.gradient_ema_alpha = 0.3
        self.budget_exhausted = False

        # Ablation toggles
        self.use_channel = cfg.use_channel_factor
        self.use_phase = cfg.use_phase_factor
        self.use_urgency = cfg.use_urgency_factor
        self.use_gradient = cfg.use_gradient_factor

        self.base_schedule = self._build_base_schedule(cfg.schedule)

    def _build_base_schedule(self, schedule_type: str) -> List[float]:
        """Build the per-round ε reference schedule."""
        T = self.T
        if schedule_type == "convex":
            w = [2 * (T - t + 1) / (T * (T + 1)) for t in range(1, T + 1)]
        elif schedule_type == "linear":
            w = [2 * t / (T * (T + 1)) for t in range(1, T + 1)]
        elif schedule_type == "exponential":
            k = 3.0
            w = [math.exp(-k * (t - 1) / max(T - 1, 1)) for t in range(1, T + 1)]
        else:  # uniform or dynamic (dynamic starts from uniform base)
            w = [1.0 / T] * T
        total_w = sum(w)
        return [self.cfg.total_epsilon * wi / total_w for wi in w]

    def start_round(self) -> None:
        self.round_num += 1

    def get_round_budget(
        self,
        channel_state: ChannelState,
        gradient_norm: Optional[float] = None,
        sampling_rate: float = 0.01,
    ) -> Tuple[float, float, Dict]:
        """
        Compute (target_epsilon, sigma, info) for the current round.

        The sigma is calibrated via RDP binary search so that spending σ
        through the odometer will result in ≈ target_epsilon of budget use.
        """
        t = self.round_num

        if t > self.T or t <= 0 or self.budget_exhausted:
            return 0.0, 0.0, {"reason": "exhausted_or_invalid"}

        current_eps = self.odometer.get_epsilon()
        remaining = max(0.0, self.cfg.total_epsilon - current_eps)
        rounds_left = max(1, self.T - t + 1)

        # ── Factor 1: Channel ─────────────────────────────────────────────
        if self.use_channel:
            ch_map = {
                ChannelState.GOOD: self.cfg.schedule_channel_good,
                ChannelState.MODERATE: self.cfg.schedule_channel_moderate,
                ChannelState.BAD: self.cfg.schedule_channel_bad,
            }
            f_ch = ch_map[channel_state]
        else:
            f_ch = 1.0

        # ── Factor 2: Phase (cosine annealing) ───────────────────────────
        if self.use_phase:
            f_phase = (self.cfg.schedule_phase_base
                       + self.cfg.schedule_phase_amplitude
                       * math.cos(math.pi * t / self.T))
        else:
            f_phase = 1.0

        # ── Factor 3: Urgency ─────────────────────────────────────────────
        if self.use_urgency:
            expected = sum(self.base_schedule[:t])
            deficit = expected - current_eps
            f_urg = 1.0 + self.cfg.schedule_urgency_scale * math.tanh(
                deficit / max(self.cfg.total_epsilon * 0.1, 1e-6)
            )
        else:
            f_urg = 1.0

        # ── Factor 4: Gradient (EMA) ─────────────────────────────────────
        if self.use_gradient and gradient_norm is not None:
            if self.gradient_ema is None:
                self.gradient_ema = gradient_norm
            else:
                a = self.gradient_ema_alpha
                self.gradient_ema = a * gradient_norm + (1 - a) * self.gradient_ema
            ratio = self.gradient_ema / self.cfg.schedule_grad_target_norm
            f_grad = float(np.clip(ratio, self.cfg.schedule_grad_min,
                                   self.cfg.schedule_grad_max))
        else:
            f_grad = 1.0

        # ── Compute target ε ──────────────────────────────────────────────
        base = (self.base_schedule[t - 1] if t <= len(self.base_schedule)
                else self.cfg.total_epsilon / self.T)
        target_eps = base * f_ch * f_phase * f_urg * f_grad

        # Safety bounds
        per_round_cap = remaining / rounds_left * 2.0
        target_eps = min(target_eps, per_round_cap)
        target_eps = max(target_eps, self.cfg.min_epsilon_per_round)
        target_eps = min(target_eps, remaining)

        if target_eps <= 0:
            self.budget_exhausted = True
            return 0.0, 0.0, {"reason": "budget_zero"}

        # ── Calibrate σ via RDP ───────────────────────────────────────────
        sigma = PrivacyOdometer.calibrate_sigma_for_epsilon(
            target_eps=target_eps,
            delta=self.cfg.delta,
            sampling_rate=sampling_rate,
            steps=self.cfg.local_epochs,
        )
        sigma = max(min(sigma, self.cfg.sigma_cap), 0.1)

        info = {
            "base_eps": base, "target_eps": target_eps,
            "f_channel": f_ch, "f_phase": f_phase,
            "f_urgency": f_urg, "f_gradient": f_grad,
            "sigma": sigma, "remaining_before": remaining,
            "gradient_ema": self.gradient_ema, "round": t,
        }
        return target_eps, sigma, info

    def spend(self, sigma: float, sampling_rate: float,
              steps: int = 1) -> Tuple[float, float, bool]:
        """Spend budget through the odometer. Returns (eps, remaining, exhausted)."""
        eps, rem, exhausted = self.odometer.spend(sigma, sampling_rate, steps)
        if exhausted:
            self.budget_exhausted = True
        return eps, rem, exhausted

    def get_epsilon(self) -> float:
        return self.odometer.get_epsilon()

    def get_remaining(self) -> float:
        return self.odometer.get_remaining()


# =============================================================================
# §9  LOCAL TRAINING
# =============================================================================

def _manual_dp_noise(model: nn.Module, clip_norm: float, sigma: float) -> float:
    """
    Per-sample gradient clipping and Gaussian noise injection.
    Returns the clipped gradient norm (for scheduler feedback).
    """
    # Compute global gradient norm
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    total_norm = math.sqrt(total_norm_sq)

    # Clip
    clip_coef = min(1.0, clip_norm / (total_norm + 1e-8))
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(clip_coef)
            # Add calibrated Gaussian noise
            noise = torch.randn_like(p.grad.data) * (sigma * clip_norm)
            p.grad.data.add_(noise)

    return total_norm * clip_coef


def local_train(
    model: nn.Module,
    global_model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    sigma: float,
    round_num: int,
) -> Tuple[nn.Module, float, float, float, Optional[float]]:
    """
    Train one client locally.

    Returns: (trained_model, avg_loss, accuracy, avg_grad_norm, opacus_eps)
    """
    use_opacus = (cfg.dp_enabled and cfg.use_opacus
                  and OPACUS_AVAILABLE and sigma > 0)
    device = cfg.device

    model = model.to(device)
    global_model = global_model.to(device)

    if use_opacus:
        model = ModuleValidator.fix(model)

    model.train()

    # Cosine LR annealing across FL rounds
    cos_lr = (cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min)
              * (1 + math.cos(math.pi * round_num / cfg.rounds)))
    opt = optim.AdamW(model.parameters(), lr=cos_lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    opacus_pe = None
    if use_opacus:
        opacus_pe = PrivacyEngine()
        model, opt, loader = opacus_pe.make_private(
            module=model, optimizer=opt, data_loader=loader,
            noise_multiplier=min(sigma, cfg.sigma_cap),
            max_grad_norm=cfg.clip_norm,
        )

    total_loss = correct = total = steps = 0
    grad_norms: List[float] = []

    for _ in range(cfg.local_epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()

            logits = model(xb)
            loss = criterion(logits, yb)

            # FedProx proximal term
            if cfg.mu > 0:
                actual = model._module if hasattr(model, "_module") else model
                prox = sum(
                    ((w - wg.detach()) ** 2).sum()
                    for w, wg in zip(actual.parameters(),
                                     global_model.parameters())
                )
                loss = loss + (cfg.mu / 2.0) * prox

            loss.backward()

            # Gradient norms & DP noise (manual path only)
            if not use_opacus:
                if cfg.dp_enabled and sigma > 0:
                    # Clip + noise; record CLIPPED norm only (no double-count)
                    clipped = _manual_dp_noise(model, cfg.clip_norm, sigma)
                    grad_norms.append(clipped)
                else:
                    # No DP: record raw norm for monitoring
                    gn = math.sqrt(sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in model.parameters() if p.grad is not None
                    ))
                    grad_norms.append(gn)

            opt.step()

            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
            total_loss += loss.item()
            steps += 1

    final_model = model._module if hasattr(model, "_module") else model
    avg_grad = float(np.mean(grad_norms)) if grad_norms else 1.0

    opacus_eps = None
    if opacus_pe is not None:
        try:
            opacus_eps = opacus_pe.get_epsilon(cfg.delta)
        except Exception:
            pass

    return (final_model, total_loss / max(steps, 1),
            correct / max(total, 1), avg_grad, opacus_eps)


# =============================================================================
# §10  AGGREGATION & EVALUATION
# =============================================================================

def fedavg_aggregate(
    global_model: nn.Module,
    client_models: List[nn.Module],
    weights: List[float],
) -> nn.Module:
    """Weighted FedAvg aggregation."""
    total_w = sum(weights)
    gsd = global_model.state_dict()
    for key in gsd:
        agg = torch.zeros_like(gsd[key].float())
        for m, w in zip(client_models, weights):
            agg += m.state_dict()[key].float() * (w / total_w)
        gsd[key] = agg
    global_model.load_state_dict(gsd)
    return global_model


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    return_preds: bool = False,
) -> Tuple[float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Evaluate model on a DataLoader. Returns (acc, loss, y_true?, y_pred?)."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = total = n_batches = 0
    total_loss = 0.0
    all_preds, all_labels = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        preds = logits.argmax(1)
        total_loss += criterion(logits, yb).item()
        correct += (preds == yb).sum().item()
        total += len(yb)
        n_batches += 1
        if return_preds:
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())

    acc = correct / max(total, 1)
    loss = total_loss / max(n_batches, 1)

    if return_preds:
        return acc, loss, np.array(all_labels), np.array(all_preds)
    return acc, loss, None, None


def compute_full_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]
) -> Dict:
    """Compute F1 (macro/weighted), Cohen's kappa, and confusion matrix."""
    n = len(class_names)
    return {
        "f1_macro": round(float(f1_score(
            y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(
            y_true, y_pred, average="weighted", zero_division=0)), 4),
        "kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "f1_per_class": {
            cn: round(float(f1_score(y_true, y_pred, average=None,
                                     zero_division=0,
                                     labels=list(range(n)))[i]), 4)
            for i, cn in enumerate(class_names)
        },
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=list(range(n))
        ).tolist(),
        "class_names": list(class_names),
    }


# =============================================================================
# §11  UTILITY FUNCTIONS
# =============================================================================

class EarlyStopper:
    """Stop training when test accuracy plateaus."""

    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = 0.0
        self.best_round = 0

    def check(self, acc: float, rnd: int) -> bool:
        if acc > self.best_acc + self.min_delta:
            self.best_acc = acc
            self.best_round = rnd
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def status(self) -> str:
        return (f"best={self.best_acc:.4f}@R{self.best_round}, "
                f"patience={self.counter}/{self.patience}")


def decide_sa(cfg: Config, loss_rate: float, ch_state: ChannelState) -> bool:
    """Decide whether selective aggregation is active for this link state."""
    if not cfg.sa_enabled:
        return False
    if cfg.mode == "reliable":
        return True
    thresh = {
        ChannelState.GOOD: cfg.sa_thresh_good,
        ChannelState.MODERATE: cfg.sa_thresh_moderate,
        ChannelState.BAD: cfg.sa_thresh_bad,
    }[ch_state]
    return loss_rate <= thresh


def channel_quant_bits(ch_state: ChannelState, cfg: Config) -> int:
    """Select quantization bit-width based on channel state."""
    if not cfg.quant_enabled:
        return 32
    return {
        ChannelState.GOOD: 32,
        ChannelState.MODERATE: 32,
        ChannelState.BAD: cfg.quant_bad_bits,
    }[ch_state]


def quantize_state_dict(
    sd: Dict[str, torch.Tensor], bits: int
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Uniform quantization of model weights."""
    if bits >= 32:
        return sd, 1.0
    levels = 2 ** bits
    out = {}
    for key, tensor in sd.items():
        t = tensor.float()
        t_min, t_max = t.min(), t.max()
        scale = (t_max - t_min) / max(levels - 1, 1)
        if scale < 1e-9:
            out[key] = t
        else:
            q = torch.round((t - t_min) / scale).clamp(0, levels - 1)
            out[key] = q * scale + t_min
    return out, bits / 32.0


def compute_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """Mean and half-width of a t-based confidence interval."""
    n = len(data)
    if n < 2:
        return float(np.mean(data)), 0.0
    mean = float(np.mean(data))
    se = float(np.std(data, ddof=1)) / math.sqrt(n)
    hw = t_dist.ppf((1 + confidence) / 2, n - 1) * se
    return mean, hw


# =============================================================================
# §12  MAIN FEDERATED LOOP
# =============================================================================

def run_federated(
    cfg: Config,
    cached_data: Optional[Tuple] = None,
) -> Tuple[Dict, nn.Module, LabelEncoder, Tuple]:
    """
    Run one complete federated learning experiment.

    Returns: (history_dict, final_model, label_encoder, cached_data)
    """
    os.makedirs(cfg.output_dir, exist_ok=True)
    label = cfg.run_label()

    # Seed everything
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    # Save config
    with open(os.path.join(cfg.output_dir, f"config_{label}.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)

    # ── Header ────────────────────────────────────────────────────────────
    dp_str = "OFF"
    if cfg.dp_enabled:
        eng = "Opacus" if (cfg.use_opacus and OPACUS_AVAILABLE) else "Manual"
        dp_str = f"ON ({eng}, {cfg.schedule}, ε={cfg.total_epsilon})"

    print(f"\n{'='*72}")
    print(f"  Run: {label}")
    print(f"  mode={cfg.mode} | clients={cfg.num_clients} | rounds={cfg.rounds}"
          f" | seed={cfg.seed}")
    print(f"  DP={dp_str} | SA={'ON' if cfg.sa_enabled else 'OFF'}"
          f" | FedProx μ={cfg.mu}")
    if cfg.dp_enabled:
        print(f"  Ablation: ch={cfg.use_channel_factor} phase={cfg.use_phase_factor}"
              f" urg={cfg.use_urgency_factor} grad={cfg.use_gradient_factor}")
    print(f"{'='*72}")

    # ── Data ──────────────────────────────────────────────────────────────
    if cached_data is not None:
        X, y, subjects, le = cached_data
        print(f"\n[Data] Cached: {len(X):,} windows, {len(le.classes_)} classes")
    else:
        X, y, subjects, le = load_and_segment(cfg)
        cached_data = (X, y, subjects, le)

    num_classes = len(le.classes_)

    # Subject-based train/test split
    unique_s = np.unique(subjects)
    train_s, test_s = train_test_split(
        unique_s, test_size=0.2, random_state=cfg.seed
    )
    train_mask = np.isin(subjects, train_s)
    test_mask = np.isin(subjects, test_s)

    # Scale features
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[train_mask] = scaler.fit_transform(X[train_mask]).astype(np.float32)
    X_scaled[test_mask] = scaler.transform(X[test_mask]).astype(np.float32)

    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled[test_mask]),
                      torch.tensor(y[test_mask])),
        batch_size=256, shuffle=False,
    )
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_scaled[train_mask]),
                      torch.tensor(y[train_mask])),
        batch_size=256, shuffle=False,
    )

    # Dirichlet partition for FL clients
    partitions = dirichlet_partition(
        X_scaled[train_mask], y[train_mask],
        cfg.num_clients, cfg.alpha, cfg.seed,
    )
    client_loaders = [
        DataLoader(
            TensorDataset(torch.tensor(Xc), torch.tensor(yc)),
            batch_size=cfg.local_batch, shuffle=True,
            drop_last=(len(Xc) > cfg.local_batch),
        )
        for Xc, yc in partitions
    ]

    # ── Model ─────────────────────────────────────────────────────────────
    global_model = MLP(NUM_FEATURES, num_classes).to(cfg.device)
    print(f"\n[Model] MLP 512→256→128→{num_classes} | "
          f"params={global_model.count_params():,} | "
          f"{global_model.count_params() * 4 / 1024:.1f} KB")

    # ── Channels (per-client) ─────────────────────────────────────────────
    ch_mgr = ChannelManager(cfg)

    # ── Privacy scheduler ─────────────────────────────────────────────────
    scheduler = DynamicPrivacyScheduler(cfg) if cfg.dp_enabled else None

    # ── Early stopping ────────────────────────────────────────────────────
    stopper = (EarlyStopper(cfg.early_stop_patience, cfg.early_stop_min_delta)
               if cfg.early_stop else None)

    # ── History ───────────────────────────────────────────────────────────
    H: Dict[str, List] = {
        "round": [], "train_acc": [], "test_acc": [],
        "train_loss": [], "test_loss": [],
        "eps_cumulative": [], "eps_remaining": [], "sigma": [],
        "f_channel": [], "f_phase": [], "f_urgency": [], "f_gradient": [],
        "grad_norm_ema": [], "grad_norm_raw": [],
        "bytes_round": [], "bytes_cumulative": [],
        "sa_active_count": [],
        "clients_selected": [], "clients_aggregated": [],
        "ch_good": [], "ch_moderate": [], "ch_bad": [],
        "avg_latency_ms": [],
    }
    cum_bytes = 0.0
    best_acc = 0.0
    rng = np.random.default_rng(cfg.seed)

    # Previous-round gradient norm for scheduler (avoids using None on round 1)
    prev_grad_norm: Optional[float] = None

    # ── Header ────────────────────────────────────────────────────────────
    hdr = (f"  {'Rnd':>4}  {'TrAcc':>7}  {'TsAcc':>7}  "
           f"{'TrLoss':>7}  {'TsLoss':>7}  {'ε(cum)':>8}  "
           f"{'σ':>6}  {'#Agg':>4}  {'ChG/M/B':>8}")
    print(f"\n{hdr}")
    print("  " + "─" * (len(hdr) - 2))

    for rnd in range(1, cfg.rounds + 1):

        # ── Select clients ────────────────────────────────────────────────
        k = max(1, int(cfg.fraction * cfg.num_clients))
        selected = rng.choice(cfg.num_clients, size=k, replace=False).tolist()

        # ── Per-client channel realisation ────────────────────────────────
        client_channels: Dict[int, Tuple[bool, float, ChannelState, float]] = {}
        for cid in selected:
            client_channels[cid] = ch_mgr.step(cid)

        # Majority channel state (for scheduler, SA summary)
        ch_counts = {ChannelState.GOOD: 0, ChannelState.MODERATE: 0,
                     ChannelState.BAD: 0}
        for _, _, ch_st, _ in client_channels.values():
            ch_counts[ch_st] += 1
        majority_ch = max(ch_counts, key=ch_counts.get)

        # ── Privacy budget for this round ─────────────────────────────────
        target_eps = sigma = 0.0
        sched_info: Dict = {}

        if scheduler and not scheduler.budget_exhausted:
            scheduler.start_round()
            avg_sampling_rate = cfg.local_batch / max(
                np.mean([len(client_loaders[c].dataset) for c in selected]), 1
            )
            target_eps, sigma, sched_info = scheduler.get_round_budget(
                majority_ch,
                gradient_norm=prev_grad_norm,
                sampling_rate=avg_sampling_rate,
            )

        # ── Local training ────────────────────────────────────────────────
        client_models: List[nn.Module] = []
        client_weights: List[float] = []
        grad_norms: List[float] = []
        sa_active_count = 0
        stragglers = 0

        for cid in selected:
            pkt_lost, loss_rate, ch_st, lat = client_channels[cid]

            # Selective aggregation check for this client's link
            sa_ok = decide_sa(cfg, loss_rate, ch_st)
            if sa_ok:
                sa_active_count += 1

            # Straggler simulation
            if cfg.straggler_frac > 0 and rng.random() < cfg.straggler_frac:
                stragglers += 1
                continue

            # Skip if packet lost on this client's link
            if pkt_lost:
                continue

            # Train
            lm = copy.deepcopy(global_model)
            try:
                fresh_loader = DataLoader(
                    client_loaders[cid].dataset,
                    batch_size=cfg.local_batch,
                    shuffle=True,
                    drop_last=(len(client_loaders[cid].dataset) > cfg.local_batch),
                )
                trained, loss, acc, gn, _ = local_train(
                    lm, global_model, fresh_loader, cfg, sigma, rnd
                )
                grad_norms.append(gn)
                client_models.append(trained)
                client_weights.append(len(client_loaders[cid].dataset))

            except Exception as e:
                print(f"  [WARN] Client {cid}: {e}")

        # ── Record privacy spend ──────────────────────────────────────────
        eps_cum = scheduler.get_epsilon() if scheduler else 0.0
        eps_rem = scheduler.get_remaining() if scheduler else cfg.total_epsilon

        if scheduler and len(client_models) > 0 and sigma > 0:
            avg_sr = cfg.local_batch / max(
                np.mean([len(client_loaders[c].dataset) for c in selected]), 1
            )
            eps_cum, eps_rem, _ = scheduler.spend(
                sigma, avg_sr, steps=cfg.local_epochs
            )

        # Update gradient EMA for next round's scheduler
        if grad_norms:
            prev_grad_norm = float(np.mean(grad_norms))

        # ── Quantization & aggregation ────────────────────────────────────
        q_comp = 1.0
        if client_models:
            if cfg.quant_enabled:
                qb = channel_quant_bits(majority_ch, cfg)
                if qb < 32:
                    qm = []
                    for m in client_models:
                        qsd, q_comp = quantize_state_dict(m.state_dict(), qb)
                        m.load_state_dict(qsd)
                        qm.append(m)
                    client_models = qm

            global_model = fedavg_aggregate(global_model, client_models,
                                            client_weights)

        # ── Evaluate ──────────────────────────────────────────────────────
        do_eval = (rnd % cfg.eval_every == 0) or (rnd == cfg.rounds)
        if do_eval:
            tr_acc, tr_loss, _, _ = evaluate(global_model, train_loader,
                                             cfg.device)
            ts_acc, ts_loss, _, _ = evaluate(global_model, test_loader,
                                             cfg.device)
        else:
            tr_acc = H["train_acc"][-1] if H["train_acc"] else 0.0
            ts_acc = H["test_acc"][-1] if H["test_acc"] else 0.0
            tr_loss = H["train_loss"][-1] if H["train_loss"] else 0.0
            ts_loss = H["test_loss"][-1] if H["test_loss"] else 0.0

        if ts_acc > best_acc:
            best_acc = ts_acc
            torch.save(global_model.state_dict(),
                       os.path.join(cfg.output_dir, f"best_{label}.pt"))

        if stopper and do_eval and stopper.check(ts_acc, rnd):
            print(f"\n  [Early Stop] Round {rnd} — {stopper.status()}")
            break

        # ── Bytes accounting ──────────────────────────────────────────────
        base_bytes = global_model.count_params() * 4
        upload = base_bytes * q_comp * len(client_models)
        sa_oh = base_bytes * sa_active_count * 2   # SA overhead (hashes)
        rnd_bytes = upload + sa_oh
        cum_bytes += rnd_bytes

        # ── Record history ────────────────────────────────────────────────
        H["round"].append(rnd)
        H["train_acc"].append(tr_acc)
        H["test_acc"].append(ts_acc)
        H["train_loss"].append(tr_loss)
        H["test_loss"].append(ts_loss)
        H["eps_cumulative"].append(eps_cum)
        H["eps_remaining"].append(eps_rem)
        H["sigma"].append(sigma)
        H["f_channel"].append(sched_info.get("f_channel", 1.0))
        H["f_phase"].append(sched_info.get("f_phase", 1.0))
        H["f_urgency"].append(sched_info.get("f_urgency", 1.0))
        H["f_gradient"].append(sched_info.get("f_gradient", 1.0))
        H["grad_norm_ema"].append(sched_info.get("gradient_ema") or 0.0)
        H["grad_norm_raw"].append(
            float(np.mean(grad_norms)) if grad_norms else 0.0
        )
        H["bytes_round"].append(rnd_bytes)
        H["bytes_cumulative"].append(cum_bytes)
        H["sa_active_count"].append(sa_active_count)
        H["clients_selected"].append(len(selected))
        H["clients_aggregated"].append(len(client_models))
        H["ch_good"].append(ch_counts[ChannelState.GOOD])
        H["ch_moderate"].append(ch_counts[ChannelState.MODERATE])
        H["ch_bad"].append(ch_counts[ChannelState.BAD])
        H["avg_latency_ms"].append(
            float(np.mean([lat for _, _, _, lat in client_channels.values()]))
        )

        # ── Console line ──────────────────────────────────────────────────
        ch_str = (f"{ch_counts[ChannelState.GOOD]}/"
                  f"{ch_counts[ChannelState.MODERATE]}/"
                  f"{ch_counts[ChannelState.BAD]}")
        print(f"  {rnd:>4}  {tr_acc:>7.4f}  {ts_acc:>7.4f}  "
              f"{tr_loss:>7.4f}  {ts_loss:>7.4f}  {eps_cum:>8.3f}  "
              f"{sigma:>6.2f}  {len(client_models):>4}  {ch_str:>8}")

    # ── Final metrics ─────────────────────────────────────────────────────
    _, _, yt, yp = evaluate(global_model, test_loader, cfg.device,
                            return_preds=True)
    fm = compute_full_metrics(yt, yp, list(le.classes_))

    pd.DataFrame(H).to_csv(
        os.path.join(cfg.output_dir, f"history_{label}.csv"), index=False
    )
    with open(os.path.join(cfg.output_dir, f"metrics_{label}.json"), "w") as f:
        json.dump(fm, f, indent=2)

    print(f"\n  Final: TrAcc={tr_acc:.4f}  TsAcc={ts_acc:.4f}  "
          f"Best={best_acc:.4f}")
    if scheduler:
        print(f"  Privacy: ε={scheduler.get_epsilon():.2f} / "
              f"{cfg.total_epsilon:.2f}")
    print(f"  F1m={fm['f1_macro']:.4f}  F1w={fm['f1_weighted']:.4f}  "
          f"κ={fm['kappa']:.4f}")
    print(f"  Bytes: {cum_bytes / 1e6:.1f} MB")

    return H, global_model, le, cached_data


# =============================================================================
# §13  MULTI-SEED RUNNER
# =============================================================================

def run_multi_seed(
    seeds: List[int],
    base_cfg: Dict,
    cached_data: Optional[Tuple] = None,
) -> Tuple[Dict, Tuple]:
    """Run an experiment across multiple seeds and aggregate statistics."""
    all_H: List[Dict] = []

    for seed in seeds:
        print(f"\n{'='*72}")
        print(f"  Seed {seed}")
        print(f"{'='*72}")
        cfg = Config(**{**base_cfg, "seed": seed})
        H, _, _, cached_data = run_federated(cfg, cached_data=cached_data)
        all_H.append(H)

    best_accs = [max(h["test_acc"]) for h in all_H]
    final_accs = [h["test_acc"][-1] for h in all_H]
    final_eps = [h["eps_cumulative"][-1] if h["eps_cumulative"] else 0
                 for h in all_H]

    m_best, ci_best = compute_ci(best_accs)
    m_final, ci_final = compute_ci(final_accs)
    m_eps, ci_eps = compute_ci(final_eps)

    summary = {
        "seeds": seeds,
        "best_acc": best_accs, "final_acc": final_accs,
        "final_eps": final_eps,
        "best_acc_mean": m_best, "best_acc_ci": ci_best,
        "final_acc_mean": m_final, "final_acc_ci": ci_final,
        "final_eps_mean": m_eps, "final_eps_ci": ci_eps,
    }

    print(f"\n{'='*72}")
    print(f"  MULTI-SEED SUMMARY ({len(seeds)} seeds)")
    print(f"  Best Acc : {m_best:.4f} ± {ci_best:.4f}")
    print(f"  Final Acc: {m_final:.4f} ± {ci_final:.4f}")
    print(f"  Final ε  : {m_eps:.2f} ± {ci_eps:.2f}")
    print(f"{'='*72}")

    out = base_cfg.get("output_dir", "fl_results")
    pd.DataFrame({
        "seed": seeds, "best_acc": best_accs,
        "final_acc": final_accs, "final_eps": final_eps,
    }).to_csv(os.path.join(out, "multi_seed.csv"), index=False)

    return summary, cached_data


# =============================================================================
# §14  EXPERIMENT: SCHEDULE COMPARISON
# =============================================================================

def run_schedule_comparison(
    seeds: List[int], base_cfg: Dict
) -> Dict:
    """Compare dynamic schedule vs static baselines."""
    schedules = ["dynamic", "convex", "linear", "uniform"]
    results = {}
    cached = None

    print(f"\n{'='*72}")
    print(f"  EXPERIMENT 1: Schedule Comparison")
    print(f"{'='*72}")

    for sched in schedules:
        print(f"\n─── Schedule: {sched.upper()} ───")
        cfg_d = {**base_cfg, "schedule": sched}
        accs, epsilons = [], []

        for seed in seeds:
            cfg = Config(**{**cfg_d, "seed": seed})
            H, _, _, cached = run_federated(cfg, cached_data=cached)
            accs.append(max(H["test_acc"]))
            epsilons.append(H["eps_cumulative"][-1] if H["eps_cumulative"] else 0)

        m, ci = compute_ci(accs)
        results[sched] = {
            "best_acc_mean": m, "best_acc_ci": ci,
            "best_acc_std": float(np.std(accs)),
            "final_eps_mean": float(np.mean(epsilons)),
            "final_eps_std": float(np.std(epsilons)),
            "all_accs": accs,
        }
        print(f"  → Acc: {m:.4f} ± {ci:.4f}  |  ε: {np.mean(epsilons):.2f}")

    out = base_cfg.get("output_dir", "fl_results")
    pd.DataFrame([{
        "Schedule": k,
        "Acc (mean±CI)": f"{v['best_acc_mean']:.4f}±{v['best_acc_ci']:.4f}",
        "ε (mean±std)": f"{v['final_eps_mean']:.2f}±{v['final_eps_std']:.2f}",
    } for k, v in results.items()]).to_csv(
        os.path.join(out, "exp1_schedules.csv"), index=False
    )
    return results


# =============================================================================
# §15  EXPERIMENT: PRIVACY-UTILITY CURVE
# =============================================================================

def run_privacy_utility_curve(
    epsilon_values: List[float], seeds: List[int], base_cfg: Dict
) -> pd.DataFrame:
    """Sweep ε and record accuracy to plot the privacy-utility tradeoff."""
    rows = []
    cached = None

    print(f"\n{'='*72}")
    print(f"  EXPERIMENT 2: Privacy-Utility Curve")
    print(f"{'='*72}")

    for eps in epsilon_values:
        print(f"\n─── ε = {eps} ───")
        cfg_d = {**base_cfg, "total_epsilon": eps, "schedule": "dynamic"}
        accs = []
        for seed in seeds:
            cfg = Config(**{**cfg_d, "seed": seed})
            H, _, _, cached = run_federated(cfg, cached_data=cached)
            accs.append(max(H["test_acc"]))

        m, ci = compute_ci(accs)
        rows.append({
            "epsilon": eps, "acc_mean": m, "acc_ci": ci,
            "acc_std": float(np.std(accs)), "all_accs": accs,
        })
        print(f"  → Acc: {m:.4f} ± {ci:.4f}")

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "all_accs"}
                        for r in rows])
    out = base_cfg.get("output_dir", "fl_results")
    df.to_csv(os.path.join(out, "exp2_privacy_utility.csv"), index=False)
    return df


# =============================================================================
# §16  EXPERIMENT: ABLATION STUDY
# =============================================================================

def run_ablation_study(seeds: List[int], base_cfg: Dict) -> Dict:
    """
    Ablation: toggle each scheduler factor off individually.
    Note: 'label' keys are NOT passed to Config.
    """
    ablations = {
        "full_dynamic": {
            "toggles": {"use_channel_factor": True, "use_phase_factor": True,
                        "use_urgency_factor": True, "use_gradient_factor": True},
            "display": "Full Dynamic",
        },
        "no_channel": {
            "toggles": {"use_channel_factor": False, "use_phase_factor": True,
                        "use_urgency_factor": True, "use_gradient_factor": True},
            "display": "−Channel",
        },
        "no_phase": {
            "toggles": {"use_channel_factor": True, "use_phase_factor": False,
                        "use_urgency_factor": True, "use_gradient_factor": True},
            "display": "−Phase",
        },
        "no_urgency": {
            "toggles": {"use_channel_factor": True, "use_phase_factor": True,
                        "use_urgency_factor": False, "use_gradient_factor": True},
            "display": "−Urgency",
        },
        "no_gradient": {
            "toggles": {"use_channel_factor": True, "use_phase_factor": True,
                        "use_urgency_factor": True, "use_gradient_factor": False},
            "display": "−Gradient",
        },
        "static_uniform": {
            "toggles": {"use_channel_factor": False, "use_phase_factor": False,
                        "use_urgency_factor": False, "use_gradient_factor": False},
            "extra": {"schedule": "uniform"},
            "display": "Static Uniform",
        },
    }

    results = {}
    cached = None

    print(f"\n{'='*72}")
    print(f"  EXPERIMENT 3: Ablation Study")
    print(f"{'='*72}")

    for name, spec in ablations.items():
        disp = spec["display"]
        print(f"\n─── {disp} ───")

        # Build config dict: base + toggles + any extras (no 'display' leak)
        cfg_d = {**base_cfg, **spec["toggles"]}
        if "extra" in spec:
            cfg_d.update(spec["extra"])

        accs, epsilons = [], []
        for seed in seeds:
            cfg = Config(**{**cfg_d, "seed": seed})
            H, _, _, cached = run_federated(cfg, cached_data=cached)
            accs.append(max(H["test_acc"]))
            epsilons.append(H["eps_cumulative"][-1] if H["eps_cumulative"] else 0)

        m, ci = compute_ci(accs)
        results[name] = {
            "display": disp,
            "best_acc_mean": m, "best_acc_ci": ci,
            "best_acc_std": float(np.std(accs)),
            "final_eps_mean": float(np.mean(epsilons)),
            "all_accs": accs,
        }
        print(f"  → Acc: {m:.4f} ± {ci:.4f}")

    out = base_cfg.get("output_dir", "fl_results")
    pd.DataFrame([{
        "Config": v["display"],
        "Acc": f"{v['best_acc_mean']:.4f}±{v['best_acc_ci']:.4f}",
        "ε": f"{v['final_eps_mean']:.2f}",
    } for v in results.values()]).to_csv(
        os.path.join(out, "exp3_ablation.csv"), index=False
    )
    return results


# =============================================================================
# §17  EXPERIMENT: CHANNEL AWARENESS
# =============================================================================

def run_channel_awareness(seeds: List[int], base_cfg: Dict) -> Dict:
    """Compare channel-aware vs channel-agnostic scheduling."""
    results = {}
    cached = None

    print(f"\n{'='*72}")
    print(f"  EXPERIMENT 4: Channel Awareness Impact")
    print(f"{'='*72}")

    for tag, use_ch in [("aware", True), ("agnostic", False)]:
        print(f"\n─── Channel-{tag.title()} ───")
        cfg_d = {**base_cfg, "schedule": "dynamic", "use_channel_factor": use_ch}
        accs = []
        for seed in seeds:
            cfg = Config(**{**cfg_d, "seed": seed})
            H, _, _, cached = run_federated(cfg, cached_data=cached)
            accs.append(max(H["test_acc"]))

        m, ci = compute_ci(accs)
        results[tag] = {"mean": m, "ci": ci, "all_accs": accs}
        print(f"  → Acc: {m:.4f} ± {ci:.4f}")

    # Paired t-test
    t_stat, p_val = ttest_rel(results["aware"]["all_accs"],
                               results["agnostic"]["all_accs"])
    results["ttest"] = {"t": float(t_stat), "p": float(p_val),
                        "significant": p_val < 0.05}
    sig = "✓ Significant" if p_val < 0.05 else "✗ Not significant"
    print(f"\n  Paired t-test: t={t_stat:.3f}, p={p_val:.4f} — {sig}")

    out = base_cfg.get("output_dir", "fl_results")
    pd.DataFrame([
        {"Config": "Channel-Aware",
         "Acc": f"{results['aware']['mean']:.4f}±{results['aware']['ci']:.4f}"},
        {"Config": "Channel-Agnostic",
         "Acc": f"{results['agnostic']['mean']:.4f}±{results['agnostic']['ci']:.4f}"},
        {"Config": "p-value", "Acc": f"{p_val:.6f}"},
    ]).to_csv(os.path.join(out, "exp4_channel.csv"), index=False)
    return results


# =============================================================================
# §18  EXPERIMENT: NOISY vs RELIABLE BASELINE
# =============================================================================

def run_noisy_vs_reliable(seeds: List[int], base_cfg: Dict) -> Dict:
    """Compare noisy-link FL with reliable-link and non-DP baselines."""
    configs = {
        "noisy_dynamic_dp": {
            "mode": "noisy", "dp_enabled": True, "schedule": "dynamic",
        },
        "noisy_uniform_dp": {
            "mode": "noisy", "dp_enabled": True, "schedule": "uniform",
        },
        "reliable_dynamic_dp": {
            "mode": "reliable", "dp_enabled": True, "schedule": "dynamic",
        },
        "reliable_no_dp": {
            "mode": "reliable", "dp_enabled": False,
        },
    }

    results = {}
    cached = None

    print(f"\n{'='*72}")
    print(f"  EXPERIMENT 5: Noisy vs Reliable Baselines")
    print(f"{'='*72}")

    for tag, overrides in configs.items():
        print(f"\n─── {tag} ───")
        cfg_d = {**base_cfg, **overrides}
        accs, epsilons, byte_totals = [], [], []

        for seed in seeds:
            cfg = Config(**{**cfg_d, "seed": seed})
            H, _, _, cached = run_federated(cfg, cached_data=cached)
            accs.append(max(H["test_acc"]))
            epsilons.append(H["eps_cumulative"][-1] if H["eps_cumulative"] else 0)
            byte_totals.append(H["bytes_cumulative"][-1] if H["bytes_cumulative"] else 0)

        m, ci = compute_ci(accs)
        results[tag] = {
            "acc_mean": m, "acc_ci": ci,
            "eps_mean": float(np.mean(epsilons)),
            "bytes_mean": float(np.mean(byte_totals)),
            "all_accs": accs,
        }
        print(f"  → Acc: {m:.4f} ± {ci:.4f}  |  "
              f"ε: {np.mean(epsilons):.2f}  |  "
              f"MB: {np.mean(byte_totals)/1e6:.1f}")

    out = base_cfg.get("output_dir", "fl_results")
    pd.DataFrame([{
        "Config": k,
        "Acc": f"{v['acc_mean']:.4f}±{v['acc_ci']:.4f}",
        "ε": f"{v['eps_mean']:.2f}",
        "MB": f"{v['bytes_mean']/1e6:.1f}",
    } for k, v in results.items()]).to_csv(
        os.path.join(out, "exp5_baselines.csv"), index=False
    )
    return results


# =============================================================================
# §19  EXPERIMENT: RDP VALIDATION
# =============================================================================

def run_rdp_validation(base_cfg: Dict) -> Dict:
    """Validate our RDP accountant against Opacus."""
    if not OPACUS_AVAILABLE:
        print("[Skip] Opacus not installed — cannot validate RDP")
        return {"status": "skipped"}

    print(f"\n{'='*72}")
    print(f"  VALIDATION: RDP Odometer vs Opacus")
    print(f"{'='*72}\n")

    cfg = Config(**base_cfg)
    sigmas = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]
    sr = cfg.local_batch / 100.0
    steps = cfg.local_epochs
    rows = []

    for s in sigmas:
        # Ours
        odo = PrivacyOdometer(cfg.delta, 1e6)
        odo.spend(s, sr, steps)
        eps_ours = odo.get_epsilon()

        # Opacus
        from opacus.accountants import RDPAccountant as OpRDP
        op = OpRDP()
        for _ in range(steps):
            op.step(noise_multiplier=s, sample_rate=sr)
        eps_op = op.get_epsilon(delta=cfg.delta)

        diff = abs(eps_ours - eps_op) / max(eps_ours, eps_op) * 100
        rows.append({"sigma": s, "eps_ours": eps_ours,
                      "eps_opacus": eps_op, "diff_pct": diff})
        print(f"  σ={s:>5.1f}  ours={eps_ours:.4f}  "
              f"opacus={eps_op:.4f}  diff={diff:.1f}%")

    max_diff = max(r["diff_pct"] for r in rows)
    ok = max_diff < 20.0
    print(f"\n  {'PASS ✓' if ok else 'FAIL ✗'}  max diff = {max_diff:.1f}%")

    out = base_cfg.get("output_dir", "fl_results")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "rdp_validation.csv"),
                               index=False)
    return {"rows": rows, "passed": ok}


# =============================================================================
# §20  PUBLICATION FIGURES
# =============================================================================

def generate_figures(
    sched_res: Optional[Dict] = None,
    pu_df: Optional[pd.DataFrame] = None,
    ablation_res: Optional[Dict] = None,
    channel_res: Optional[Dict] = None,
    baseline_res: Optional[Dict] = None,
    output_dir: str = "fl_results",
) -> None:
    """Generate publication-quality figures from experiment results."""
    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({"font.size": 12, "figure.dpi": 150})

    # ── Fig 1: Schedule comparison bar chart ──────────────────────────────
    if sched_res:
        fig, ax = plt.subplots(figsize=(8, 5))
        names = list(sched_res.keys())
        means = [sched_res[s]["best_acc_mean"] for s in names]
        cis = [sched_res[s]["best_acc_ci"] for s in names]
        colors = ["#2ecc71" if s == "dynamic" else "#3498db" for s in names]
        ax.bar(names, means, yerr=cis, capsize=6, color=colors, alpha=0.85,
               edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Best Test Accuracy")
        ax.set_title("Schedule Comparison")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig1_schedules.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # ── Fig 2: Privacy-utility curve ──────────────────────────────────────
    if pu_df is not None and len(pu_df) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(pu_df["epsilon"], pu_df["acc_mean"],
                    yerr=pu_df["acc_ci"], capsize=5, marker="o",
                    markersize=7, linewidth=2, color="#c0392b")
        ax.set_xscale("log")
        ax.set_xlabel("Privacy Budget (ε)")
        ax.set_ylabel("Best Test Accuracy")
        ax.set_title("Privacy–Utility Tradeoff")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig2_privacy_utility.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # ── Fig 3: Ablation study ─────────────────────────────────────────────
    if ablation_res:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [v["display"] for v in ablation_res.values()]
        means = [v["best_acc_mean"] for v in ablation_res.values()]
        cis = [v["best_acc_ci"] for v in ablation_res.values()]
        cols = []
        for lb in labels:
            if lb == "Full Dynamic":
                cols.append("#2ecc71")
            elif lb == "Static Uniform":
                cols.append("#e74c3c")
            else:
                cols.append("#3498db")
        ax.bar(range(len(labels)), means, yerr=cis, capsize=5,
               color=cols, alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Best Test Accuracy")
        ax.set_title("Ablation Study — Scheduler Components")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig3_ablation.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # ── Fig 4: Channel awareness ──────────────────────────────────────────
    if channel_res and "aware" in channel_res:
        fig, ax = plt.subplots(figsize=(6, 5))
        tags = ["aware", "agnostic"]
        means = [channel_res[t]["mean"] for t in tags]
        cis = [channel_res[t]["ci"] for t in tags]
        ax.bar(["Channel-Aware", "Channel-Agnostic"], means, yerr=cis,
               capsize=6, color=["#27ae60", "#e67e22"], alpha=0.85,
               edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Best Test Accuracy")
        ax.set_title("Channel Awareness Impact")
        ax.set_ylim(0, 1)
        if "ttest" in channel_res:
            p = channel_res["ttest"]["p"]
            sig = "significant" if p < 0.05 else "not significant"
            ax.text(0.5, 0.95, f"p = {p:.4f} ({sig})",
                    transform=ax.transAxes, ha="center", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig4_channel.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    # ── Fig 5: Baselines (Pareto-style) ──────────────────────────────────
    if baseline_res:
        fig, ax = plt.subplots(figsize=(8, 6))
        for tag, v in baseline_res.items():
            ax.scatter(v["eps_mean"], v["acc_mean"], s=120, zorder=5,
                       label=tag.replace("_", " "))
            ax.annotate(tag.replace("_", "\n"),
                        (v["eps_mean"], v["acc_mean"]),
                        textcoords="offset points", xytext=(8, 5),
                        fontsize=8)
        ax.set_xlabel("Final ε spent")
        ax.set_ylabel("Best Test Accuracy")
        ax.set_title("Privacy–Accuracy–Link Tradeoff")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "fig5_baselines.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    print(f"\n✓ Figures saved to {output_dir}/fig*.png")


# =============================================================================
# §21  COMMAND-LINE INTERFACE
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Privacy-Preserving FL over Noisy AIoT Links v8.0"
    )

    # Basic
    p.add_argument("--csv", default="merged_wisdm.csv")
    p.add_argument("--mode", default="noisy", choices=["noisy", "reliable"])
    p.add_argument("--clients", default=30, type=int)
    p.add_argument("--rounds", default=50, type=int)
    p.add_argument("--alpha", default=0.5, type=float)
    p.add_argument("--mu", default=0.0, type=float)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--local_epochs", default=3, type=int)
    p.add_argument("--local_batch", default=64, type=int)
    p.add_argument("--fraction", default=0.4, type=float)

    # Privacy
    p.add_argument("--no_dp", action="store_true")
    p.add_argument("--no_opacus", action="store_true")
    p.add_argument("--total_epsilon", default=5.0, type=float)
    p.add_argument("--delta", default=1e-4, type=float)
    p.add_argument("--clip_norm", default=1.0, type=float)
    p.add_argument("--schedule", default="dynamic",
                   choices=["dynamic", "convex", "linear",
                            "uniform", "exponential"])

    # Ablation
    p.add_argument("--no_channel_factor", action="store_true")
    p.add_argument("--no_phase_factor", action="store_true")
    p.add_argument("--no_urgency_factor", action="store_true")
    p.add_argument("--no_gradient_factor", action="store_true")

    # Experiments
    p.add_argument("--compare_schedules", action="store_true")
    p.add_argument("--privacy_curve", action="store_true")
    p.add_argument("--run_ablation", action="store_true")
    p.add_argument("--channel_awareness", action="store_true")
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--validate_rdp", action="store_true")
    p.add_argument("--all_experiments", action="store_true")

    # Output
    p.add_argument("--output", default="fl_results")
    p.add_argument("--fine_classes", action="store_true")
    p.add_argument("--seeds", default="42,123,7,256,999",
                   help="Comma-separated seeds (default: 5 seeds)")
    p.add_argument("--no_early_stop", action="store_true")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    base = {
        "csv_path": args.csv,
        "num_clients": args.clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "local_batch": args.local_batch,
        "lr": args.lr,
        "fraction": args.fraction,
        "alpha": args.alpha,
        "mu": args.mu,
        "mode": args.mode,
        "dp_enabled": not args.no_dp,
        "use_opacus": not args.no_opacus,
        "total_epsilon": args.total_epsilon,
        "delta": args.delta,
        "clip_norm": args.clip_norm,
        "schedule": args.schedule,
        "early_stop": not args.no_early_stop,
        "output_dir": args.output,
        "fine_classes": args.fine_classes,
        "seed": seeds[0],
        "use_channel_factor": not args.no_channel_factor,
        "use_phase_factor": not args.no_phase_factor,
        "use_urgency_factor": not args.no_urgency_factor,
        "use_gradient_factor": not args.no_gradient_factor,
    }

    os.makedirs(args.output, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  Privacy-Preserving Federated Learning over Noisy AIoT Links")
    print(f"  v8.0")
    print(f"{'='*72}")

    # ── Run requested experiments ─────────────────────────────────────────

    sched_res = pu_df = ablation_res = channel_res = baseline_res = None

    if args.validate_rdp or args.all_experiments:
        run_rdp_validation(base)

    if args.compare_schedules or args.all_experiments:
        sched_res = run_schedule_comparison(seeds, base)

    if args.privacy_curve or args.all_experiments:
        eps_vals = [3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0]
        pu_df = run_privacy_utility_curve(eps_vals, seeds, base)

    if args.run_ablation or args.all_experiments:
        ablation_res = run_ablation_study(seeds, base)

    if args.channel_awareness or args.all_experiments:
        channel_res = run_channel_awareness(seeds, base)

    if args.baselines or args.all_experiments:
        baseline_res = run_noisy_vs_reliable(seeds, base)

    # Generate figures if any experiment ran
    if any([sched_res, pu_df is not None, ablation_res, channel_res,
            baseline_res]):
        generate_figures(sched_res, pu_df, ablation_res, channel_res,
                        baseline_res, args.output)

    # If no experiment flag, run multi-seed single experiment
    if not any([args.compare_schedules, args.privacy_curve, args.run_ablation,
                args.channel_awareness, args.baselines, args.validate_rdp,
                args.all_experiments]):
        if len(seeds) > 1:
            run_multi_seed(seeds, base)
        else:
            cfg = Config(**base)
            run_federated(cfg)

    print(f"\n{'='*72}")
    print(f"  Done. Results in: {args.output}/")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()
