"""
Builds a leak-free, trade-level sequential signal set from a scored dollar-bar
dataframe.

Both build_meta_labels.py (training) and diagnostics.py (evaluation) must
turn a per-bar scored dataframe into a set of meta-model training/eval rows
the same way, or the two silently drift apart (this happened before: dense,
non-deduplicated per-bar "signals" let recent_signal_pnl_5 leak the current
trade's own outcome via overlapping triple-barrier windows, producing a
model that looked great offline and blocked ~99.97% of everything live).

The live SignalEngine can only ever hold one position at a time - while
active_position is True it never evaluates new entries at all. This module
mirrors that constraint: once a bar clears the entry threshold and is taken
as a signal, no bar within the next `horizon` bars (the same
forecast_horizon_bars used both to build label_long/label_short and as
RiskManager's live time-stop) can be selected as a new signal.
"""
import numpy as np
import pandas as pd


def _select_sequential_indices(candidate: np.ndarray, horizon: int) -> np.ndarray:
    n = len(candidate)
    selected = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if candidate[i]:
            selected[i] = True
            i += horizon
        else:
            i += 1
    return selected


def build_sequential_meta_signals(df: pd.DataFrame, entry_threshold: float, horizon: int) -> pd.DataFrame:
    """
    df must be sorted by close_time (single symbol only - call once per
    symbol, never on a multi-symbol concat) and already contain: long_prob,
    short_prob, label_long, label_short, close_time, close, ema_4h.

    Returns one row per real, non-overlapping trade opportunity: a bar only
    counts as a signal if it isn't within `horizon` bars of a prior selected
    signal, matching the live bot's single-position lockout.
    """
    df = df.reset_index(drop=True)
    long_prob = df["long_prob"].to_numpy()
    short_prob = df["short_prob"].to_numpy()

    is_long_sig = (long_prob >= entry_threshold) & (long_prob > short_prob)
    is_short_sig = (short_prob >= entry_threshold) & (short_prob > long_prob)
    candidate = is_long_sig | is_short_sig

    selected = _select_sequential_indices(candidate, horizon)
    sig_df = df.loc[selected].copy().reset_index(drop=True)

    if len(sig_df) == 0:
        return sig_df

    sig_df["is_long_signal"] = np.where(sig_df["long_prob"] > sig_df["short_prob"], 1, 0)
    sig_df["confidence_gap"] = np.abs(sig_df["long_prob"] - sig_df["short_prob"])
    sig_df["trend_alignment"] = (
        np.sign(sig_df["is_long_signal"] - 0.5) == np.sign(sig_df["close"] - sig_df["ema_4h"])
    ).astype(int)

    # Continuous trend-extension measure (ATR-normalized distance from the 4H EMA) -
    # near zero in calm/choppy conditions, large during violent trending breakouts.
    # Complements the binary trend_alignment/regime features with a magnitude signal.
    atr_safe = sig_df["atr_14"].replace(0, np.nan)
    sig_df["trend_strength"] = ((sig_df["close"] - sig_df["ema_4h"]) / atr_safe).fillna(0.0)

    is_correct_long = (sig_df["is_long_signal"] == 1) & (sig_df["label_long"] == 1)
    is_correct_short = (sig_df["is_long_signal"] == 0) & (sig_df["label_short"] == 1)
    sig_df["is_win"] = np.where(is_correct_long | is_correct_short, 1, 0)

    # Mean of up to the last 5 REAL, already-resolved trades (min_periods=1
    # mirrors deque(maxlen=5) filling up gradually rather than requiring a
    # full window of 5 before producing a real value). shift(1) excludes the
    # current trade's own outcome - this only reflects trades already closed
    # before this one was taken, exactly like the live deque which is only
    # ever appended to at trade exit.
    sig_df["recent_signal_pnl_5"] = (
        sig_df["is_win"].rolling(window=5, min_periods=1).mean().shift(1).fillna(0.5)
    )

    sig_df["timestamp"] = pd.to_datetime(sig_df["close_time"], unit="ms")
    sig_df["hours_since_last_trade"] = sig_df["timestamp"].diff().dt.total_seconds() / 3600.0
    sig_df["hours_since_last_trade"] = sig_df["hours_since_last_trade"].fillna(100.0)

    sig_df["meta_label"] = sig_df["is_win"]

    return sig_df
