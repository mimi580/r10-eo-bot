"""
R_10 EVEN/ODD SIGNAL ENGINE
==============================
Two-layer architecture combining:
  Option A: Statistical distribution analysis (medium window)
  Option B: Sequence-based prediction (short window)

DIGIT EXTRACTION:
  R_10 prices look like 5000.23 → last digit = 3 (from int(price*10) % 10)
  This is the digit the contract settles on.

LAYER 1 — DISTRIBUTION CONTEXT (Option A):
  A1: Recency-Weighted Frequency — is even/odd rate biased right now?
  A2: Z-Score Significance       — is the bias statistically real?
  A3: Chi-Square Consistency     — does full 0-9 distribution favor even/odd?

LAYER 2 — SEQUENCE PREDICTION (Option B):
  B1: Transition Matrix  — what typically follows the current digit type?
  B2: Run Analysis       — clustering or alternating pattern in sequence?
  B3: Positional Bias    — given last 3 digits, what appears at position +5?

CONFLUENCE:
  Layer 1: 2 of 3 models agree → direction established
  Layer 2: 2 of 3 models agree → direction confirmed
  Both layers must point the same way → trade fires
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Tuple
from scipy import stats as scipy_stats

import settings as S


@dataclass
class ModelResult:
    name:       str
    tradeable:  bool
    confidence: float
    direction:  Optional[str]   # "EVEN" | "ODD" | None
    detail:     dict = field(default_factory=dict)


@dataclass
class EOSignal:
    tradeable:    bool
    direction:    Optional[str]
    confidence:   float
    layer1_score: int
    layer2_score: int
    reasons:      list
    models:       dict = field(default_factory=dict)
    skip_reason:  str = ""


# ─────────────────────────────────────────────────────────────────
# DIGIT EXTRACTION
# ─────────────────────────────────────────────────────────────────

def extract_digit(price: float) -> int:
    """Extract last digit from R_10 price. e.g. 5000.23 → 3"""
    return int(round(price * 10)) % 10


def is_even(digit: int) -> bool:
    return digit in S.EVEN_DIGITS


def prices_to_digits(prices: np.ndarray) -> np.ndarray:
    """Convert price array to last-digit array."""
    return np.array([extract_digit(p) for p in prices], dtype=int)


def prices_to_binary(prices: np.ndarray) -> np.ndarray:
    """Convert prices to binary even(1)/odd(0) array."""
    digits = prices_to_digits(prices)
    return np.array([1 if is_even(d) else 0 for d in digits], dtype=int)


# ─────────────────────────────────────────────────────────────────
# PRE-FILTER: ENTROPY GATE
# ─────────────────────────────────────────────────────────────────

def entropy_gate(prices: np.ndarray) -> Tuple[bool, str]:
    """
    Binary Shannon entropy of even/odd sequence.
    Max entropy = 1.0 bit (perfectly uniform = no edge).
    Block if entropy above threshold — no structure to exploit.
    """
    if len(prices) < S.ENTROPY_WINDOW + 1:
        return True, ""

    binary = prices_to_binary(prices[-S.ENTROPY_WINDOW:])
    p_even = float(np.mean(binary))
    p_odd  = 1.0 - p_even

    if p_even == 0 or p_odd == 0:
        entropy = 0.0
    else:
        entropy = -(p_even * np.log2(p_even) + p_odd * np.log2(p_odd))

    if entropy > S.ENTROPY_MAX:
        return False, f"ENTROPY_HIGH {entropy:.4f}>{S.ENTROPY_MAX}"
    return True, ""


# ─────────────────────────────────────────────────────────────────
# ══ LAYER 1 — DISTRIBUTION CONTEXT (Option A) ══
# ─────────────────────────────────────────────────────────────────

def model_A1_frequency(prices: np.ndarray) -> ModelResult:
    """
    Recency-weighted even/odd frequency.

    Exponential decay weighting: recent digits count more.
    This makes the model responsive to regime shifts — if R_10
    suddenly starts producing more even digits, this detects it
    faster than equal-weight frequency.

    Halflife = FREQ_HALFLIFE digits.
    Minimum bias = FREQ_BIAS_MIN above 0.50 to vote.
    """
    name = "freq_bias"
    fail = ModelResult(name, False, 0.0, None)

    if len(prices) < S.FREQ_WINDOW + 1:
        return fail

    binary = prices_to_binary(prices[-S.FREQ_WINDOW:])
    n      = len(binary)

    # Exponential recency weights
    alpha   = np.log(2) / S.FREQ_HALFLIFE
    weights = np.exp(-alpha * np.arange(n)[::-1])
    weights /= weights.sum()

    w_even = float(np.dot(weights, binary))         # weighted P(even)
    w_odd  = 1.0 - w_even

    bias = abs(w_even - S.TRUE_PROB)

    if bias < S.FREQ_BIAS_MIN:
        return ModelResult(name, False, 0.0, None, {
            "w_even": round(w_even, 4),
            "bias":   round(bias, 4),
            "reason": "BIAS_WEAK",
        })

    direction = "EVEN" if w_even > S.TRUE_PROB else "ODD"
    conf      = float(np.clip(
        (bias - S.FREQ_BIAS_MIN) / (0.10 - S.FREQ_BIAS_MIN + 1e-6),
        0.0, 1.0))

    return ModelResult(name, True, conf, direction, {
        "w_even": round(w_even, 4),
        "w_odd":  round(w_odd, 4),
        "bias":   round(bias, 4),
    })


def model_A2_zscore(prices: np.ndarray) -> ModelResult:
    """
    Z-score significance test across multiple window sizes.

    Tests H0: P(even) = 0.50 against H1: P(even) ≠ 0.50.
    Scans ZSCORE_WINDOWS and picks the window with highest |Z|.
    Requires |Z| > ZSCORE_MIN to vote.

    Recency-weighted counts for responsiveness.
    """
    name = "zscore"
    fail = ModelResult(name, False, 0.0, None)

    if len(prices) < max(S.ZSCORE_WINDOWS) + 1:
        return fail

    best_z      = 0.0
    best_dir    = None
    best_window = 0
    best_p_obs  = 0.0

    for window in S.ZSCORE_WINDOWS:
        if len(prices) < window + 1:
            continue

        binary = prices_to_binary(prices[-window:])
        n      = len(binary)

        # Recency-weighted even rate
        alpha   = np.log(2) / (window // 4)
        weights = np.exp(-alpha * np.arange(n)[::-1])
        weights /= weights.sum()
        p_obs   = float(np.dot(weights, binary))

        se = float(np.sqrt(S.TRUE_PROB * (1 - S.TRUE_PROB) / n))
        z  = (p_obs - S.TRUE_PROB) / (se + 1e-10)

        if abs(z) > abs(best_z):
            best_z      = z
            best_dir    = "EVEN" if z > 0 else "ODD"
            best_window = window
            best_p_obs  = p_obs

    if abs(best_z) < S.ZSCORE_MIN:
        return ModelResult(name, False, 0.0, None, {
            "z":      round(best_z, 3),
            "reason": f"Z_WEAK |{best_z:.2f}|<{S.ZSCORE_MIN}",
        })

    conf = float(np.clip(
        (abs(best_z) - S.ZSCORE_MIN) / (S.ZSCORE_STRONG - S.ZSCORE_MIN + 1e-6),
        0.0, 1.0))

    return ModelResult(name, True, conf, best_dir, {
        "z":       round(best_z, 3),
        "window":  best_window,
        "p_obs":   round(best_p_obs, 4),
    })


def model_A3_chisquare(prices: np.ndarray) -> ModelResult:
    """
    Chi-square test on full 0-9 digit distribution.
    Tests whether the distribution is non-uniform, then checks
    whether the non-uniformity favors even or odd digits.

    This catches cases where specific digits (e.g. digit 2 and 4)
    are over-represented, creating even bias even when the overall
    even/odd split looks close to 50/50.
    """
    name = "chisquare"
    fail = ModelResult(name, False, 0.0, None)

    if len(prices) < S.CHISQ_WINDOW + 1:
        return fail

    digits   = prices_to_digits(prices[-S.CHISQ_WINDOW:])
    observed = np.bincount(digits, minlength=10)
    expected = np.full(10, S.CHISQ_WINDOW / 10.0)

    chi2, pvalue = scipy_stats.chisquare(observed, f_exp=expected)

    if pvalue > S.CHISQ_PVALUE_MAX:
        return ModelResult(name, False, 0.0, None, {
            "chi2":   round(float(chi2), 3),
            "pvalue": round(float(pvalue), 5),
            "reason": "UNIFORM",
        })

    # Which side has excess?
    even_rate = float(np.sum(observed[[0,2,4,6,8]]) / S.CHISQ_WINDOW)
    odd_rate  = float(np.sum(observed[[1,3,5,7,9]]) / S.CHISQ_WINDOW)
    direction = "EVEN" if even_rate > odd_rate else "ODD"
    excess    = abs(even_rate - odd_rate)

    conf = float(np.clip(1.0 - pvalue / S.CHISQ_PVALUE_MAX, 0.0, 1.0))

    return ModelResult(name, True, conf, direction, {
        "chi2":      round(float(chi2), 3),
        "pvalue":    round(float(pvalue), 5),
        "even_rate": round(even_rate, 4),
        "odd_rate":  round(odd_rate, 4),
        "excess":    round(excess, 4),
    })


# ─────────────────────────────────────────────────────────────────
# ══ LAYER 2 — SEQUENCE PREDICTION (Option B) ══
# ─────────────────────────────────────────────────────────────────

def model_B1_transition(prices: np.ndarray) -> ModelResult:
    """
    Lag-1 transition probability matrix.

    Builds P(next_type | current_type) from last TRANSITION_WINDOW digits.
    Under true randomness: P(even|even) = P(even|odd) = 0.50.
    Deviation = exploitable sequential dependency.

    Uses the CURRENT last digit to predict the next one.
    """
    name = "transition"
    fail = ModelResult(name, False, 0.0, None)

    if len(prices) < S.TRANSITION_WINDOW + 2:
        return fail

    binary = prices_to_binary(prices[-S.TRANSITION_WINDOW:])
    n      = len(binary)

    # Build transition counts with recency weighting
    alpha   = np.log(2) / (S.TRANSITION_WINDOW // 4)
    weights = np.exp(-alpha * np.arange(n - 1)[::-1])
    weights /= weights.sum()

    # 2x2 matrix: [from_even, from_odd] x [to_even, to_odd]
    matrix = np.zeros((2, 2))
    for i in range(n - 1):
        from_type = binary[i]       # 1=even, 0=odd
        to_type   = binary[i + 1]
        matrix[from_type][to_type] += weights[i]

    # Normalize rows
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    matrix /= row_sums

    # Current digit type
    current_is_even = int(is_even(extract_digit(float(prices[-1]))))
    row = matrix[current_is_even]

    p_even_next = float(row[1])   # P(even | current type)
    p_odd_next  = float(row[0])   # P(odd  | current type)

    dev = abs(p_even_next - S.TRUE_PROB)

    if dev < S.TRANSITION_MIN_DEV:
        return ModelResult(name, False, 0.0, None, {
            "p_even_next": round(p_even_next, 4),
            "dev":         round(dev, 4),
            "current":     "even" if current_is_even else "odd",
            "reason":      "TRANSITION_WEAK",
        })

    direction = "EVEN" if p_even_next > S.TRUE_PROB else "ODD"
    conf      = float(np.clip(
        (dev - S.TRANSITION_MIN_DEV) / (0.15 - S.TRANSITION_MIN_DEV + 1e-6),
        0.0, 1.0))

    return ModelResult(name, True, conf, direction, {
        "p_even_next": round(p_even_next, 4),
        "p_odd_next":  round(p_odd_next, 4),
        "dev":         round(dev, 4),
        "current":     "even" if current_is_even else "odd",
    })


def model_B2_runs(prices: np.ndarray) -> ModelResult:
    """
    Wald-Wolfowitz runs test adapted for even/odd sequence.

    Detects whether the sequence is:
    - Clustering  (fewer runs than random) → bet CONTINUATION
      e.g. EEEEOOO → next is likely same as current
    - Alternating (more runs than random)  → bet REVERSAL
      e.g. EOEOEO → next is likely opposite of current

    Both patterns are exploitable, just in different directions.
    """
    name = "runs"
    fail = ModelResult(name, False, 0.0, None)

    if len(prices) < S.RUN_WINDOW + 1:
        return fail

    binary = prices_to_binary(prices[-S.RUN_WINDOW:])
    n      = len(binary)
    n1     = int(np.sum(binary))      # count of even
    n2     = n - n1                   # count of odd

    if n1 == 0 or n2 == 0:
        return fail

    # Count runs
    n_runs = 1 + int(np.sum(binary[:-1] != binary[1:]))

    # Expected runs and variance under H0
    mu_r  = 2.0 * n1 * n2 / n + 1.0
    var_r = (2.0 * n1 * n2 * (2.0 * n1 * n2 - n) /
             (n**2 * (n - 1) + 1e-8))
    if var_r <= 0:
        return fail

    z_runs = (n_runs - mu_r) / float(np.sqrt(var_r))

    if abs(z_runs) < S.RUN_Z_MIN:
        return ModelResult(name, False, 0.0, None, {
            "z_runs":  round(z_runs, 3),
            "n_runs":  n_runs,
            "mu_runs": round(mu_r, 2),
            "reason":  f"RUNS_RANDOM |{z_runs:.2f}|<{S.RUN_Z_MIN}",
        })

    # Current digit type
    current_is_even = is_even(extract_digit(float(prices[-1])))

    if z_runs < 0:
        # Clustering — bet continuation
        direction = "EVEN" if current_is_even else "ODD"
        pattern   = "CLUSTERING"
    else:
        # Alternating — bet reversal
        direction = "ODD" if current_is_even else "EVEN"
        pattern   = "ALTERNATING"

    conf = float(np.clip(
        (abs(z_runs) - S.RUN_Z_MIN) / (3.0 - S.RUN_Z_MIN + 1e-6),
        0.0, 1.0))

    return ModelResult(name, True, conf, direction, {
        "z_runs":  round(z_runs, 3),
        "n_runs":  n_runs,
        "mu_runs": round(mu_r, 2),
        "pattern": pattern,
        "current": "even" if current_is_even else "odd",
    })


def model_B3_positional(prices: np.ndarray) -> ModelResult:
    """
    Positional bias: P(even at position +5 | last 3 digits).

    This is the core Option B insight. Instead of asking "is even
    generally more common?", it asks: "given that the last 3 digits
    were [even, odd, even], what was the digit 5 ticks later,
    historically speaking?"

    Builds a lookup table from POSITIONAL_WINDOW digits of history:
      context = tuple of last 3 even/odd values
      for each context, count how often position +5 was even vs odd

    Requires POSITIONAL_MIN_N observations per context to be reliable.
    Requires P(even|context) > POSITIONAL_MIN_PROB to vote.
    """
    name = "positional"
    fail = ModelResult(name, False, 0.0, None)

    k       = S.POSITIONAL_LOOKBACK
    horizon = S.EXPIRY_TICKS     # = 5

    if len(prices) < S.POSITIONAL_WINDOW + horizon + k:
        return fail

    binary = prices_to_binary(prices[-S.POSITIONAL_WINDOW:])
    n      = len(binary)

    # Build conditional frequency table
    table = {}   # context_tuple → [even_count, total_count]
    for i in range(k, n - horizon):
        context = tuple(binary[i - k:i])
        outcome = binary[i + horizon - 1]   # digit at position +5
        if context not in table:
            table[context] = [0, 0]
        table[context][0] += outcome   # even count
        table[context][1] += 1         # total count

    # Current context
    if len(binary) < k:
        return fail
    current_context = tuple(binary[-k:])

    if current_context not in table:
        return ModelResult(name, False, 0.0, None, {
            "context": current_context,
            "reason":  "CONTEXT_UNSEEN",
        })

    even_count, total = table[current_context]
    if total < S.POSITIONAL_MIN_N:
        return ModelResult(name, False, 0.0, None, {
            "context": current_context,
            "total":   total,
            "reason":  f"INSUFFICIENT_OBS {total}<{S.POSITIONAL_MIN_N}",
        })

    p_even = even_count / total
    p_odd  = 1.0 - p_even

    if max(p_even, p_odd) < S.POSITIONAL_MIN_PROB:
        return ModelResult(name, False, 0.0, None, {
            "context": current_context,
            "p_even":  round(p_even, 4),
            "total":   total,
            "reason":  f"P_WEAK {max(p_even,p_odd):.3f}<{S.POSITIONAL_MIN_PROB}",
        })

    direction = "EVEN" if p_even >= p_odd else "ODD"
    prob      = max(p_even, p_odd)
    conf      = float(np.clip(
        (prob - S.POSITIONAL_MIN_PROB) / (0.70 - S.POSITIONAL_MIN_PROB + 1e-6),
        0.0, 1.0))

    return ModelResult(name, True, conf, direction, {
        "context":   current_context,
        "p_even":    round(p_even, 4),
        "p_odd":     round(p_odd, 4),
        "n_obs":     total,
        "n_even":    even_count,
    })


# ─────────────────────────────────────────────────────────────────
# CONFLUENCE ENGINE
# ─────────────────────────────────────────────────────────────────

def _layer_direction(models: dict, required: int) -> Tuple[Optional[str], int, list]:
    """
    Find majority direction among passing models.
    Returns (direction, score, reasons).
    """
    passing = {k: v for k, v in models.items() if v.tradeable}
    score   = len(passing)
    reasons = [
        f"{k}:{'OK' if v.tradeable else 'NO'}"
        f"({v.direction[0] if v.direction else '-'},{v.confidence:.2f})"
        for k, v in models.items()
    ]

    if score < required:
        return None, score, reasons

    even_votes = sum(1 for v in passing.values() if v.direction == "EVEN")
    odd_votes  = sum(1 for v in passing.values() if v.direction == "ODD")

    if even_votes == 0 and odd_votes == 0:
        return None, score, reasons

    # Require clear majority — can't be a tie
    if even_votes == odd_votes:
        return None, score, reasons

    # All agreeing models must point same way
    direction = "EVEN" if even_votes > odd_votes else "ODD"
    agreeing  = sum(1 for v in passing.values() if v.direction == direction)

    if agreeing < required:
        return None, score, reasons

    return direction, score, reasons


def evaluate(prices: np.ndarray) -> EOSignal:
    """
    Full two-layer evaluation.

    LAYER 1 (A1+A2+A3): Is there a digit distribution bias right now?
    LAYER 2 (B1+B2+B3): Does the sequence predict the 5th-next digit?

    Both layers must agree on direction.
    """
    def skip(reason, l1=0, l2=0, reasons=None):
        return EOSignal(False, None, 0.0, l1, l2,
                        reasons or [], skip_reason=reason)

    # ── Entropy gate ──────────────────────────────────────────────
    ok, reason = entropy_gate(prices)
    if not ok:
        return skip(reason)

    # ── Layer 1 ───────────────────────────────────────────────────
    mA1 = model_A1_frequency(prices)
    mA2 = model_A2_zscore(prices)
    mA3 = model_A3_chisquare(prices)

    l1_models   = {"freq_bias": mA1, "zscore": mA2, "chisquare": mA3}
    l1_dir, l1_score, l1_reasons = _layer_direction(l1_models, S.LAYER1_REQUIRED)

    if l1_dir is None:
        return skip(
            f"LAYER1_WEAK {l1_score}/{len(l1_models)}",
            l1=l1_score, reasons=l1_reasons
        )

    # ── Layer 2 ───────────────────────────────────────────────────
    mB1 = model_B1_transition(prices)
    mB2 = model_B2_runs(prices)
    mB3 = model_B3_positional(prices)

    l2_models   = {"transition": mB1, "runs": mB2, "positional": mB3}
    l2_dir, l2_score, l2_reasons = _layer_direction(l2_models, S.LAYER2_REQUIRED)

    if l2_dir is None:
        return skip(
            f"LAYER2_WEAK {l2_score}/{len(l2_models)}",
            l1=l1_score, l2=l2_score,
            reasons=l1_reasons + l2_reasons
        )

    # ── Direction agreement ───────────────────────────────────────
    if l1_dir != l2_dir:
        return skip(
            f"LAYERS_DISAGREE L1={l1_dir} L2={l2_dir}",
            l1=l1_score, l2=l2_score,
            reasons=l1_reasons + l2_reasons
        )

    direction = l1_dir

    # ── Confidence ────────────────────────────────────────────────
    l1_passing = [v for v in l1_models.values() if v.tradeable and v.direction == direction]
    l2_passing = [v for v in l2_models.values() if v.tradeable and v.direction == direction]

    l1_conf = float(np.mean([v.confidence for v in l1_passing])) if l1_passing else 0.0
    l2_conf = float(np.mean([v.confidence for v in l2_passing])) if l2_passing else 0.0

    # Layer 1 weighted 55% (distribution context), Layer 2 weighted 45% (sequence)
    confidence = round(float(0.55 * l1_conf + 0.45 * l2_conf), 4)

    all_confs  = [v.confidence for v in l1_passing + l2_passing]
    top_conf   = max(all_confs) if all_confs else 0.0

    if confidence < S.CONFIDENCE_MIN:
        return skip(
            f"LOW_CONF {confidence:.3f}<{S.CONFIDENCE_MIN}",
            l1=l1_score, l2=l2_score,
            reasons=l1_reasons + l2_reasons
        )

    if top_conf < S.TOP_CONF_MIN:
        return skip(
            f"NO_STRONG_MODEL top={top_conf:.3f}<{S.TOP_CONF_MIN}",
            l1=l1_score, l2=l2_score,
            reasons=l1_reasons + l2_reasons
        )

    all_reasons = l1_reasons + l2_reasons

    return EOSignal(
        tradeable    = True,
        direction    = direction,
        confidence   = confidence,
        layer1_score = l1_score,
        layer2_score = l2_score,
        reasons      = all_reasons,
        models       = {
            **{k: v.detail for k, v in l1_models.items()},
            **{k: v.detail for k, v in l2_models.items()},
        }
    )
