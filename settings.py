"""
R_10 EVEN/ODD 5-TICK BOT — SETTINGS
=======================================
Contract : DIGITEVEN / DIGITODD on R_10
Expiry   : 5 ticks
Symbol   : R_10 (Volatility 10 Index)

TRUE PROBABILITY:
  Even (digits 0,2,4,6,8) = 0.50 exactly
  Odd  (digits 1,3,5,7,9) = 0.50 exactly

BREAK-EVEN WIN RATE:
  At 87% payout: 1/(1+0.87) = 53.5%
  At 85% payout: 1/(1+0.85) = 54.1%
  Models must find edge above this threshold.

TWO-LAYER ARCHITECTURE:
  Layer 1 (Option A) — Distribution context (100-150 digit window):
    Detects whether the digit stream is currently biased toward
    even or odd. Statistical approach — frequency + significance.

  Layer 2 (Option B) — Sequence prediction (5-30 digit window):
    Given the last few digits, predicts what the 5th-next digit
    is likely to be based on transition patterns and run structure.

  Both layers must agree on direction before a trade fires.
  Layer 1 answers: IS there a bias right now?
  Layer 2 answers: WHAT does the sequence say comes next?
"""

import os

# ── Connection ────────────────────────────────────────────────────
DERIV_APP_ID    = "1089"
DERIV_API_TOKEN = os.environ.get("DERIV_API_TOKEN", "")
DERIV_WS_URL    = "wss://ws.derivws.com/websockets/v3"

# ── Contract ──────────────────────────────────────────────────────
SYMBOL          = "R_10"
EXPIRY_TICKS    = 5
EXPIRY_UNIT     = "t"
EVEN_DIGITS     = {0, 2, 4, 6, 8}
ODD_DIGITS      = {1, 3, 5, 7, 9}
TRUE_PROB       = 0.50

# ── Tick / digit buffer ───────────────────────────────────────────
TICK_BUFFER     = 1000     # raw prices
DIGIT_BUFFER    = 500      # extracted last digits
WARMUP_TICKS    = 150      # minimum before any model votes

# ─────────────────────────────────────────────────────────────────
# LAYER 1 — DISTRIBUTION CONTEXT  (Option A)
# Statistical detection of even/odd bias over medium window
# ─────────────────────────────────────────────────────────────────

# ── Pre-filter: Shannon Entropy Gate ─────────────────────────────
# Binary entropy of even/odd sequence. Max = 1.0 bit (uniform).
# If entropy too high → distribution is flat → no exploitable edge.
ENTROPY_WINDOW      = 60       # digits
ENTROPY_MAX         = 0.985    # block if above this (near-uniform)

# ── Model A1: Recency-Weighted Frequency Bias ─────────────────────
# Exponentially weighted even/odd rate over FREQ_WINDOW digits.
# Recent digits count more (halflife = FREQ_HALFLIFE digits).
FREQ_WINDOW         = 120      # digits
FREQ_HALFLIFE       = 20       # recency decay halflife
FREQ_BIAS_MIN       = 0.030    # minimum |weighted_even - 0.50| to vote

# ── Model A2: Z-Score Significance ───────────────────────────────
# Tests whether observed even rate significantly deviates from 0.50.
# Uses multiple window sizes and picks the most significant.
ZSCORE_WINDOWS      = [30, 50, 80, 120]
ZSCORE_MIN          = 1.8      # |Z| must exceed to vote
ZSCORE_STRONG       = 2.5      # above this = strong signal

# ── Model A3: Chi-Square Consistency ─────────────────────────────
# Tests full digit distribution (0-9) for non-uniformity.
# Then checks whether the non-uniformity favors even or odd digits.
CHISQ_WINDOW        = 150      # digits
CHISQ_PVALUE_MAX    = 0.15     # reject uniform if p < this

# ─────────────────────────────────────────────────────────────────
# LAYER 2 — SEQUENCE PREDICTION  (Option B)
# Predicts the 5th-next digit from current sequence patterns
# ─────────────────────────────────────────────────────────────────

# ── Model B1: Lag-1 Transition Matrix ────────────────────────────
# P(even|last_was_even), P(odd|last_was_even), etc.
# Deviation from 0.50 = exploitable transition pattern.
TRANSITION_WINDOW   = 100      # digits for matrix estimation
TRANSITION_MIN_DEV  = 0.055    # minimum deviation from 0.50 to vote

# ── Model B2: Run Analysis (Wald-Wolfowitz adapted) ──────────────
# Detects clustering (fewer runs) or alternating (more runs).
# Clustering → bet continuation. Alternating → bet reversal.
RUN_WINDOW          = 60       # digits
RUN_Z_MIN           = 1.5      # |Z| must exceed to detect pattern

# ── Model B3: Positional Bias (5-step lookahead) ─────────────────
# Core Option B insight: in the historical digit stream, given the
# last K digits, what was the digit at position +5 more often?
# Builds a conditional frequency table: P(even at +5 | last 3 digits)
POSITIONAL_WINDOW   = 200      # digits of history to analyze
POSITIONAL_LOOKBACK = 3        # how many recent digits to condition on
POSITIONAL_MIN_PROB = 0.545    # minimum P(even|context) to vote
POSITIONAL_MIN_N    = 8        # minimum observations per context

# ─────────────────────────────────────────────────────────────────
# CONFLUENCE
# ─────────────────────────────────────────────────────────────────
# Layer 1: need 2 of 3 models to agree on direction
# Layer 2: need 2 of 3 models to agree on direction
# Both layers must agree with each other
LAYER1_REQUIRED     = 2        # of 3
LAYER2_REQUIRED     = 2        # of 3
CONFIDENCE_MIN      = 0.71
TOP_CONF_MIN        = 0.75

# ─────────────────────────────────────────────────────────────────
# TRADE MANAGEMENT
# ─────────────────────────────────────────────────────────────────
FIRST_STAKE         = 0.35
MARTINGALE_FACTOR   = 1.75
MARTINGALE_AFTER    = 1        # step up after 3 consecutive losses
MARTINGALE_MAX_STEP = 3

TARGET_PROFIT       = 100.0
STOP_LOSS           = 20.0

# Short cooldown — digit contracts settle in 5 ticks (~5 seconds)
COOLDOWN_MIN        = 20       # seconds
COOLDOWN_MAX        = 40       # seconds

RECONNECT_BASE      = 3
RECONNECT_MAX       = 60

TELEGRAM_TOKEN      = os.environ.get("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID", "")

LOG_FILE            = "/tmp/r10_eo_bot.log"
TRADES_FILE         = "/tmp/r10_eo_trades.csv"
STATS_FILE          = "/tmp/r10_eo_stats.json"
