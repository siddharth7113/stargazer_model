"""Star Gazer: a verifiable exoplanet-finding benchmark on TESS light curves.

Public API
----------
LightCurveCache
    Fetch and load TESS light curves.
select_targets
    Choose host stars with no known signal.
EpisodeSet, BuildConfig
    Build and read injection-recovery episode sets.
Detector, Prediction
    The interface any detection system implements.
BLSDetector
    The classical baseline.
evaluate, Results
    Run a detector and write a provenance-carrying results file.
score_episode, score_set
    The scoring rules, usable directly for RL rewards.
"""

__version__ = "0.1.0"

from .baseline import BLSDetector
from .data import LightCurve, LightCurveCache
from .episodes import BuildConfig, EpisodeSet
from .evaluate import Results, evaluate
from .inject import TransitTruth, inject, make_truth, sample_truth
from .score import score_episode, score_set
from .targets import load_targets, save_targets, select_targets
from .types import Detector, Prediction

__all__ = [
    "BLSDetector",
    "BuildConfig",
    "Detector",
    "EpisodeSet",
    "LightCurve",
    "LightCurveCache",
    "Prediction",
    "Results",
    "TransitTruth",
    "evaluate",
    "inject",
    "load_targets",
    "make_truth",
    "sample_truth",
    "save_targets",
    "score_episode",
    "score_set",
    "select_targets",
]
