"""Star Gazer: a verifiable exoplanet-finding benchmark on TESS light curves.

Terms
-----
These terms have one meaning throughout the package.

light curve
    A time series of a star's brightness. In this package, a light curve is TESS
    2-minute data with arrays ``time`` (days), ``flux`` (normalised to a median of
    one) and ``flux_err``.
transit
    A periodic dip in a light curve when a planet crosses the star. Its parameters
    are period, epoch (time of one dip centre), depth and duration.
injection
    Multiplying a real light curve by a synthetic transit model with known
    parameters. This creates an exact answer key for a real noise background.
episode
    One light curve that a detection system analyses, with or without an injected
    transit. It is one row of the benchmark and one rollout of the environment.
truth
    The hidden answer for an episode: whether a transit was injected, and with what
    parameters. Only the scorer reads it.
episode set
    A directory of episodes with one truth file, one config file and one manifest
    of file hashes. It is the unit of evaluation.
detector
    Any object that analyses one episode and returns a prediction. The BLS baseline
    is one; a language-model agent is another.
prediction
    What a detector returns for one episode: detected or not, and the claimed
    period, epoch, depth and duration.
results
    The file ``evaluate`` writes: each prediction, each score and the provenance
    needed to reproduce the run.

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
ToolAgentDetector, ChatBackend
    A language-model agent with analysis tools, and the provider interface.
evaluate, Results
    Run a detector and write a provenance-carrying results file.
score_episode, score_set
    The scoring rules, usable directly for RL rewards.
"""

__version__ = "0.1.0"

from stargazer.agent import ChatBackend, ToolAgentDetector
from stargazer.baseline import BLSDetector
from stargazer.data import LightCurve, LightCurveCache
from stargazer.episodes import BuildConfig, EpisodeSet
from stargazer.evaluate import Results, evaluate
from stargazer.inject import TransitTruth, inject, make_truth, sample_truth
from stargazer.score import score_episode, score_set
from stargazer.targets import load_targets, save_targets, select_targets
from stargazer.types import Detector, Prediction

__all__ = [
    "BLSDetector",
    "BuildConfig",
    "ChatBackend",
    "Detector",
    "EpisodeSet",
    "LightCurve",
    "LightCurveCache",
    "Prediction",
    "Results",
    "ToolAgentDetector",
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
