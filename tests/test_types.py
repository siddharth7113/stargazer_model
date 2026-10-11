"""Unit tests for stargazer.types: the Prediction record and the Detector protocol."""

from stargazer.types import Detector, Prediction


def test_to_dict_flattens_extra_to_top_level():
    """A results file stores one flat dict per prediction, so ``extra`` keys must sit beside the fields."""
    data = Prediction(detected=True, period=3.0, extra={"sde": 9.1}).to_dict()
    assert data == {"detected": True, "period": 3.0, "t0": None, "duration": None, "depth": None, "sde": 9.1}


def test_from_dict_keeps_unknown_keys_in_extra():
    """Reading a results file back must not drop detector statistics it does not know about."""
    pred = Prediction.from_dict({"detected": True, "period": 2.0, "sde": 8.0, "note": "x"})
    assert pred.period == 2.0 and pred.extra == {"sde": 8.0, "note": "x"}


def test_from_dict_missing_detected_means_not_detected():
    """An external predictions file that omits ``detected`` is treated as "none", never as a detection."""
    assert Prediction.from_dict({}).detected is False


def test_detector_protocol_is_runtime_checkable():
    """Any class with name, config and detect counts as a Detector without inheriting from anything."""

    class Minimal:
        name = "minimal"
        config = {}

        def detect(self, time, flux, flux_err):
            return Prediction(detected=False)

    assert isinstance(Minimal(), Detector)
    assert not isinstance(object(), Detector)
