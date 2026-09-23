from pathlib import Path

import numpy as np
import pytest
import torch

from beat_this.model.dbn import DBNDownBeatTrackingProcessor
from beat_this.model.postprocessor import Postprocessor

FPS = 50
# reference outputs of madmom, see generate_dbn_fixture.py
FIXTURE = np.load(Path(__file__).parent / "dbn_fixture.npz")
SYNTHETIC_CASES = sorted(
    key[len("synthetic_") : -len("_act")]
    for key in FIXTURE.files
    if key.startswith("synthetic_") and key.endswith("_act")
)


def assert_times_close(actual, expected):
    """Assert that beat times match to within one frame."""
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1 / FPS + 1e-9)


def test_postprocessor_dbn_matches_madmom():
    postprocessor = Postprocessor(type="dbn", fps=FPS)
    beat, downbeat = postprocessor(
        torch.from_numpy(FIXTURE["model_beat_logits"]),
        torch.from_numpy(FIXTURE["model_downbeat_logits"]),
    )
    assert_times_close(beat, FIXTURE["model_beat"])
    assert_times_close(downbeat, FIXTURE["model_downbeat"])


@pytest.mark.parametrize("case", SYNTHETIC_CASES)
def test_dbn_matches_madmom(case):
    dbn = DBNDownBeatTrackingProcessor(
        beats_per_bar=[3, 4],
        min_bpm=55.0,
        max_bpm=215.0,
        fps=FPS,
        transition_lambda=100,
    )
    out = dbn(FIXTURE[f"synthetic_{case}_act"])
    expected = FIXTURE[f"synthetic_{case}_out"]
    assert out.shape == expected.shape
    assert_times_close(out[:, 0], expected[:, 0])
    np.testing.assert_array_equal(out[:, 1], expected[:, 1])
