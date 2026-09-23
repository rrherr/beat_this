"""
Generates tests/dbn_fixture.npz, the reference outputs of madmom's
DBNDownBeatTrackingProcessor that test_dbn.py compares the vendored DBN with.
Only needed to regenerate the fixture; requires madmom
(pip install git+https://github.com/CPJKU/madmom.git).

Run from the repository root: python tests/generate_dbn_fixture.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf
from madmom.features.downbeats import DBNDownBeatTrackingProcessor

from beat_this.inference import Audio2Frames
from beat_this.model.postprocessor import Postprocessor

FPS = 50
DBN_KWARGS = dict(
    beats_per_bar=[3, 4],
    min_bpm=55.0,
    max_bpm=215.0,
    fps=FPS,
    transition_lambda=100,
)


def synthetic_activations(rng, segments, noise=0.05):
    """
    Peaky beat and downbeat activations for a sequence of
    (num_frames, bpm, beats_per_bar) segments, in the combined format passed
    to the DBN by the Postprocessor.
    """
    beat, downbeat = [], []
    for num_frames, bpm, meter in segments:
        period = FPS * 60 / bpm
        phase = (np.arange(num_frames) / period) % meter
        dist = np.minimum(phase % 1, 1 - phase % 1) * period
        b = np.exp(-0.5 * (dist / 1.5) ** 2)
        d = b * ((phase < 0.5) | (phase > meter - 0.5))
        beat.append(b)
        downbeat.append(d)
    beat = np.concatenate(beat)
    downbeat = np.concatenate(downbeat)
    beat = np.clip(0.9 * beat + noise * rng.random(len(beat)), 1e-5, 1 - 1e-5)
    downbeat = np.clip(0.9 * downbeat + noise * rng.random(len(beat)), 1e-5, 1 - 1e-5)
    return np.vstack((np.maximum(beat - downbeat, 5e-6), downbeat)).T


def main():
    dbn = DBNDownBeatTrackingProcessor(**DBN_KWARGS)
    fixture = {}

    # real model predictions on the test audio, postprocessed with madmom
    audio, sr = sf.read(Path("tests/It Don't Mean A Thing - Kings of Swing.mp3"))
    beat_logits, downbeat_logits = Audio2Frames(device="cpu")(audio, sr)
    postprocessor = Postprocessor(type="dbn", fps=FPS)
    postprocessor.dbn = dbn
    beat, downbeat = postprocessor(beat_logits, downbeat_logits)
    fixture["model_beat_logits"] = beat_logits.numpy()
    fixture["model_downbeat_logits"] = downbeat_logits.numpy()
    fixture["model_beat"] = beat
    fixture["model_downbeat"] = downbeat

    # synthetic activations covering edge cases
    rng = np.random.default_rng(0)
    cases = {
        "steady_4_4": [(1500, 120, 4)],
        "steady_3_4": [(1500, 90, 3)],
        "tempo_change": [(750, 100, 4), (750, 140, 4)],
        "meter_change": [(750, 120, 3), (750, 120, 4)],
        "extreme_tempi": [(500, 60, 4), (500, 200, 4)],
        "noisy": [(1500, 128, 4)],
        "very_short": [(30, 120, 4)],
    }
    for name, segments in cases.items():
        noise = 0.4 if name == "noisy" else 0.05
        act = synthetic_activations(rng, segments, noise=noise)
        fixture[f"synthetic_{name}_act"] = act
        fixture[f"synthetic_{name}_out"] = dbn(act)
    # activations below the threshold everywhere
    act = np.full((200, 2), 0.01)
    fixture["synthetic_silent_act"] = act
    fixture["synthetic_silent_out"] = dbn(act)

    np.savez_compressed("tests/dbn_fixture.npz", **fixture)
    for key, value in fixture.items():
        print(key, value.shape)


if __name__ == "__main__":
    main()
