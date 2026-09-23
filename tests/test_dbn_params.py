import pytest

import beat_this.cli as cli
import beat_this.inference as inference
from beat_this.model import dbn
from beat_this.model.postprocessor import Postprocessor


@pytest.fixture
def dbn_kwargs(monkeypatch):
    """Records the keyword arguments the DBN is created with."""
    recorded = {}

    class FakeDBN:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    monkeypatch.setattr(dbn, "DBNDownBeatTrackingProcessor", FakeDBN)
    return recorded


def test_postprocessor_defaults(dbn_kwargs):
    Postprocessor(type="dbn")
    assert dbn_kwargs == dict(
        beats_per_bar=(3, 4),
        min_bpm=55.0,
        max_bpm=215.0,
        fps=50,
        transition_lambda=100,
    )


def test_postprocessor_custom_params(dbn_kwargs):
    Postprocessor(type="dbn", beats_per_bar=[4], min_bpm=80, max_bpm=160)
    assert dbn_kwargs["beats_per_bar"] == (4,)
    assert dbn_kwargs["min_bpm"] == 80.0
    assert dbn_kwargs["max_bpm"] == 160.0


def test_postprocessor_single_beats_per_bar(dbn_kwargs):
    Postprocessor(type="dbn", beats_per_bar=3)
    assert dbn_kwargs["beats_per_bar"] == (3,)


def test_postprocessor_custom_params_used():
    postprocessor = Postprocessor(type="dbn", beats_per_bar=(2, 4))
    assert [hmm.num_beats for hmm in postprocessor.dbn.hmms] == [2, 4]


@pytest.mark.parametrize(
    "kwargs", [dict(beats_per_bar=(4,)), dict(min_bpm=80.0), dict(max_bpm=160.0)]
)
def test_postprocessor_params_require_dbn(kwargs):
    with pytest.raises(ValueError, match="only be set when using the DBN"):
        Postprocessor(type="minimal", **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(beats_per_bar=()),
        dict(beats_per_bar=(0, 4)),
        dict(beats_per_bar=(3.5,)),
        dict(beats_per_bar=(True,)),
        dict(min_bpm=0.0),
        dict(min_bpm=-10.0),
        dict(min_bpm=120.0, max_bpm=120.0),
        dict(min_bpm=200.0, max_bpm=100.0),
        dict(max_bpm=10000.0),
    ],
)
def test_postprocessor_invalid_params(kwargs):
    with pytest.raises(ValueError):
        Postprocessor(type="dbn", **kwargs)


def test_audio2beats_passes_params(monkeypatch, dbn_kwargs):
    # skip loading the model
    monkeypatch.setattr(inference.Audio2Frames, "__init__", lambda self, *a: None)
    inference.File2Beats(dbn=True, beats_per_bar=(4,), min_bpm=80.0, max_bpm=160.0)
    assert dbn_kwargs["beats_per_bar"] == (4,)
    assert dbn_kwargs["min_bpm"] == 80.0
    assert dbn_kwargs["max_bpm"] == 160.0


def test_audio2beats_params_require_dbn():
    # fails before loading the model
    with pytest.raises(ValueError, match="only be set when using the DBN"):
        inference.Audio2Beats(beats_per_bar=(4,))


@pytest.fixture
def file2file_kwargs(monkeypatch):
    """Records the keyword arguments of File2File in the command line tool."""
    recorded = {}

    class FakeFile2File:
        def __init__(self, *args, **kwargs):
            recorded.update(kwargs)

        def __call__(self, audio_path, output_path):
            pass

    monkeypatch.setattr(cli, "File2File", FakeFile2File)
    return recorded


def test_cli_passes_params(file2file_kwargs, tmp_path):
    cli.main(
        [
            str(tmp_path / "song.mp3"),
            "--dbn",
            "--beats-per-bar",
            "3,4,7",
            "--min-bpm",
            "80",
            "--max-bpm",
            "160.5",
        ]
    )
    assert file2file_kwargs == dict(
        beats_per_bar=(3, 4, 7), min_bpm=80.0, max_bpm=160.5
    )


def test_cli_defaults(file2file_kwargs, tmp_path):
    cli.main([str(tmp_path / "song.mp3"), "--dbn"])
    assert file2file_kwargs == dict(beats_per_bar=None, min_bpm=None, max_bpm=None)


@pytest.mark.parametrize(
    "args, message",
    [
        (["--beats-per-bar", "4"], "--beats-per-bar can only be used with --dbn"),
        (
            ["--min-bpm", "80", "--max-bpm", "90"],
            "--min-bpm, --max-bpm can only be used with --dbn",
        ),
        (["--dbn", "--beats-per-bar", "3,x"], "expected comma-separated integers"),
        (["--dbn", "--beats-per-bar", "0"], "beats_per_bar must be"),
        (["--dbn", "--min-bpm", "200", "--max-bpm", "100"], "0 < min_bpm < max_bpm"),
    ],
)
def test_cli_errors(file2file_kwargs, tmp_path, capsys, args, message):
    with pytest.raises(SystemExit) as exc_info:
        cli.main([str(tmp_path / "song.mp3")] + args)
    assert exc_info.value.code == 2
    assert message in capsys.readouterr().err
    assert not file2file_kwargs
