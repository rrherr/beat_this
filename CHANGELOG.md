# Changelog

All notable changes to this project are documented below.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- Vendored madmom's DBN postprocessing with a pure NumPy implementation, so `--dbn` / `dbn=True` no longer requires installing madmom
- DBN postprocessing: Support configuring beats per bar and tempo range (`--beats-per-bar`, `--min-bpm`, `--max-bpm` on the command line; `beats_per_bar`, `min_bpm`, `max_bpm` in the Python API)

## [1.1.0] - 2026-04-14

- Clarified installation instructions for madmom and mir_eval
- Load checkpoints with `weights_only=True` when supported
- Fix checkpoint downloads after server-side update
- Provide separate `infer_beat_numbers()` function
- Command-line tool: Support saving raw activations / logits
- Training script: Support resuming from previous checkpoint
- Migrate to pyproject.toml (thanks to @JacobLinCool)
- Support non-CUDA accelerator chips (thanks to @tillt)
- Published on PyPI (thanks to @MarvinSchenkel)

## [1.0] - 2024-10-18

- Initial release
